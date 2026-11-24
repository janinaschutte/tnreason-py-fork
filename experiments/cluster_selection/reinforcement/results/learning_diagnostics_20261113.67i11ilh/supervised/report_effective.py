"""Audit fixed inputs/assignments and aggregate the separately guarded rows."""
from copy import deepcopy
from pathlib import Path
import runpy
import torch

RUN = runpy.run_path(str(Path(__file__).with_name("run.py")))
OUT, read, write, digest = (RUN[name] for name in ("OUT", "read", "write", "digest"))


def aggregate(rows):
    done = [row for row in rows if row["solved"] and row["valid"]]
    return dict(instances=len(rows), complete=len(done), nodes=sum(row["nodes"] for row in done) if len(done) == len(rows) else None,
        branch_free=sum(row["nodes"] == 1 for row in done), censored=[row["puzzle"] for row in rows if not row["solved"]])


def main():
    plan = read(OUT / "protocol.json")
    for path, checksum in plan["source_sha256"].items():
        assert digest(path) == checksum, f"Frozen input changed: {path}"
    if (OUT / "effective_summary.json").exists():
        assert read(OUT / "final_audit.json")["report_source_sha256"] == digest(__file__)
        print("Effective report already saved; no files changed.")
        return
    source = read(plan["data"])
    raw = source["network"]
    colors, dimensions = raw["colors"], raw["dimensions"]
    from experiments.cluster_selection.supervised.collect import sudoku_domains
    givens = {idx: source["initial_domains"][idx] if "initial_domains" in source else sudoku_domains(source["puzzles"][idx]) for idx in plan["validation_indices"]}
    provenance, checked, replacements = {}, [], {}

    def row(path):
        value = read(path)
        provenance[str(path)] = digest(path)
        if value["solved"]:
            assignment = value["assignment"]
            assert value["valid"] and set(assignment) == set(colors)
            assert all(0 <= assignment[color] < dim for color, dim in zip(colors, dimensions))
            assert all(assignment[color] in values for color, values in givens[value["puzzle"]].items())
            for core in raw["cores"]:
                index = 0
                for color, dim in zip(core["colors"], core["shape"]):
                    index = index * dim + assignment[colors[color]]
                assert core["values"][index]
            checked.append(str(path))
        else:
            assert value["status"] in ("wall_timeout", "node_limit")
            assert value["nodes"] is None if value["status"] == "wall_timeout" else value["nodes"] >= plan["nodes"]
        return value

    for directory in sorted((OUT / "retries_1800").glob("*")):
        protocol = read(directory / "protocol.json")
        assert protocol["original_protocol_sha256"] == digest(OUT / "protocol.json")
        assert protocol["original_row_sha256"] == digest(protocol["original_row"])
        assert not read(protocol["original_row"])["solved"]
        assert protocol["checkpoint_sha256"] == digest(protocol["checkpoint"])
        assert protocol["seconds"] == 1800 and protocol["original_seconds"] == 600
        assert protocol["nodes"] == plan["nodes"] and protocol["branching"] == "name"
        assert protocol["cluster_attempts"] == 32 and protocol["decoder"] == "discrepancy"
        assert protocol["total_work"] is None and protocol["baseline_steps"] == 8192
        assert protocol["baseline_candidates"] == 32 and not protocol["stop_on_noop"] and protocol["beta"] == 0.
        assert all(protocol["source_sha256"].get(path) == checksum for path, checksum in plan["source_sha256"].items())
        for path, checksum in protocol["source_sha256"].items():
            assert digest(path) == checksum
        retry = row(directory / "row.json")
        assert retry["retry_protocol_sha256"] == digest(directory / "protocol.json")
        assert retry["puzzle"] == protocol["puzzle"]
        replacements[protocol["original_row"]] = retry
        provenance[str(directory / "protocol.json")] = digest(directory / "protocol.json")
    effective = deepcopy(read(OUT / "summary.json"))
    all_rows = {}
    for name, directory in [(f"{arm}/{update}", OUT / arm / f"dfs_{update:04d}") for arm in RUN["ARMS"] for update in (1000, 4000)] + [(name, OUT / f"baseline_{name}") for name in ("plain", "heuristic32", "parent")]:
        rows = []
        for path in sorted(directory.glob("puzzle_*.json")):
            original = row(path)
            rows.append(replacements.get(str(path), original))
        assert len(rows) == 23 and {item["puzzle"] for item in rows} == set(plan["validation_indices"])
        all_rows[name] = rows
        if "/" in name:
            arm, update = name.split("/")
            effective["arms"][arm][update]["dfs"] = aggregate(rows)
        else:
            effective["baselines"][name] = aggregate(rows)
    parent = torch.load(plan["parent"], map_location="cpu", weights_only=True)
    for arm in RUN["ARMS"]:
        for update in RUN["MILESTONES"]:
            path = OUT / arm / f"policy_{update:04d}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["training"]["protocol_sha256"] == digest(OUT / "protocol.json")
            assert saved["training"]["updates"] == update and saved["training"]["arm"] == arm
            assert [item["state"] for item in saved["history"]] == plan["update_order"][:update]
            assert read(OUT / arm / f"utility_{update:04d}.json")["checkpoint_sha256"] == digest(path)
            if update == 0:
                assert all(torch.equal(value, parent["weights"][key]) for key, value in saved["weights"].items())
            provenance[str(path)] = digest(path)
    effective.update(branching="name", no_validation_selection=True,
        guard_note="Original600-second rows retained. Only their censored observations are replaced by separately frozen1800-second retries; remaining limits stay unknown. Timings are not comparable isolated measurements.",
        original_summary_sha256=digest(OUT / "summary.json"))
    write(OUT / "effective_summary.json", effective)
    write(OUT / "final_audit.json", dict(report_source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json"),
        frozen_input_count=len(plan["source_sha256"]), all_frozen_inputs_match=True,
        checked_valid_assignments=len(checked), checked_assignments=checked, result_sha256=provenance,
        exposure_audit_sha256=digest(OUT / "exposure_audit.json"), reproduction_sha256=digest(OUT / "reproduction_0240.json"),
        note="No training or solving in this postprocessor. Every saved solved initial/retry assignment checked against all original cores, dimensions and givens; both arms' checkpoint histories match the same fixed update schedule and initial parent weights."))
    lines = ["Controlled constructive-teacher fit diagnostic", plan["note"],
        "Same parent, seed29, fresh Adam3e-4, fixed24 states, identical update order. All legal core orders remain available.",
        "Common branching: MRV, maximum core degree, then name. Allowance32; discrepancy decoder; no cumulative contraction-work cap.",
        effective["guard_note"], "Arm/update | useful train/24 | roots/13 | branches/11 | useful validation roots/15 | validation DFS nodes"]
    for arm, values in effective["arms"].items():
        for update, value in values.items():
            lines.append(f"{arm}/{update} | {value['train']['all']['useful']} | {value['train']['root']['useful']} | {value['train']['branch']['useful']} | {value['validation']['all']['useful']} | {value.get('dfs', {}).get('nodes', 'not evaluated')}")
    lines.append("Fresh baselines: " + str(effective["baselines"]))
    lines.extend(["The pool arm exactly reproduces all240 earlier loss/gradient/state records before continuing.",
        "Longer training improves free construction on these fixed training states, but fit remains limited. The single-teacher comparison changes both teacher multiplicity and length; it does not isolate their separate effects.",
        "Validation was examined previously and supplies no teacher gradients here. No final test, checkpoint selection or production supervised/mask change."])
    with (OUT / "REPORT_EFFECTIVE.txt").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print("audited", len(checked), "valid assignments; effective baselines", effective["baselines"])


if __name__ == "__main__":
    main()
