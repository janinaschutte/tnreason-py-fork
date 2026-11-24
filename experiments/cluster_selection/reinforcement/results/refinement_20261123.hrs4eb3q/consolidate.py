"""Consolidate completed development evaluations; no solver or training is run."""
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp

from analyze import compact, paired, BOOTSTRAPS, BOOTSTRAP_SEED

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    common = BASE / "common_v2_2wlie6na"
    references = BASE / "baselines32_wgk4zatg"
    shared16 = dict(plain=references / "plain.json", expert=references / "expert.json")
    scopes = {
        "development16_fixed32": dict(**shared16, heuristic=references / "heuristic.json",
            **{name: common / f"{filename}.json" for name, filename in dict(parent="parent", untrained="untrained",
                rms="rms_seed0", group0="group_seed0", group1="group_seed1", global_context="global_seed0",
                objective_rms="objective_rms_seed0", objective_group="objective_group_seed0").items()}),
        "development55_fixed32": dict(plain=BASE / "full55_references_watyom61/plain.json",
            expert=BASE / "full55_references_watyom61/expert.json",
            heuristic=BASE / "full55_heuristic32_gv2r9ug_/result.json",
            parent=BASE / "full55_q13qt7s1/parent.json", global_context=BASE / "full55_q13qt7s1/global_seed0.json",
            objective_rms=BASE / "full55_objective_rms_seed0_2s1qqgfn/result.json",
            untrained=BASE / "full55_random_216gsslg/untrained.json",
            prior_gpu_replay=BASE / "prior_gpu_replay_arli8b26/full55.json"),
        "development16_time30": dict(**shared16,
            heuristic=BASE / "progressive30_78ufubt4/heuristic_progressive_time30.json",
            prior_gpu_replay=BASE / "prior_gpu_replay_arli8b26/time16_recovered.json"),
        "development16_serial_time30_v4": dict(**shared16,
            heuristic=BASE / "progressive30_78ufubt4/heuristic_progressive_time30.json",
            prior_gpu_replay=BASE / "serial_time30_c8uq_f33/prior_gpu.json",
            untrained=BASE / "serial_time30_c8uq_f33/untrained.json",
            global_rms_seed0=BASE / "serial_time30_c8uq_f33/global_rms.json"),
    }
    assert read(BASE / "serial_time30_c8uq_f33/summary.json")["audit"]["sources_inputs_checkpoints_unchanged"]
    pending = []
    for name in ("context075_seed0", "context050_seed0", "global_rms_seed0"):
        path, gate = BASE / name / "validation_000002048.json", BASE / f"{name}_after.json"
        if path.exists() and gate.exists():
            assert read(gate)["complete"] and read(gate)["result"]["trials"] == 2048
            scopes["development16_fixed32"][name] = path
        else:
            pending.append(f"fixed32/{name}")
        completed = []
        for candidate in BASE.glob(f"time30_{name}_*/result.json"):
            audit = candidate.parent / "audit.json"
            if audit.exists():
                assert read(audit)["sources_and_inputs_unchanged"] and read(audit)["retained_references"] == 16
                completed.append(candidate)
        if len(completed) > 1:
            raise ValueError(f"Multiple complete {name} time-only runs; select a protocol explicitly")
        if completed:
            scopes["development16_time30"][name] = completed[0]
        else:
            pending.append(f"time30/{name}")
    for filename in ("full55_summary.json", "time16_summary.json"):
        receipt = read(BASE / "prior_gpu_replay_arli8b26" / filename)
        valid = receipt.get("source_inputs_checkpoint_unchanged", receipt.get("sources_inputs_checkpoint_unchanged"))
        assert valid and receipt["totals"]["dfs_complete"] == receipt["totals"]["instances"]
    for method, label in dict(parent="parent", untrained="untrained", global_="global_context", role="objective_rms").items():
        method = method.rstrip("_")
        choices = []
        for path in BASE.glob(f"time30_{method}_*/result.json"):
            protocol, audit = path.parent / "protocol.json", path.parent / "audit.json"
            if (not audit.exists() or str(BASE / "time_roots_v2.py") not in read(protocol)["hashes"]):
                continue
            assert read(audit)["sources_and_inputs_unchanged"] and read(audit)["retained_references"] == 16
            choices.append(path)
        if len(choices) > 1:
            raise ValueError(f"Multiple complete v2 {method} time-only runs; select a protocol explicitly")
        if choices:
            scopes["development16_time30"][label] = choices[0]
        else:
            pending.append(f"time30/{method}")
    pilot_originals = read(BASE / "pilot_data.json")["pilot_preparation"]["original_indices"]
    inputs, summaries, comparisons, comparison_references, context_comparisons, rows = {}, {}, {}, {}, {}, []
    for scope, paths in scopes.items():
        count = 55 if "55" in scope else 16
        reports, normalized = {}, {}
        for name, path in paths.items():
            inputs[str(path)] = digest(path)
            report = reports[name] = read(path)
            assert len(report["rows"]) == len({row["index"] for row in report["rows"]}) == count
            originals = {row["index"]: row["index"] if count == 55 else pilot_originals[row["index"]]
                         for row in report["rows"]}
            normalized[name] = {row["index"]: compact(row, originals) for row in report["rows"]}
            totals = report["totals"]
            rows.append(dict(scope=scope, method=name, instances=count, root_solved=totals["root_solved"],
                dfs_complete=totals["dfs_complete"], dfs_nodes=totals["dfs_nodes"], root_seconds=totals["root_seconds"],
                residual_seconds=totals["residual_seconds"],
                root_deadline_hits=sum(row["root"]["status"] == "wall_timeout" or
                    ("time30" in scope and row["root"]["status"] == "resource_guard"
                     and row["root"]["solve_s"] >= 30.) for row in report["rows"]),
                sampled_trials=totals.get("sampled_trials"), sampled_useful=totals.get("sampled_useful"),
                sampled_seconds=totals.get("sampled_seconds"), source=str(path)))
        reference = "prior_gpu_replay" if scope.endswith("serial_time30_v4") else "parent"
        comparison_references[scope] = reference
        if reference not in normalized:
            comparisons[scope] = dict(pending="parent time-only endpoint")
        else:
            wanted = set(normalized[reference])
            assert all(set(item) == wanted for item in normalized.values())
            comparisons[scope] = {name: paired(item, normalized[reference]) for name, item in normalized.items() if name != reference}
        summaries[scope] = {name: report["totals"] for name, report in reports.items()}
        context_comparisons[scope] = {
            f"{name}_versus_{reference}": paired(normalized[name], normalized[reference])
            for name in ("context075_seed0", "context050_seed0", "global_rms_seed0") if name in normalized
            for reference in ("untrained", "global_context") if reference in normalized}
    out = Path(mkdtemp(prefix="consolidated_", dir=BASE))
    caveats = [
        "All evaluations reuse existing development validation; no heldout100 data or results were read.",
        "Fixed32 limits generic application attempts; time30 has no attempt cap and a 30-second root wall ceiling.",
        "DFS is plain MAC after retaining root pruning, not a full search that continues applying clusters at every DFS node.",
        "The expert uses 51 Sudoku-specific rule families and larger cluster limits; it is a reference in every table.",
        "Common fixed32 neural rows were re-evaluated serially on CPU3; reused plain/heuristic/expert timings came from earlier runs.",
        "Context and global-RMS fixed32 endpoints use their audited training-end evaluations, not the earlier common serial re-evaluation.",
        "Prior GPU replay is a separate historical checkpoint, not the small CPU replay checkpoint in the ongoing100 benchmark.",
        "Prior GPU replay time16 was recovered from complete durable rows and independently audited after a post-evaluation reference-index error.",
        "Time-only neural rows run on distinct pinned CPUs with shared-host contention; equal ceilings do not guarantee equal operations.",
        "The final serial_time30_v4 scope re-evaluates only GPU replay, untrained and global-RMS, interleaved on CPU0; other rows there are reused references.",
        "Serial v4 uses the same optimized domain-feature implementation for all three models. Its paired baseline is the stronger GPU replay checkpoint.",
        "Root and residual times exclude model loading, setup, training and separate sampled single-cluster diagnostics.",
        "Root deadline hits are expected censoring under time30; residual incompleteness is reported separately and never counted as zero nodes.",
        "Paired puzzle bootstrap intervals exclude training-seed, checkpoint selection and evaluation-seed uncertainty; all puzzles remain included.",
    ]
    result = dict(pending=pending, summaries=summaries, versus_parent=comparisons,
        comparison_references=comparison_references, context_comparisons=context_comparisons, caveats=caveats,
        bootstrap=dict(resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED), input_sha256=inputs,
        driver_sha256=digest(Path(__file__)), helper_sha256=digest(BASE / "analyze.py"))
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (out / "tables.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = []
    for scope in scopes:
        lines.extend([scope, "Method                  Roots     DFS Root(s) Residual(s) Deadline hits"])
        for row in rows:
            if row["scope"] == scope:
                lines.append(f"{row['method']:22} {row['root_solved']:2}/{row['instances']:<3} {str(row['dfs_nodes']):>5} "
                    f"{row['root_seconds']:8.2f} {row['residual_seconds']:11.2f} {row['root_deadline_hits']:13}")
        lines.append("")
    lines.extend(["Pending: " + (", ".join(pending) or "none"), "", *caveats])
    (out / "TABLES.txt").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
