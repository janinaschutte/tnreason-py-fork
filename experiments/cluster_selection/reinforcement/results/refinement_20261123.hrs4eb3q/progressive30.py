"""Additional progressive wall-time heuristic on the same fixed pilot validation."""
import csv
from hashlib import sha256
import importlib.util
import json
from math import log
from pathlib import Path
import sys
from tempfile import mkdtemp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import torch
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.actor import load
    from experiments.cluster_selection.reinforcement.experiment import instances, solve
    from experiments.cluster_selection.reinforcement.train import _state
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    baseline = Path(sys.argv[1]).resolve()
    if not all((baseline / f"{name}.json").exists() for name in ("plain", "expert", "heuristic", "fixed_init", "untrained")):
        raise ValueError("Wait for all five fixed-attempt baselines to finish")
    helper = HERE.parent / "learning_diagnostics_20261113.67i11ilh/matched_time_20261115.PW6lJf/run.py"
    spec = importlib.util.spec_from_file_location("pilot_progressive_helper", helper)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    data = json.loads((HERE / "pilot_data.json").read_text())
    expert_plan = json.loads((baseline / "expert_raw/protocol.json").read_text())
    checkpoint = HERE.parent / "reinforce_sampling_grow_gpu_20261106.q3mc4c0g/job_7906.0/policy.best.pt"
    model = load(checkpoint)
    out = Path(mkdtemp(prefix="progressive30_", dir=HERE))
    (out / "rows").mkdir()
    plan = {**expert_plan, "method": "heuristic_progressive_time30", "technical_iterator_bound": 2 ** 31 - 1,
        "structural_initial_quota": 32, "structural_initial_steps": 8192,
        "limits": {key: model.config[key] for key in ("max_cores", "max_open_colors", "max_work", "max_size")},
        "baseline_comparison": str(baseline), "aggregate_attempt_limit": None,
        "note": "Progressive structural32/64/... generation restarts after pruning;30s root ceiling; no aggregate contraction cap",
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out / "protocol.json", plan)
    raw = data["network"]
    base = _state({**data, "samples": [{"domains": [list(range(n)) for n in raw["dimensions"]]}]})
    network, initial = base.constraint_network, instances(data)[0]
    engine.OUT = out
    engine.CONTEXT.update(plan=plan, network=network, models={"teacher": model}, runtime=runtime,
                          domains=[initial[i] for i in data["validation_indices"]])
    rows = []
    print(out, flush=True)
    for position, index in enumerate(data["validation_indices"]):
        original_solve = network.solve_mac
        network.solve_mac = engine.forbid
        try:
            recorded = engine.evaluate("heuristic", position)
        finally:
            network.solve_mac = original_solve
        final = recorded["snapshots"][-1]["domains"]
        entropy = lambda ds: sum(log(len(ds[c])) for c in data["reward_colors"])
        start_entropy = entropy(recorded["snapshots"][0]["domains"])
        root = dict(status=recorded["status"], solved=recorded["status"] == "solved", domains=final,
            progress=1 - entropy(final) / start_entropy if start_entropy else 1., solve_s=recorded["wall_s"])
        residual = solve(network, final, "plain", model, nodes=10000, seconds=30.)
        if residual["status"] in {"invalid", "unsat"}:
            raise AssertionError("Progressive heuristic violated a known satisfiable puzzle")
        row = dict(index=index, root=root, residual=residual)
        write(out / "rows" / f"residual_{position:02d}.json", row)
        rows.append(row)
    complete = sum(row["residual"]["status"] == "solved" for row in rows)
    totals = dict(instances=len(rows), root_solved=sum(row["root"]["solved"] for row in rows),
        mean_progress=sum(row["root"]["progress"] for row in rows) / len(rows),
        root_seconds=sum(row["root"]["solve_s"] for row in rows), dfs_complete=complete,
        dfs_nodes=sum(row["residual"]["nodes"] for row in rows) if complete == len(rows) else None,
        residual_seconds=sum(row["residual"]["solve_s"] for row in rows))
    write(out / "heuristic_progressive_time30.json", dict(rows=rows, totals=totals, settings=plan))
    reports = {name: json.loads((baseline / f"{name}.json").read_text())
               for name in ("plain", "expert", "heuristic", "fixed_init", "untrained")}
    reports["heuristic_progressive_time30"] = dict(rows=rows, totals=totals)
    summary = {name: report["totals"] for name, report in reports.items()}
    for name, report in reports.items():
        retained = [row for row in report["rows"] if row["index"] != 39]
        summary[name]["excluding_pilot39_diagnostic"] = dict(instances=len(retained),
            root_solved=sum(row["root"]["solved"] for row in retained),
            dfs_nodes=sum(row["residual"]["nodes"] for row in retained)
                if all(row["residual"]["status"] == "solved" for row in retained) else None)
    write(out / "comparison.json", dict(results=summary,
        note="All16 remain evaluation criterion. Pilot39/source21 plain158node exclusion is diagnostic only; progressive heuristic has no32attempt cap."))
    with (out / "per_puzzle.csv").open("x", newline="") as stream:
        fields = ["pilot_index", "original_index", "method", "root_solved", "residual_dfs_nodes", "root_seconds", "residual_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        mapping = data["pilot_preparation"]["original_indices"]
        for name, report in reports.items():
            for row in report["rows"]:
                writer.writerow(dict(pilot_index=row["index"], original_index=mapping[row["index"]], method=name,
                    root_solved=row["root"]["solved"], residual_dfs_nodes=row["residual"]["nodes"],
                    root_seconds=row["root"]["solve_s"], residual_seconds=row["residual"]["solve_s"]))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
