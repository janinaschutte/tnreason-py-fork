"""Pilot-only fixed validation baselines; never load the reused test set."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from math import log
from pathlib import Path
import platform
import os
import sys
from tempfile import mkdtemp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def expert(data, model, plain, out, seconds, nodes):
    """Reuse the original expert evaluator with pilot-only domains and references."""
    from demonstrations.sudoku.inferenceClusters import RuleLimits, available_rules
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import instances, solve
    from experiments.cluster_selection.reinforcement.train import _state
    path = HERE.parent / "learning_diagnostics_20261113.67i11ilh/matched_time_20261115.PW6lJf/run.py"
    spec = importlib.util.spec_from_file_location("pilot_frozen_expert", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    expert_out = out / "expert_raw"
    (expert_out / "rows").mkdir(parents=True)
    references = {}
    for position, row in enumerate(plain["rows"]):
        if row["residual"]["status"] != "solved":
            raise AssertionError("Plain residual reference solve did not complete")
        reference_path = expert_out / f"reference_{position:02d}.json"
        write(reference_path, dict(assignment=row["residual"]["assignment"],
                                   source="Separate plain-MAC residual solve; never a policy input"))
        references[str(position)] = str(reference_path)
    limits = RuleLimits()
    families = list(available_rules())
    assert len(families) == 51
    plan = dict(seconds=seconds, budgets_s=[seconds], expert_limits=asdict(limits),
                expert_families=families, reference_assignments=references,
                source_sha256={str(path): digest(path)}, pilot_data_sha256=digest(HERE / "pilot_data.json"))
    write(expert_out / "protocol.json", plan)
    raw = data["network"]
    base = _state({**data, "samples": [{"domains": [list(range(n)) for n in raw["dimensions"]]}]})
    network = base.constraint_network
    initial = instances(data)[0]
    engine.OUT = expert_out
    engine.CONTEXT.update(plan=plan, network=network, models={}, runtime=runtime,
                          domains=[initial[i] for i in data["validation_indices"]])
    rows = []
    for position, index in enumerate(data["validation_indices"]):
        original_solve = network.solve_mac
        network.solve_mac = engine.forbid
        try:
            recorded = engine.evaluate("expert", position)
        finally:
            network.solve_mac = original_solve
        final = recorded["snapshots"][-1]["domains"]
        entropy = lambda ds: sum(log(len(ds[c])) for c in data["reward_colors"])
        start_entropy = entropy(recorded["snapshots"][0]["domains"])
        root = dict(status=recorded["status"], solved=recorded["status"] == "solved",
            valid=recorded["valid"], nodes=0, domains=final, baseline_solved=start_entropy == 0,
            progress=1 - entropy(final) / start_entropy if start_entropy else 1.,
            solve_s=recorded["wall_s"], expert_raw=str(expert_out / "rows" / f"expert_{position:02d}.json"))
        residual = solve(network, final, "plain", model, nodes=nodes, seconds=seconds)
        if residual["status"] in {"invalid", "unsat"}:
            raise AssertionError("Expert residual solve violated a known satisfiable puzzle")
        rows.append(dict(index=index, root=root, residual=residual))
    complete = sum(row["residual"]["status"] == "solved" for row in rows)
    totals = dict(instances=len(rows), root_solved=sum(row["root"]["solved"] for row in rows),
        mean_progress=sum(row["root"]["progress"] for row in rows) / len(rows),
        root_seconds=sum(row["root"]["solve_s"] for row in rows), dfs_complete=complete,
        dfs_nodes=sum(row["residual"]["nodes"] for row in rows) if complete == len(rows) else None,
        residual_seconds=sum(row["residual"]["solve_s"] for row in rows))
    return dict(rows=rows, totals=totals, settings=dict(mode="expert", seconds=seconds, nodes=nodes,
        families=families, limits=asdict(limits), note="51 Sudoku-specific families; larger cluster limits"))


def main():
    import torch
    from experiments.cluster_selection.reinforcement.actor import Policy, load
    from experiments.cluster_selection.reinforcement.refine_eval import evaluate, evaluate_baseline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = json.loads((HERE / "pilot_data.json").read_text())
    checkpoint = HERE.parent / "reinforce_sampling_grow_gpu_20261106.q3mc4c0g/job_7906.0/policy.best.pt"
    initial = load(checkpoint)
    assert initial.network == data["network"]
    torch.manual_seed(0)
    untrained = Policy(data["network"], **initial.config)
    untrained.reward_colors, untrained.beta = data["reward_colors"], initial.beta
    untrained.eval()
    out = Path(mkdtemp(prefix="baselines32_", dir=HERE))
    source_paths = [Path(__file__), HERE / "pilot_data.json", HERE / "split_manifest.json", checkpoint,
        ROOT / "experiments/cluster_selection/reinforcement/refine_eval.py",
        ROOT / "experiments/cluster_selection/reinforcement/actor.py"]
    protocol = dict(methods=["plain", "expert", "heuristic", "fixed_init", "untrained"],
        settings=vars(args), seed=0, actor_config=initial.actor_config, model_config=initial.config,
        validation_indices=data["validation_indices"], source_sha256={str(p): digest(p) for p in source_paths},
        affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(), torch=torch.__version__,
        note="Existing development validation only; generic32 attempts/30s per root, separate residual plain MAC; expert larger original51family bounds")
    write(out / "protocol.json", protocol)
    print(out, flush=True)
    reports = {}
    for method in protocol["methods"]:
        print(f"START {method}", flush=True)
        if method == "expert":
            report = expert(data, initial, reports["plain"], out, args.seconds, args.nodes)
        elif method in {"plain", "heuristic"}:
            report = evaluate_baseline(initial, data, mode=method, attempts=args.attempts,
                seconds=args.seconds, seed=0, nodes=args.nodes)
        else:
            report = evaluate(initial if method == "fixed_init" else untrained, data,
                attempts=args.attempts, seconds=args.seconds, samples=args.samples, seed=0, nodes=args.nodes)
        write(out / f"{method}.json", report)
        reports[method] = report
        print(method, json.dumps(report["totals"], sort_keys=True), flush=True)
    assert all(digest(Path(path)) == value for path, value in protocol["source_sha256"].items())
    write(out / "summary.json", {method: report["totals"] for method, report in reports.items()})


if __name__ == "__main__":
    main()
