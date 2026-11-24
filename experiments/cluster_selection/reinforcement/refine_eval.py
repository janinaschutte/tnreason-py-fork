"""Fixed development validation: sampled pruning, root propagation, residual DFS."""
from copy import deepcopy
import json
from math import isfinite
import random
from time import perf_counter
from types import SimpleNamespace

import torch

from .experiment import fingerprints, instances, propagate, solve
from .learn import measure
from .train import _closed, _state


def _selection(data, indices):
    initial, _ = instances(data)
    selected = list(data.get("validation_indices", ()) if indices is None else indices)
    training = list(data.get("train_indices", ()))
    if (not selected or len(set(selected)) != len(selected)
            or any(isinstance(idx, bool) or not isinstance(idx, int)
                   or not 0 <= idx < len(initial) for idx in selected + training)):
        raise ValueError("Expected distinct, valid evaluation indices and valid training indices")
    if set(selected) & set(training) or fingerprints(data, [initial[i] for i in selected]) & fingerprints(
            data, [initial[i] for i in training]):
        raise ValueError("Validation instances overlap training instances")
    return initial, selected


def evaluate(model, data, indices=None, *, attempts=16, seconds=30., samples=8, seed=0, nodes=10000,
             progress=False, stop_requested=None):
    """Evaluate a CPU copy without changing the policy or Python/Torch RNG state.

    Sampled one-step trials share each GAC-closed root and roll back independently.
    Root propagation has its own wall limit; attempts=0 removes aggregate
    contraction/work allowances while retaining per-cluster affordability limits.
    residual plain MAC starts from its retained domains with a separate wall limit.
    """
    return _evaluate(model, data, indices, attempts, seconds, samples, seed, nodes, "trained", progress, stop_requested)


def evaluate_baseline(model, data, indices=None, *, mode="heuristic", attempts=16,
                      seconds=30., seed=0, nodes=10000, progress=False):
    """Use the same network, limits and residual search for plain MAC or structure.

    Sudoku-specific expert execution uses a separate contraction implementation;
    its existing benchmark wrapper must be reported separately.
    """
    if mode not in {"plain", "heuristic"} or attempts == 0:
        raise ValueError("Baseline mode must be plain or heuristic with positive attempts")
    return _evaluate(model, data, indices, attempts, seconds, 0, seed, nodes, mode, progress, None)


@torch.no_grad()
def _evaluate(model, data, indices, attempts, seconds, samples, seed, nodes, mode, progress, stop_requested):
    if (isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0
            or isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1
            or isinstance(samples, bool) or not isinstance(samples, int) or samples < 0
            or isinstance(seed, bool) or not isinstance(seed, int)
            or not isfinite(seconds) or seconds <= 0):
        raise ValueError("Expected positive budgets, a nonnegative sample count and an integer seed")
    if (model.network != data["network"]
            or list(model.reward_colors) != list(data.get("reward_colors", data["network"]["colors"]))):
        raise ValueError("Policy and evaluation data must have the same network and reward colors")
    initial, selected = _selection(data, indices)
    validator = deepcopy(model).cpu().eval()
    predictor, root_attempts = validator, attempts or 1_000_000_000
    if attempts == 0:
        config = {key: value for key, value in validator.config.items() if key != "max_total_work"}
        predictor = SimpleNamespace(kind="construction", network=validator.network, config=config,
            reward_colors=validator.reward_colors, proposals=lambda domains, **_: validator.proposals(
                domains, max_candidates=root_attempts, max_steps=root_attempts))
    raw = data["network"]
    base = _state({**data, "reward_colors": data.get("reward_colors", raw["colors"]),
                   "samples": [{"domains": [list(range(n)) for n in raw["dimensions"]]}]})
    network, rows, python_rng = base.constraint_network, [], random.getstate()

    def check_stop():
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Validation stop requested")

    try:
        with torch.random.fork_rng(devices=[]):
            # Only CPU sampling occurs; do not change CUDA generator states.
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            for index in selected:
                check_stop()
                closed = network.create_mac_state(initial[index])
                consistent = closed.propagate()
                sampled = dict(trials=0, useful=0, reward=0., solved=0, seconds=0.)
                if samples and consistent and not closed.is_solved():
                    sample = dict(domains=[closed.domains[color] for color in raw["colors"]])
                    began = perf_counter()
                    with _closed(base, sample, raw["colors"]):
                        before = base.snapshot_domains()
                        for _ in range(samples):
                            check_stop()
                            candidate, _, _, _ = validator.rollout(base.domains, beta=0.)
                            metrics = measure(validator, base, candidate, objective="pruning",
                                              known_satisfiable="puzzles" in data)
                            sampled["trials"] += 1
                            for key in ("useful", "reward", "solved"):
                                sampled[key] += metrics[key]
                            if base.snapshot_domains() != before:
                                raise AssertionError("A sampled validation trial changed its input domains")
                    sampled["seconds"] = perf_counter() - began
                check_stop()
                root = propagate(network, initial[index], mode, predictor, seconds=seconds,
                                 total_work=None, baseline_candidates=root_attempts, cluster_attempts=root_attempts)
                check_stop()
                residual = (dict(status="unsat", solved=False, valid=False, assignment={}, nodes=1,
                    solve_s=0., setup_s=0., work=dict(contractions=0, estimated_work=0, max_tensor_size=0,
                    budget_exhausted=False), mac_statistics={}, cluster_statistics={}, from_root_proof=True)
                    if root["status"] == "unsat" else solve(network, root["domains"], "plain", validator,
                        nodes=nodes, seconds=seconds, total_work=None))
                if root["status"] == "invalid" or residual["status"] == "invalid":
                    raise AssertionError("Validation returned an invalid solution")
                if "puzzles" in data and (root["status"] == "unsat" or residual["status"] == "unsat"):
                    raise AssertionError("Validation contradicted a known satisfiable puzzle")
                rows.append(dict(index=index, sampled=sampled, root=root, residual=residual))
                if progress:
                    print(f"{len(rows)}/{len(selected)} index {index}: root {root['status']}, "
                          f"DFS {residual['nodes']} ({residual['status']})", flush=True)
    finally:
        random.setstate(python_rng)
    complete = sum(row["residual"]["status"] in {"solved", "unsat"} for row in rows)
    totals = dict(instances=len(rows), sampled_trials=sum(row["sampled"]["trials"] for row in rows),
        sampled_useful=sum(row["sampled"]["useful"] for row in rows),
        sampled_reward=sum(row["sampled"]["reward"] for row in rows),
        sampled_solved=sum(row["sampled"]["solved"] for row in rows),
        sampled_seconds=sum(row["sampled"]["seconds"] for row in rows),
        root_solved=sum(row["root"]["solved"] for row in rows),
        root_resolved=sum(row["root"]["status"] in {"solved", "unsat"} for row in rows),
        baseline_solved=sum(row["root"]["baseline_solved"] for row in rows),
        baseline_resolved=sum(row["root"]["baseline_solved"] or row["root"]["baseline_contradiction"] for row in rows),
        mean_progress=sum(row["root"]["progress"] or 0. for row in rows) / len(rows),
        root_seconds=sum(row["root"]["solve_s"] for row in rows), dfs_complete=complete,
        dfs_nodes=sum(row["residual"]["nodes"] for row in rows) if complete == len(rows) else None,
        residual_seconds=sum(row["residual"]["solve_s"] for row in rows))
    totals["sampled_useful_fraction"] = totals["sampled_useful"] / max(1, totals["sampled_trials"])
    totals["sampled_mean_reward"] = totals["sampled_reward"] / max(1, totals["sampled_trials"])
    return json.loads(json.dumps(dict(rows=rows, totals=totals, settings=dict(mode=mode,
        attempts=attempts, time_only=attempts == 0, seconds=seconds, samples=samples, seed=seed, nodes=nodes,
        branching="MRV, maximum incident-core degree, variable name; fresh residual state",
        timing="Root and residual wall limits are separate; model copy, setup and one-step trials are excluded")),
        allow_nan=False))


def main(argv=None):
    """Evaluate adapter-aware checkpoints without using the legacy path loader."""
    import argparse
    from hashlib import sha256
    from pathlib import Path
    from .actor import load
    from .experiment import instance_hashes
    from .refine import source_hashes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; never overwritten")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--mode", choices=("trained", "plain", "heuristic"), default="trained")
    parser.add_argument("--full-mac", action="store_true", help="Use clusters throughout DFS, with one whole-solve time limit")
    parser.add_argument("--attempts", type=int, default=32, help="0: time-only learned root propagation")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=0, help="Optional independent one-step root trials")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.attempts < 0 or args.nodes < 1 or args.samples < 0
            or not isfinite(args.seconds) or args.seconds <= 0
            or (args.full_mac or args.mode != "trained") and args.attempts == 0
            or args.mode != "trained" and args.samples or args.full_mac and args.samples):
        raise ValueError("Invalid budgets: full MAC/baselines require positive attempts; full MAC excludes sampled trials")
    began = perf_counter()
    data = json.loads(args.data.read_text())
    initial, selected = _selection(data, args.indices)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    exposure = saved.get("training", {}).get("train_instance_hashes")
    if exposure is None:
        raise ValueError("Checkpoint lacks training fingerprints; evaluation disjointness cannot be checked")
    if set(exposure) & instance_hashes(data, [initial[index] for index in selected]):
        raise ValueError("Evaluation instances overlap checkpoint training instances")
    model = load(args.checkpoint)
    if (model.network != data["network"]
            or list(model.reward_colors) != list(data.get("reward_colors", data["network"]["colors"]))):
        raise ValueError("Policy and evaluation data must have the same network and reward colors")
    torch.set_num_threads(1)
    protocol = dict(data_sha256=sha256(args.data.read_bytes()).hexdigest(),
        checkpoint_sha256=sha256(args.checkpoint.read_bytes()).hexdigest(), source_sha256=source_hashes(),
        training_fingerprints=len(exposure), selected_indices=selected, actor_config=model.actor_config, config=model.config, beta=model.beta,
        load_seconds=perf_counter() - began,
        data=str(args.data.resolve()), checkpoint=str(args.checkpoint.resolve()),
        settings={**{key: value for key, value in vars(args).items() if key not in {"data", "checkpoint", "output"}},
            "branching": "MRV, maximum incident-core degree, variable name"})
    if args.full_mac:
        raw = data["network"]
        base = _state({**data, "reward_colors": data.get("reward_colors", raw["colors"]),
                       "samples": [{"domains": [list(range(n)) for n in raw["dimensions"]]}]})
        rows, python_rng = [], random.getstate()
        try:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.random.default_generator.manual_seed(args.seed)
                random.seed(args.seed)
                for index in selected:
                    row = solve(base.constraint_network, initial[index], args.mode, model, nodes=args.nodes,
                        seconds=args.seconds, total_work=None, baseline_candidates=args.attempts,
                        cluster_attempts=args.attempts)
                    if row["status"] == "invalid" or "puzzles" in data and row["status"] == "unsat":
                        raise AssertionError("Full MAC returned invalid pruning or an invalid solution")
                    rows.append(dict(index=index, **row))
                    print(f"{len(rows)}/{len(selected)} index {index}: {row['status']}, DFS {row['nodes']}", flush=True)
        finally:
            random.setstate(python_rng)
        complete = sum(row["status"] in {"solved", "unsat"} for row in rows)
        report = dict(rows=rows, totals=dict(instances=len(rows), dfs_complete=complete,
            root_solved=sum(row["solved"] and row["nodes"] == 1 for row in rows),
            root_resolved=sum(row["status"] in {"solved", "unsat"} and row["nodes"] == 1 for row in rows),
            dfs_nodes=sum(row["nodes"] for row in rows) if complete == len(rows) else None,
            solve_seconds=sum(row["solve_s"] for row in rows)),
            scope="Full MAC: clusters at each propagation closure; one wall limit for root and DFS together")
    else:
        options = dict(attempts=args.attempts, seconds=args.seconds, seed=args.seed, nodes=args.nodes)
        report = (evaluate(model, data, selected, samples=args.samples, progress=True, **options)
            if args.mode == "trained" else evaluate_baseline(model, data, selected, mode=args.mode, progress=True, **options))
        report["scope"] = "Root cluster propagation followed by separate plain MAC residual search"
    if (source_hashes() != protocol["source_sha256"]
            or sha256(args.data.read_bytes()).hexdigest() != protocol["data_sha256"]
            or sha256(args.checkpoint.read_bytes()).hexdigest() != protocol["checkpoint_sha256"]):
        raise RuntimeError("Source, dataset or checkpoint changed during evaluation")
    report["protocol"] = protocol
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["totals"], allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    main()
