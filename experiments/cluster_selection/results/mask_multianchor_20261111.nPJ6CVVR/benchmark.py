"""Sequential CPU validation sweep; predictions only, without retraining."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from time import perf_counter, process_time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement.experiment import fingerprints, instances, solve
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.workload_to_numpy import NumpyCore


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["plain", "heuristic16", "heuristic32", "heuristic64",
                                                        "mask1", "mask2", "mask4", "mask8"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=814)
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.methods)) != len(args.methods):
        raise ValueError("Distinct methods and positive repeat count required")
    allowed = {"plain", "heuristic16", "heuristic32", "heuristic64", "mask1", "mask2", "mask4", "mask8"}
    if set(args.methods) - allowed:
        raise ValueError("Unsupported method")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows").mkdir()
    (args.output / "code").mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    checkpoint = ROOT / "experiments/cluster_selection/results/mask_sudoku.FGN32oF0/train/epoch_0025.pt"
    training = checkpoint.parent
    data_path = ROOT / "experiments/cluster_selection/reinforcement/results/unique_training_10k_20261106.f9b9837e/development.json"
    data = json.loads(data_path.read_text())
    selection = json.loads((training / "selection.json").read_text())
    manifest = json.loads((training / "manifest.json").read_text())
    if digest(data_path) != manifest["source_protocol"]["data_sha256"]:
        raise AssertionError("Source dataset differs from original training protocol")
    chosen = selection["validation_indices"]
    if chosen != data["validation_indices"] or chosen != manifest["validation_indices"] or len(chosen) != 55:
        raise AssertionError("Expected the original complete frozen 55-puzzle validation split")
    if set(chosen) & (set(data["train_indices"]) | set(data.get("test_indices", []))):
        raise AssertionError("Training or test index overlap")
    initial, _ = instances(data)
    if fingerprints(data, [initial[i] for i in chosen]) & fingerprints(data, [initial[i] for i in data["train_indices"]]):
        raise AssertionError("Initial validation/training state overlap")
    model = runtime.load(checkpoint).cpu().eval()
    if model.network != data["network"]:
        raise AssertionError("Checkpoint/source network mismatch")
    raw = data["network"]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    models = {}
    for name in args.methods:
        models[name] = deepcopy(model)
        if name.startswith("mask"):
            models[name].config["anchor_candidates"] = int(name[4:])
    code_paths = [Path(__file__), ROOT / "experiments/cluster_selection/mask_model.py",
        ROOT / "experiments/cluster_selection/encoder.py", ROOT / "experiments/cluster_selection/runtime.py",
        ROOT / "experiments/cluster_selection/reinforcement/experiment.py",
        ROOT / "experiments/cluster_selection/candidate_generation.py",
        ROOT / "experiments/cluster_selection/candidate_evaluation_mac.py",
        ROOT / "experiments/cluster_selection/contraction_cost.py",
        ROOT / "experiments/constraint_networks/constraint_networks.py",
        ROOT / "experiments/constraint_networks/mac.py"]
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in code_paths}
    for index, path in enumerate(code_paths):
        shutil.copyfile(path, args.output / "code" / f"{index:02d}_{path.name}")
    ordering = []
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        puzzles = list(chosen)
        rng.shuffle(puzzles)
        for puzzle in puzzles:
            methods = list(args.methods)
            rng.shuffle(methods)
            ordering.extend((repeat, puzzle, method) for method in methods)
    protocol = dict(version=1, task="top-anchor mask decoding and comparable-time structural validation",
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint),
        source=str(data_path.relative_to(ROOT)), source_sha256=digest(data_path),
        original_selection_sha256=digest(training / "selection.json"), validation_indices=chosen,
        frozen_order=ordering, methods=args.methods, repeats=args.repeats, seed=args.seed,
        model_config={name: predictor.config for name, predictor in models.items()},
        nodes=10000, seconds=300, learned_allowance=64, total_work=None,
        heuristic_steps="max(8192,256*cap)", mask_stop_on_noop="True for K1; False for K>1",
        iteration="Top-K anchors at current domains; discard frontier and recompute after pruning",
        timing="One process, sequential interleaved solves; setup_s+solve_s; inference and preparation included; checkpoint loading excluded",
        no_test_evaluation=True, no_retraining=True, code_sha256=code_hashes,
        host=dict(uname=list(platform.uname()), python=sys.version, torch=torch.__version__, numpy=np.__version__,
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  pid=os.getpid(), load_average=os.getloadavg()))
    write(args.output / "protocol.json", protocol)
    # Exclude lazy torch operator/thread initialization from every measured method.
    warmup_state = network.create_mac_state(initial[chosen[0]])
    warmup_state.propagate()
    warmup_domains = warmup_state.snapshot_domains()
    with torch.no_grad():
        for name, predictor in models.items():
            if name.startswith("mask"):
                if name == "mask1":
                    predictor.propose(warmup_domains)
                else:
                    list(predictor.proposals(warmup_domains, max_candidates=8192))
    rows = []
    for position, (repeat, puzzle, name) in enumerate(ordering, 1):
        if {str(path.relative_to(ROOT)): digest(path) for path in code_paths} != code_hashes:
            raise AssertionError("Production source changed during benchmark")
        mode = "trained" if name.startswith("mask") else "heuristic" if name.startswith("heuristic") else "plain"
        cap = int(name.removeprefix("heuristic")) if mode == "heuristic" else 64
        began, cpu_began = perf_counter(), process_time()
        with torch.no_grad():
            outcome = solve(network, initial[puzzle], mode, models[name], nodes=10000, seconds=300,
                baseline_candidates=cap, baseline_steps=max(8192, 256 * cap), cluster_attempts=64,
                stop_on_noop=name == "mask1")
        row = dict(repeat=repeat, puzzle=puzzle, method=name, order=position,
                   external_elapsed_s=perf_counter() - began, process_cpu_s=process_time() - cpu_began,
                   **outcome)
        if row["status"] == "invalid":
            raise AssertionError("Invalid complete assignment")
        write(args.output / "rows" / f"r{repeat:02d}_p{puzzle:05d}_{name}.json", row)
        rows.append(row)
        print(f"{position}/{len(ordering)} {name} puzzle {puzzle}: {row['status']}, "
              f"{row['nodes']} nodes, {row['setup_s']+row['solve_s']:.3f}s", flush=True)
    summary = {}
    original = json.loads((training / "validation_0025.json").read_text())["trained"]
    heuristic = json.loads((training / "baselines.json").read_text())["heuristic"]
    for name in args.methods:
        summaries = []
        for repeat in range(args.repeats):
            group = [row for row in rows if row["repeat"] == repeat and row["method"] == name]
            complete = all(row["status"] == "solved" and row["valid"] for row in group)
            result = dict(repeat=repeat, instances=len(group), solved=sum(row["solved"] for row in group),
                total_nodes=sum(row["nodes"] for row in group) if complete else None,
                branch_free=sum(row["solved"] and row["nodes"] == 1 for row in group),
                total_setup_s=sum(row["setup_s"] for row in group), total_solve_s=sum(row["solve_s"] for row in group),
                total_time_s=sum(row["setup_s"]+row["solve_s"] for row in group),
                total_process_cpu_s=sum(row["process_cpu_s"] for row in group),
                contractions=sum(row["work"]["contractions"] for row in group),
                candidates=sum(row["cluster_statistics"].get("candidates", 0) for row in group),
                rejected=sum(row["cluster_statistics"].get("rejected", 0) for row in group),
                skipped=sum(row["cluster_statistics"].get("skipped", 0) for row in group),
                useful=sum(row["cluster_statistics"].get("useful", 0) for row in group))
            expected = original if name == "mask1" else heuristic if name == "heuristic64" else None
            if expected:
                expected_nodes = {row["puzzle"]: row["solve"]["nodes"] for row in expected["rows"]}
                result["original_per_puzzle_nodes_equal"] = all(row["nodes"] == expected_nodes[row["puzzle"]] for row in group)
                if not result["original_per_puzzle_nodes_equal"]:
                    raise AssertionError(f"{name} node counts changed from saved original validation")
            summaries.append(result)
        summary[name] = summaries
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
