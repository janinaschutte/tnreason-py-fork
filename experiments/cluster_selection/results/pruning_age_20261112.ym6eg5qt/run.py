"""Paired validation-only test of variable age after MRV and degree ties."""
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes, solve
from experiments.cluster_selection.reinforcement.train import _state
from experiments.constraint_networks.mac import ConstraintNetworkSolver

OUT = Path(__file__).resolve().parent
TRAIN = ROOT / "experiments/cluster_selection/results/mask_sudoku_300ep.6lb43_yj/train"
DATA = ROOT / "experiments/cluster_selection/reinforcement/results/unique_training_10k_20261106.f9b9837e/development.json"
CHECKPOINT = TRAIN / "epoch_0020.pt"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(47)
    data, training = read(DATA), read(TRAIN / "manifest.json")
    initial, _ = instances(data)
    ids = training["validation_indices"]
    if (len(ids) != 55 or set(ids) - set(data["validation_indices"])
            or set(ids) & set(data["train_indices"]) or set(ids) & set(data.get("test_indices", ()))):
        raise AssertionError("Expected the same 55 reserved validation puzzles")
    if digest(DATA) != training["source_protocol"]["data_sha256"]:
        raise AssertionError("Source data changed")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if set(saved["training"]["train_instance_hashes"]) & instance_hashes(data, [initial[i] for i in ids]):
        raise AssertionError("Training exposure overlaps validation")
    model = runtime.load(CHECKPOINT).cpu().eval()
    if model.network != data["network"] or model.config.get("anchor_candidates", 1) != 1:
        raise AssertionError("Expected the fixed best single-anchor mask model")
    network = _state({**data, "samples": [dict(domains=[list(range(d))
                     for d in data["network"]["dimensions"]])]}).constraint_network
    references = read(TRAIN / "baselines.json")
    references.update(read(TRAIN / "validation_0020.json"))
    reference_nodes = {mode: {r["puzzle"]: r["solve"]["nodes"] for r in value["rows"]}
                       for mode, value in references.items()}
    cases = [(tie, mode) for tie in ("name", "oldest_pruned")
             for mode in ("plain", "heuristic", "trained")]
    Random(47).shuffle(cases)
    sources = [Path(__file__), ROOT / "experiments/constraint_networks/mac.py",
        ROOT / "experiments/constraint_networks/constraint_networks.py", Path(runtime.__file__),
        ROOT / "experiments/cluster_selection/mask_model.py", ROOT / "experiments/cluster_selection/encoder.py",
        ROOT / "experiments/cluster_selection/candidate_generation.py",
        ROOT / "experiments/cluster_selection/contraction_cost.py",
        ROOT / "experiments/cluster_selection/candidate_evaluation_mac.py",
        ROOT / "experiments/cluster_selection/reinforcement/experiment.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    protocol = dict(version=1, data_sha256=digest(DATA), checkpoint_sha256=digest(CHECKPOINT),
        source_sha256=hashes, validation_indices=ids, cases=cases, seed=47,
        node_guard=10000, seconds_guard=300, allowance=64, generator_steps=16384,
        stop_on_noop=True, total_work=None, model_config=model.config,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        reference_nodes=reference_nodes, no_training=True, no_test_queries=True,
        age_definition="Last reduction position in current reversible domain trail; never reduced=0. "
            "Includes retained temporary-summary pruning; speculative reductions and backtracking undo it.",
        rule="MRV, highest incident-core degree, oldest pruning (optional), name. Value order unchanged.")
    path = OUT / "protocol.json"
    if path.exists():
        if read(path) != json.loads(json.dumps(protocol)):
            raise AssertionError("Frozen protocol changed")
    else:
        write(path, protocol)
    (OUT / "rows").mkdir(exist_ok=True)
    protocol_hash = digest(path)
    original_solve, original_select = network.solve_mac, ConstraintNetworkSolver._select_color
    summaries = {}
    try:
        for tie, mode in cases:
            if any(digest(ROOT / p) != expected for p, expected in hashes.items()):
                raise AssertionError("Source changed during comparison")
            network.solve_mac = partial(original_solve, variable_tie_break=tie)
            rows = []
            for position, puzzle in enumerate(ids, 1):
                path = OUT / "rows" / f"{tie}_{mode}_{puzzle}.json"
                if path.exists():
                    record = read(path)
                    if record["protocol_sha256"] != protocol_hash:
                        raise AssertionError("Saved row provenance changed")
                else:
                    counters = dict(choices=0, different_from_name=0)

                    def select(solver, state):
                        chosen = original_select(solver, state)
                        legacy = min((c for c, d in state.domains.items() if len(d) > 1),
                            key=lambda c: (len(state.domains[c]), -len(state._core_keys_by_color.get(c, ())), c))
                        counters["choices"] += 1
                        counters["different_from_name"] += int(chosen != legacy)
                        if tie == "name" and chosen != legacy:
                            raise AssertionError("Default branching changed")
                        return chosen

                    ConstraintNetworkSolver._select_color = select
                    try:
                        with torch.no_grad():
                            row = solve(network, initial[puzzle], mode, model, nodes=10000, seconds=300,
                                baseline_candidates=64, baseline_steps=16384, cluster_attempts=64,
                                stop_on_noop=True)
                    finally:
                        ConstraintNetworkSolver._select_color = original_select
                    if row["status"] in ("invalid", "unsat"):
                        raise AssertionError("Invalid validation solution")
                    record = dict(protocol_sha256=protocol_hash, tie=tie, mode=mode, puzzle=puzzle,
                                  branching=counters, **row)
                    write(path, record)
                rows.append(record)
                if position == 1 or position % 10 == 0 or position == len(ids):
                    print(tie, mode, f"{position}/{len(ids)}", puzzle, record["status"], record["nodes"], flush=True)
            complete = all(r["solved"] and r["valid"] for r in rows)
            result = dict(complete=complete, completed=sum(r["solved"] and r["valid"] for r in rows),
                nodes=sum(r["nodes"] for r in rows) if complete else None,
                branch_free=sum(r["solved"] and r["nodes"] == 1 for r in rows),
                choices=sum(r["branching"]["choices"] for r in rows),
                choices_different_from_name=sum(r["branching"]["different_from_name"] for r in rows),
                nodes_by_puzzle={r["puzzle"]: r["nodes"] for r in rows})
            if tie == "name":
                result["matches_historical_nodes"] = result["nodes_by_puzzle"] == reference_nodes[mode]
            summaries[f"{tie}_{mode}"] = result
            print("COMPLETE", tie, mode, json.dumps(result), flush=True)
    finally:
        network.solve_mac = original_solve
        ConstraintNetworkSolver._select_color = original_select
    if not (OUT / "summary.json").exists():
        write(OUT / "summary.json", summaries)


if __name__ == "__main__":
    main()
