"""Paired parent/global-policy check on all 55 existing development puzzles."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from time import sleep

import torch

from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes


ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent
DATA = ROOT / "experiments/cluster_selection/reinforcement/results/unique_training_10k_20261106.f9b9837e/development.json"
PARENT = ROOT / "experiments/cluster_selection/reinforcement/results/learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, default=BASE / "group_seed1/final.pt")
    args = parser.parse_args()
    if not args.wait_for.exists():
        print(f"Waiting for CPU 2 to become free: {args.wait_for}", flush=True)
    while not args.wait_for.exists():
        sleep(15)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(tempfile.mkdtemp(prefix="full55_", dir=BASE))
    data = json.loads(DATA.read_text())
    selected = data["validation_indices"]
    assert len(selected) == len(set(selected)) == 55
    initial, _ = instances(data)
    validation_hashes = instance_hashes(data, [initial[index] for index in selected])
    sources = refine.source_hashes()
    settings = dict(attempts=32, samples=8, seconds=30., seed=100000, nodes=10000)
    checkpoints = dict(parent=PARENT, global_seed0=BASE / "global_seed0/final.pt")
    fixed_hashes = {"data": digest(DATA), **{name: digest(path) for name, path in checkpoints.items()}}
    write(output / "protocol.json", dict(data=str(DATA), data_sha256=fixed_hashes["data"],
        indices=selected, instances=55, settings=settings, source_sha256=sources,
        driver_sha256=digest(Path(__file__)), checkpoints={name: str(path) for name, path in checkpoints.items()},
        checkpoint_sha256={name: fixed_hashes[name] for name in checkpoints},
        scope="Previously used development validation, never heldout100; root propagation then separate plain MAC",
        timing="CPU 2, sequential methods on shared host; no comparative speed claim"))
    results = {}
    original_propagate, original_solve = refine_eval.propagate, refine_eval.solve
    for name, checkpoint in checkpoints.items():
        assert digest(DATA) == fixed_hashes["data"] and digest(checkpoint) == fixed_hashes[name]
        assert sources == refine.source_hashes()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        inherited = set(saved["training"]["train_instance_hashes"])
        assert not inherited & validation_hashes
        model = load(checkpoint)
        write(output / f"{name}_before.json", dict(data_sha256=digest(DATA), checkpoint_sha256=digest(checkpoint),
            source_sha256=refine.source_hashes(), config=model.config, actor_config=model.actor_config,
            training_fingerprint_count=len(inherited), validation_fingerprint_overlap=0))
        progress = {"roots": 0, "completed": 0, "nodes": 0, "censored": 0}

        def record_root(*args, **kwargs):
            row = original_propagate(*args, **kwargs)
            progress["roots"] += int(row["solved"])
            return row

        def record_residual(*args, **kwargs):
            row = original_solve(*args, **kwargs)
            position = progress["completed"]
            progress["completed"] += 1
            if row["status"] in {"solved", "unsat"}:
                progress["nodes"] += row["nodes"]
            else:
                progress["censored"] += 1
            print(f"{name}: {progress['completed']}/55, index {selected[position]}, "
                  f"roots {progress['roots']}, completed DFS nodes {progress['nodes']}, "
                  f"censored {progress['censored']}", flush=True)
            return row

        refine_eval.propagate, refine_eval.solve = record_root, record_residual
        try:
            results[name] = refine_eval.evaluate(model, data, **settings)
        finally:
            refine_eval.propagate, refine_eval.solve = original_propagate, original_solve
        assert [row["index"] for row in results[name]["rows"]] == selected
        assert sources == refine.source_hashes() and digest(checkpoint) == fixed_hashes[name]
        write(output / f"{name}.json", results[name])
        write(output / f"{name}_after.json", dict(data_sha256=digest(DATA), checkpoint_sha256=digest(checkpoint),
            source_sha256=refine.source_hashes(), complete=True))
        print(name, results[name]["totals"], flush=True)
    pairs = []
    for parent, trained in zip(results["parent"]["rows"], results["global_seed0"]["rows"]):
        assert parent["index"] == trained["index"]
        complete = all(row["residual"]["status"] in {"solved", "unsat"} for row in (parent, trained))
        pairs.append(dict(index=parent["index"], parent_root=parent["root"]["solved"],
            trained_root=trained["root"]["solved"], parent_nodes=parent["residual"]["nodes"] if complete else None,
            trained_nodes=trained["residual"]["nodes"] if complete else None, both_complete=complete))
    summary = dict(totals={name: result["totals"] for name, result in results.items()},
        root_wins=sum(row["trained_root"] and not row["parent_root"] for row in pairs),
        root_losses=sum(row["parent_root"] and not row["trained_root"] for row in pairs),
        dfs_wins=sum(row["trained_nodes"] < row["parent_nodes"] for row in pairs if row["both_complete"]),
        dfs_losses=sum(row["parent_nodes"] < row["trained_nodes"] for row in pairs if row["both_complete"]),
        complete_pairs=sum(row["both_complete"] for row in pairs), pairs=pairs,
        interpretation="Existing development validation; this comparison does not establish fresh test performance.")
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
