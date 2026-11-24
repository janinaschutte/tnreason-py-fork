"""Retrospective validation loss on fixed, evaluation-only heuristic labels."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.library import _run
from experiments.cluster_selection.reinforcement.experiment import fingerprints, instances
from experiments.cluster_selection.runtime import load
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.workload_to_numpy import NumpyCore


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    output = Path(__file__).resolve().parent
    labels = output / "evaluation_labels"
    labels.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(29)
    training = ROOT / "experiments/cluster_selection/results/mask_sudoku.FGN32oF0/train"
    data_path = ROOT / "experiments/cluster_selection/reinforcement/results/unique_training_10k_20261106.f9b9837e/development.json"
    manifest = json.loads((training / "manifest.json").read_text())
    selection = json.loads((training / "selection.json").read_text())
    data = json.loads(data_path.read_text())
    if digest(data_path) != manifest["source_protocol"]["data_sha256"]:
        raise AssertionError("Dataset differs from original training source")
    chosen = selection["validation_indices"]
    if chosen != manifest["validation_indices"] or chosen != data["validation_indices"] or len(chosen) != 55:
        raise AssertionError("Expected original complete 55-puzzle validation split")
    if set(chosen) & (set(data["train_indices"]) | set(data.get("test_indices", []))):
        raise AssertionError("Validation index overlaps training or test")
    initial, _ = instances(data)
    if fingerprints(data, [initial[i] for i in chosen]) & fingerprints(data, [initial[i] for i in data["train_indices"]]):
        raise AssertionError("Validation/training initial-domain overlap")
    raw = data["network"]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    if network_data(network) != raw:
        raise AssertionError("Raw network variable/core ordering changed")
    reward = data.get("reward_colors", raw["colors"])
    config = dict(allowance=64, max_steps=16384, max_cores=64, max_open_colors=6,
        max_work=1_000_000, max_size=100_000, nodes=10000, seconds=300,
        compact_domains=False, record_failures=False)
    checkpoints = {epoch: training / (f"epoch_{epoch:04d}.pt" if epoch else "initial.pt")
                   for epoch in (0, 5, 10, 15, 20, 25, 30)}
    code = [Path(__file__), ROOT / "experiments/cluster_selection/library.py",
        ROOT / "experiments/cluster_selection/mask_model.py", ROOT / "experiments/cluster_selection/encoder.py",
        ROOT / "experiments/cluster_selection/runtime.py", ROOT / "experiments/cluster_selection/candidate_generation.py",
        ROOT / "experiments/cluster_selection/candidate_evaluation_mac.py",
        ROOT / "experiments/cluster_selection/contraction_cost.py", ROOT / "experiments/constraint_networks/mac.py",
        ROOT / "experiments/constraint_networks/constraint_networks.py"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in code}
    write(output / "validation_loss_protocol.json", dict(version=1,
        kind="retrospective_validation_mask_loss_evaluation_only", source=str(data_path.relative_to(ROOT)),
        source_sha256=digest(data_path), original_manifest_sha256=digest(training / "manifest.json"),
        original_selection_sha256=digest(training / "selection.json"), selected_validation_puzzles=chosen,
        config=config, fixed_anchor_seed=29, anchors="One uniformly chosen actually-pruned nonsingleton variable per label, fixed for every checkpoint",
        excluded="Contradictory applications; no final test data", label_weighting="Equal weight per successful application, matching training",
        checkpoint_sha256={str(epoch):digest(path) for epoch,path in checkpoints.items()},
        code_sha256=hashes, threads=1, batch_size=16, torch_version=torch.__version__,
        no_retraining=True, note="These labels must never enter a training corpus or teacher collection."))
    baseline = json.loads((training / "baselines.json").read_text())["heuristic"]
    expected = {row["puzzle"]: row["solve"]["nodes"] for row in baseline["rows"]}
    counts, examples, contradictions = {}, [], 0
    dimensions = np.asarray(raw["dimensions"])
    max_dim, variables, cores = int(dimensions.max()), len(dimensions), len(raw["cores"])
    for position, puzzle in enumerate(chosen, 1):
        record = _run(network, initial[puzzle], puzzle, reward, config)
        if record["status"] != "solved" or record["nodes"] != expected[puzzle]:
            raise AssertionError("Heuristic solve differs from frozen original validation")
        write(labels / f"puzzle_{puzzle:05d}.json", record)
        counts[puzzle] = 0
        for trial in record["applications"]:
            if trial["contradiction"]:
                contradictions += 1
                continue
            state = np.asarray(record["states"][trial["state"]], dtype=np.int64)
            allowed = (state[:, None] & (1 << np.arange(max_dim))) != 0
            if allowed.shape != (variables,max_dim) or not allowed.any(1).all() or np.any(allowed & (np.arange(max_dim)[None] >= dimensions[:,None])):
                raise AssertionError("Invalid recorded domains")
            pruned = np.zeros(variables, dtype=bool)
            removed = 0
            for change in trial["pruned_values"]:
                variable, values = change["variable"], change["removed"]
                if (not values or len(values) != len(set(values)) or pruned[variable]
                        or not all(0 <= v < dimensions[variable] and allowed[variable,v] for v in values)
                        or len(values) >= allowed[variable].sum()):
                    raise AssertionError("Pruning label inconsistent with original domains")
                pruned[variable] = True
                removed += len(values)
            if removed != trial["removed_values"] or not pruned.any() or np.any(pruned & (allowed.sum(1) <= 1)):
                raise AssertionError("Incomplete pruning labels")
            core_mask, output_mask = np.zeros(cores,dtype=bool), np.zeros(variables,dtype=bool)
            core_mask[trial["cores"]], output_mask[trial["open"]] = True, True
            if not core_mask.any() or not output_mask.any():
                raise AssertionError("Expected nonempty noncontradictory teacher cluster")
            examples.append((allowed, core_mask, output_mask, pruned, puzzle))
            counts[puzzle] += 1
        if position == 1 or position % 5 == 0 or position == len(chosen):
            print(f"labels {position}/55: {len(examples)} pruning examples, {contradictions} contradictions excluded", flush=True)
    if sum(expected.values()) != 286 or not examples:
        raise AssertionError("Unexpected heuristic node total or empty validation labels")
    arrays = {key: np.stack([example[i] for example in examples]) for i,key in
              enumerate(("allowed","cores","outputs","pruned","puzzles"))}
    rng = np.random.default_rng(29)
    anchors = np.asarray([rng.choice(np.flatnonzero(target)) for target in arrays["pruned"]], dtype=np.int64)
    with (labels / "labels.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays, anchors=anchors)
    summary = dict(kind="evaluation_only_validation_mask_labels", samples=len(examples),
        selected_validation_puzzles=55, labelled_puzzles=sum(n>0 for n in counts.values()),
        labelled_puzzle_ids=[p for p,n in counts.items() if n], zero_label_puzzles=[p for p,n in counts.items() if not n],
        labels_per_puzzle=counts, contradictions_excluded=contradictions, heuristic_total_nodes=286,
        all_original_per_puzzle_nodes_equal=True, labels_sha256=digest(labels / "labels.npz"),
        anchor_sha256=sha256(anchors.tobytes()).hexdigest())
    write(labels / "summary.json", summary)
    tensors = {key: torch.from_numpy(value) for key,value in arrays.items() if key != "puzzles"}
    anchor_tensor, rows = torch.from_numpy(anchors), []
    for epoch,path in checkpoints.items():
        if {str(p.relative_to(ROOT)):digest(p) for p in code} != hashes:
            raise AssertionError("Evaluation code changed during run")
        model, totals, began = load(path).cpu().eval(), {}, perf_counter()
        if model.network != raw:
            raise AssertionError("Checkpoint network differs")
        with torch.no_grad():
            for start in range(0,len(examples),16):
                batch = {key:value[start:start+16] for key,value in tensors.items()}
                loss, metrics = model.loss(batch["allowed"],batch["cores"],batch["outputs"],batch["pruned"],
                                           anchors=anchor_tensor[start:start+16])
                for key,value in {**metrics,"loss":loss}.items():
                    number = float(value)
                    if not np.isfinite(number):
                        raise AssertionError("Nonfinite validation loss")
                    totals[key] = totals.get(key,0.) + len(batch["allowed"])*number
        row = dict(epoch=epoch,metrics={key:value/len(examples) for key,value in totals.items()},
                   checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path),seconds=perf_counter()-began)
        write(output / f"validation_loss_epoch_{epoch:04d}.json",row)
        rows.append(row)
        print(f"epoch {epoch}: validation loss {row['metrics']['loss']:.4f}, anchor hit {row['metrics']['anchor_hit']:.3f}",flush=True)
    write(output / "validation_loss.json",dict(version=1,**summary,rows=rows,
        protocol_sha256=digest(output / "validation_loss_protocol.json"),no_retraining=True,
        note="Retrospective fixed-label validation loss; never supplied to the completed training job."))


if __name__ == "__main__":
    main()
