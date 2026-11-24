"""Durable time-only evaluation of a completed cached-state ablation."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import sleep

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))
from context_pilot import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("context075_seed0", "context050_seed0", "global_rms_seed0"))
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    checkpoint, gate = BASE / args.method / "final.pt", BASE / f"{args.method}_after.json"
    if not gate.exists() and not args.wait:
        raise SystemExit("Wait for the completed pilot's after receipt")
    while not gate.exists():
        sleep(15)
    receipt = json.loads(gate.read_text())
    assert receipt["complete"] and receipt["result"]["trials"] == 2048 and not receipt["result"]["interrupted"]
    if args.cpu not in os.sched_getaffinity(0):
        raise SystemExit("Requested CPU is outside the scheduler affinity")
    os.sched_setaffinity(0, {args.cpu})
    import torch
    from experiments.cluster_selection.reinforcement.actor import load
    from experiments.cluster_selection.reinforcement import refine, refine_eval

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["fresh_trials"] == 2048
    data_path, references_path = BASE / "pilot_data.json", BASE / "baselines32_wgk4zatg/plain.json"
    data, model = json.loads(data_path.read_text()), load(checkpoint)
    output = Path(mkdtemp(prefix=f"time30_{args.method}_", dir=BASE))
    print(output, flush=True)
    sources = refine.source_hashes()
    paths = dict(data=data_path, checkpoint=checkpoint, training_gate=gate, references=references_path,
                 driver=Path(__file__), helper=BASE / "context_pilot.py")
    hashes = {name: digest(path) for name, path in paths.items()}
    settings = dict(attempts=0, samples=0, seconds=30., nodes=10000, seed=100000)
    write(output / "protocol.json", dict(method=args.method, paths={name: str(path) for name, path in paths.items()},
        hashes=hashes, sources=sources, settings=settings, cpu=sorted(os.sched_getaffinity(0)),
        scope="Same16 development puzzles, never heldout100;30s per root without attempt cap, then separate plain MAC"))
    original_root, original_solve = refine_eval.propagate, refine_eval.solve
    roots, completed = [], []

    def observed_root(*pos, **kw):
        row = original_root(*pos, **kw)
        roots.append(row)
        return row

    def observed_solve(*pos, **kw):
        row = original_solve(*pos, **kw)
        position = len(completed)
        completed.append(row)
        write(output / f"row_{position:02d}.json", dict(index=data["validation_indices"][position], root=roots[position], residual=row))
        print(f"{args.method} {len(completed)}/16 residual {row['status']}, nodes {row['nodes']}", flush=True)
        return row

    refine_eval.propagate, refine_eval.solve = observed_root, observed_solve
    try:
        result = refine_eval.evaluate(model, data, **settings)
    finally:
        refine_eval.propagate, refine_eval.solve = original_root, original_solve
    references = {row["index"]: row["residual"]["assignment"] for row in json.loads(references_path.read_text())["rows"]}
    assert [row["index"] for row in result["rows"]] == data["validation_indices"]
    assert all(all(value in row["root"]["domains"][color] for color, value in references[row["index"]].items()) for row in result["rows"])
    assert sources == refine.source_hashes() and hashes == {name: digest(path) for name, path in paths.items()}
    write(output / "result.json", result)
    write(output / "audit.json", dict(sources_and_inputs_unchanged=True, instances=len(completed),
        complete=result["totals"]["dfs_complete"], outputs_valid=True, retained_references=len(completed)))
    print(output, result["totals"], flush=True)


if __name__ == "__main__":
    main()
