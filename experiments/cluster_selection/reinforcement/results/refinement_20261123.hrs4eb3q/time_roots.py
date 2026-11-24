"""Time-only root check on the same 16 development puzzles; no training."""
import argparse
from pathlib import Path
import sys
from tempfile import mkdtemp

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))

import torch
from full_validation import digest, write
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement import refine, refine_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("parent", "untrained", "global", "role"))
    args = parser.parse_args()
    paths = dict(parent=BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt",
        untrained=BASE / "common_v2_2wlie6na/untrained.pt", global_=BASE / "global_seed0/final.pt",
        role=BASE / "objective_rms_seed0/final.pt")
    checkpoint = paths["global_" if args.method == "global" else args.method]
    data_path = BASE / "pilot_data.json"
    reference_path = BASE / "baselines32_wgk4zatg/plain.json"
    import json
    data, model = json.loads(data_path.read_text()), load(checkpoint)
    torch.set_num_threads(1)
    output = Path(mkdtemp(prefix=f"time30_{args.method}_", dir=BASE))
    settings = dict(attempts=0, samples=0, seconds=30., nodes=10000, seed=100000)
    sources = refine.source_hashes()
    hashes = {str(path): digest(path) for path in (data_path, checkpoint, reference_path, Path(__file__), BASE / "full_validation.py")}
    write(output / "protocol.json", dict(method=args.method, data=str(data_path), checkpoint=str(checkpoint),
        hashes=hashes, sources=sources, settings=settings,
        scope="Same 16 development puzzles; no test100; root then plain residual MAC",
        timing="30s wall per root on a shared CPU host, no aggregate contraction allowance"))
    original, completed = refine_eval.solve, []

    def observed(*pos, **kw):
        row = original(*pos, **kw)
        completed.append(row)
        print(f"{args.method} {len(completed)}/16 residual {row['status']}, nodes {row['nodes']}", flush=True)
        return row

    refine_eval.solve = observed
    try:
        result = refine_eval.evaluate(model, data, **settings)
    finally:
        refine_eval.solve = original
    references = {row["index"]: row["residual"]["assignment"] for row in json.loads(reference_path.read_text())["rows"]}
    assert all(all(value in row["root"]["domains"][color] for color, value in references[row["index"]].items())
               for row in result["rows"])
    assert sources == refine.source_hashes() and all(digest(path) == wanted for path, wanted in hashes.items())
    write(output / "result.json", result)
    write(output / "audit.json", dict(sources_and_inputs_unchanged=True, instances=len(completed),
        complete=result["totals"]["dfs_complete"], outputs_valid=True, retained_references=len(completed)))
    print(output, result["totals"], flush=True)


if __name__ == "__main__":
    main()
