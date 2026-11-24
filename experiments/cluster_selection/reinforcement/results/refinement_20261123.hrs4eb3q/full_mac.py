"""Whole-solve MAC check with durable rows and adapter-aware model loading."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[4]))

import torch
from experiments.cluster_selection.reinforcement import refine, refine_eval


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_path = BASE / "pilot_data.json"
    data = json.loads(data_path.read_text())
    parent = BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    paths = dict(plain=parent, parent=parent, global_context=BASE / "global_seed0/final.pt")
    output = Path(mkdtemp(prefix="fullmac32_", dir=BASE))
    sources = refine.source_hashes()
    inputs = {str(path): digest(path) for path in (data_path, Path(__file__), *paths.values())}
    write(output / "protocol.json", dict(data=str(data_path), checkpoint_paths={key: str(path) for key, path in paths.items()},
        hashes=inputs, sources=sources, affinity=sorted(os.sched_getaffinity(0)),
        settings=dict(attempts=32, seconds=30., nodes=10000, seed=100000, samples=0, full_mac=True),
        note="16 fixed development validation puzzles. Clusters are used at every DFS propagation closure. "
             "The 30s wall limit covers root propagation and all DFS together. Default MRV/degree/name branching. "
             "Methods run serially on CPU3 while other local jobs can run on other physical cores. "
             "Censored DFS totals are None; incomplete counts are never compared as completed node totals."))
    original, reports = refine_eval.solve, {}
    for name, checkpoint in paths.items():
        folder = output / name
        folder.mkdir()
        positions = []

        def observed(*pos, **kw):
            row = original(*pos, **kw)
            index = data["validation_indices"][len(positions)]
            positions.append(index)
            write(folder / f"row_{index:02d}.json", dict(index=index, **row))
            return row

        refine_eval.solve = observed
        print(f"START {name}", flush=True)
        try:
            reports[name] = refine_eval.main([str(data_path), "--checkpoint", str(checkpoint), "--output",
                str(folder / "result.json"), "--full-mac", "--mode", "plain" if name == "plain" else "trained",
                "--attempts", "32", "--seconds", "30", "--nodes", "10000", "--seed", "100000"])
        finally:
            refine_eval.solve = original
        assert positions == data["validation_indices"]
    assert sources == refine.source_hashes() and all(digest(Path(path)) == value for path, value in inputs.items())
    write(output / "summary.json", {name: report["totals"] for name, report in reports.items()})
    expert_path = BASE / "baselines32_wgk4zatg/expert.json"
    expert_totals = json.loads(expert_path.read_text())["totals"]
    write(output / "expert_reference.json", dict(source=str(expert_path), source_sha256=digest(expert_path),
        scope="Root-only 51-family Sudoku expert, then separate plain-MAC residual search; not full MAC throughout DFS",
        totals=expert_totals,
        note="Earlier CPU2 observation with larger core/work/size limits; reference only, not a matched full-MAC row."))
    print(output, flush=True)


if __name__ == "__main__":
    main()
