"""One extra policy on the original full55 development protocol."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import mkdtemp

import torch

from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes


BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paired_full55_helpers", BASE / "full_validation.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE / "objective_rms_seed0/final.pt")
    parser.add_argument("--label", default="objective_rms_seed0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(mkdtemp(prefix=f"full55_{args.label}_", dir=BASE))
    protocol = json.loads((BASE / "full55_q13qt7s1/protocol.json").read_text())
    data = json.loads(helpers.DATA.read_text())
    assert protocol["data_sha256"] == helpers.digest(helpers.DATA)
    assert protocol["source_sha256"] == refine.source_hashes()
    assert protocol["indices"] == data["validation_indices"] and len(set(protocol["indices"])) == 55
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    initial = instances(data)[0]
    overlap = set(saved["training"]["train_instance_hashes"]) & instance_hashes(data, [initial[i] for i in protocol["indices"]])
    assert not overlap
    protocol.update(checkpoints={args.label: str(args.checkpoint)},
        checkpoint_sha256={args.label: helpers.digest(args.checkpoint)}, paired_protocol=str(BASE / "full55_q13qt7s1/protocol.json"),
        driver_sha256=helpers.digest(Path(__file__)), helper_driver_sha256=helpers.digest(BASE / "full_validation.py"))
    helpers.write(output / "protocol.json", protocol)
    model = load(args.checkpoint)
    helpers.write(output / "before.json", dict(checkpoint_sha256=helpers.digest(args.checkpoint),
        data_sha256=helpers.digest(helpers.DATA), source_sha256=refine.source_hashes(),
        config=model.config, actor_config=model.actor_config, validation_fingerprint_overlap=len(overlap)))
    original = refine_eval.solve
    progress = {"roots": 0, "nodes": 0, "complete": 0}

    def record(*a, **kw):
        row = original(*a, **kw)
        progress["complete"] += 1
        progress["nodes"] += row["nodes"]
        print(f"{args.label}: {progress['complete']}/55, residual DFS nodes {progress['nodes']}", flush=True)
        return row

    refine_eval.solve = record
    try:
        result = refine_eval.evaluate(model, data, **protocol["settings"])
    finally:
        refine_eval.solve = original
    assert protocol["source_sha256"] == refine.source_hashes()
    assert protocol["checkpoint_sha256"][args.label] == helpers.digest(args.checkpoint)
    assert protocol["data_sha256"] == helpers.digest(helpers.DATA)
    helpers.write(output / "result.json", result)
    helpers.write(output / "summary.json", dict(label=args.label, totals=result["totals"], source_hashes_unchanged=True,
        checkpoint_unchanged=True, data_unchanged=True, validation_fingerprint_overlap=0))
    print(output, result["totals"], flush=True)


if __name__ == "__main__":
    main()
