"""Verify frozen inputs, then run one controlled weights-only continuation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys

folder = Path(__file__).resolve().parent
repo = folder.parents[5]
read = lambda path: json.loads(path.read_text())
digest = lambda path: sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("reinforce", "lower_lr", "self_imitation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pilot", action="store_true", help="Frozen 16-training/8-validation mechanism pilot")
    args = parser.parse_args()
    plan = read(folder / "protocol.json")
    inputs = read(folder / "inputs.json")
    if digest(folder / "inputs.json") != plan["inputs_sha256"]:
        raise SystemExit("Frozen continuation inputs changed")
    for relative, expected in plan["source_sha256"].items():
        if digest(repo / relative) != expected:
            raise SystemExit(f"Source differs from the prepared experiment: {relative}")
    for label in ("data", "parent"):
        if digest(repo / inputs[label]["relative"]) != inputs[label]["sha256"]:
            raise SystemExit(f"Frozen {label} changed")
    data = read(repo / inputs["data"]["relative"])
    if (len(data["train_indices"]), len(data["validation_indices"])) != (10000, 55):
        raise SystemExit("Expected the frozen explicit 10000/55 development split")
    budget = inputs["pilot" if args.pilot else "gpu"]
    options = [str(repo / inputs["data"]["relative"]), "--output", str(args.output),
               "--device", args.device, "--initial-checkpoint", str(repo / inputs["parent"]["relative"]),
               *inputs["common_cli"], *inputs["arms"][args.arm]["cli"], *budget["cli"]]
    print("Frozen inputs verified; fresh Adam, RNG, teacher cache, pools and reward scale; parent weights retained.", flush=True)
    print(shlex.join([sys.executable, "-u", "-m", "experiments.cluster_selection.reinforcement.learn", *options]), flush=True)
    sys.path.insert(0, str(repo))
    from experiments.cluster_selection.reinforcement import learn
    sys.argv = ["learn", *options]
    learn.main()


if __name__ == "__main__":
    main()
