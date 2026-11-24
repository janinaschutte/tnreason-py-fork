"""Fresh untrained constructor on all 55 existing development-validation puzzles."""
from pathlib import Path
import argparse
import json
import os
import platform
import sys
from tempfile import mkdtemp
from time import sleep

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=BASE / "objective_rms_seed0/final.pt")
    parser.add_argument("--random", type=Path, default=BASE / "common_v2_2wlie6na/untrained.pt")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if not args.gate.exists():
        if not args.wait:
            raise SystemExit("Wait for the objective RMS pilot's final.pt before using CPU 0")
        print(f"Waiting for the objective RMS final checkpoint: {args.gate}", flush=True)
        while not args.gate.exists():
            sleep(15)
    if 0 not in os.sched_getaffinity(0):
        raise SystemExit("CPU 0 is outside the scheduler affinity")
    os.sched_setaffinity(0, {0})
    import torch
    from full_validation import DATA, PARENT, digest, write
    from experiments.cluster_selection.reinforcement import refine, refine_eval
    from experiments.cluster_selection.reinforcement.actor import Policy, load
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(mkdtemp(prefix="full55_random_", dir=BASE))
    print(output, flush=True)
    data, sources = json.loads(DATA.read_text()), refine.source_hashes()
    selected = data["validation_indices"]
    assert len(selected) == len(set(selected)) == 55
    initial, _ = instances(data)
    validation_hashes = instance_hashes(data, [initial[index] for index in selected])
    parent_saved = torch.load(PARENT, map_location="cpu", weights_only=True)
    assert not set(parent_saved["training"]["train_instance_hashes"]) & validation_hashes
    parent = load(PARENT)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        fresh = Policy(parent.network, **parent.config)
    fresh.reward_colors, fresh.beta = parent.reward_colors, parent.beta
    model = load(args.random)
    assert model.network == parent.network and model.config == parent.config
    assert model.actor_config == fresh.actor_config
    assert not any(model.actor_config.values()), "Random reference must have all adapter flags disabled"
    assert model.reward_colors == parent.reward_colors and model.beta == parent.beta
    assert model.state_dict().keys() == fresh.state_dict().keys()
    assert all(torch.equal(value, fresh.state_dict()[key]) for key, value in model.state_dict().items())
    settings = dict(attempts=32, samples=8, seconds=30., seed=100000, nodes=10000)
    paths = dict(data=DATA, parent=PARENT, random=args.random, gate=args.gate,
                 driver=Path(__file__), helper=BASE / "full_validation.py")
    hashes = {name: digest(path) for name, path in paths.items()}

    def verify():
        assert sources == refine.source_hashes()
        assert hashes == {name: digest(path) for name, path in paths.items()}

    receipt = dict(source_sha256=sources, artifact_sha256=hashes, config=model.config,
        actor_config=model.actor_config, random_initialization_seed=0, exact_fresh_weights_verified=True,
        indices=selected, validation_fingerprint_overlap=0, validation_hash_count=len(validation_hashes))
    write(output / "protocol.json", dict(paths={name: str(path) for name, path in paths.items()},
        settings=settings, **receipt, cpu=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
        torch_version=str(torch.__version__), hostname=platform.node(),
        scope="All 55 existing development-validation puzzles, never heldout100; root propagation then separate plain MAC",
        timing="CPU 0, shared host; timing is not matched against other methods"))
    verify()
    write(output / "untrained_before.json", receipt)
    propagate, solve = refine_eval.propagate, refine_eval.solve
    progress = dict(roots=0, completed=0, nodes=0, censored=0)

    def record_root(*pos, **kw):
        row = propagate(*pos, **kw)
        progress["roots"] += int(row["solved"])
        return row

    def record_residual(*pos, **kw):
        row = solve(*pos, **kw)
        index = selected[progress["completed"]]
        progress["completed"] += 1
        complete = row["status"] in {"solved", "unsat"}
        progress["nodes"] += row["nodes"] if complete else 0
        progress["censored"] += int(not complete)
        print(f"untrained: {progress['completed']}/55, index {index}, roots {progress['roots']}, "
              f"completed DFS nodes {progress['nodes']}, censored {progress['censored']}", flush=True)
        return row

    refine_eval.propagate, refine_eval.solve = record_root, record_residual
    try:
        result = refine_eval.evaluate(model, data, **settings)
    finally:
        refine_eval.propagate, refine_eval.solve = propagate, solve
    assert [row["index"] for row in result["rows"]] == selected
    assert result["totals"]["instances"] == progress["completed"] == 55
    verify()
    write(output / "untrained.json", result)
    write(output / "untrained_after.json", dict(**receipt, complete=True, progress=progress))
    print(json.dumps(result["totals"], sort_keys=True), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
