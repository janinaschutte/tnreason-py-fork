"""Controlled root/context mixture ablation using existing cached training roots."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_fraction", type=float, choices=(.75, .5))
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        raise SystemExit("Requested CPU is outside the scheduler affinity")
    os.sched_setaffinity(0, {args.cpu})
    import torch
    from experiments.cluster_selection.reinforcement import refine
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.cluster_selection.reinforcement.train import _state

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_path, cache_path = BASE / "pilot_data.json", BASE / "pilot_states.json"
    parent = BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    control = BASE / "global_seed0/policy_000000000.pt"
    data, cached = json.loads(data_path.read_text()), json.loads(cache_path.read_text())
    assert digest(data_path) == cached["pilot_data_sha256"] and not cached["contains_source_solutions"]
    assert sorted(record["puzzle"] for record in cached["states"] if record["split"] == "train") == data["train_indices"]
    with torch.random.fork_rng(devices=[]):
        initialized = refine.initialize(data, seed=0, initial=parent, global_context=True)
    old = torch.load(control, map_location="cpu", weights_only=True)
    assert initialized.config == old["config"] and initialized.reward_colors == old["reward_colors"]
    assert initialized.state_dict().keys() == old["weights"].keys()
    assert all(torch.equal(value, old["weights"][name]) for name, value in initialized.state_dict().items())
    del initialized, old
    paths = dict(data=data_path, cache=cache_path, parent=parent, original_global_initial=control, driver=Path(__file__))
    hashes = {name: digest(path) for name, path in paths.items()}
    sources = refine.source_hashes()
    label = "context075_seed0" if args.root_fraction == .75 else "context050_seed0"
    output = BASE / label
    assert not output.exists(), "Use a new output directory; never replace an earlier run"
    settings = dict(trials=2048, seed=0, device="cpu", groups=4, rollouts=4, lr=1e-4,
        normalization="group", root_fraction=args.root_fraction, context_capacity=3,
        global_context=True, distance_features=False, objective_features=False,
        retain_graphs=True, epochs=1, clip_ratio=None, entropy_weight=.001,
        target_kl=.02, solve_bonus=0., validation_every=2048, validation_attempts=32,
        validation_samples=8, validation_seconds=30., checkpoint_every=1024)
    receipt = dict(paths={name: str(path) for name, path in paths.items()}, hashes=hashes,
        source_sha256=sources, settings=settings, cpu=sorted(os.sched_getaffinity(0)),
        initial_weights_equal_original_global_pilot=True,
        scope="32 existing training roots and16 development-validation puzzles; no heldout100",
        differences="Root/context mixture; retained graphs avoid rescoring with tested equivalent gradients. New CLI source closure recorded.",
        interpretation="Fresh-trial budget matched; realized root/context exposures differ and training usefulness is not directly comparable")
    write(BASE / f"{label}_before.json", receipt)

    def prepared(current, **unused):
        assert refine.digest(current) == refine.digest(data)
        raw = current["network"]
        base = {**current, "samples": [dict(domains=[list(range(dim)) for dim in raw["dimensions"]])]}
        pools = {record["puzzle"]: [dict(domains=tuple(tuple(values) for values in record["domains"]),
            root=True, known_satisfiable=True)] for record in cached["states"] if record["split"] == "train"}
        initial, _ = instances(current)
        return base, _state(base), pools, [initial[index] for index in current["validation_indices"]]

    original = refine.training_states
    refine.training_states = prepared
    try:
        result = refine.fit(data, output=output, initial=parent, **settings)
    finally:
        refine.training_states = original
    assert sources == refine.source_hashes() and hashes == {name: digest(path) for name, path in paths.items()}
    assert result["trials"] == 2048 and not result["interrupted"]
    history = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    write(BASE / f"{label}_after.json", dict(**receipt, result=result, complete=True,
        root_trials=sum(row.get("root_trials", 0) for row in history),
        context_trials=sum(row.get("context_trials", 0) for row in history),
        root_useful=sum(row.get("root_useful", 0) for row in history),
        context_useful=sum(row.get("context_useful", 0) for row in history),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(output, result, flush=True)


if __name__ == "__main__":
    main()
