"""Missing global-context + RMS control, using the same cached development pilot."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))
from context_pilot import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=0)
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
    assert sorted(row["puzzle"] for row in cached["states"] if row["split"] == "train") == data["train_indices"]
    with torch.random.fork_rng(devices=[]):
        model = refine.initialize(data, seed=0, initial=parent, global_context=True)
    old = torch.load(control, map_location="cpu", weights_only=True)
    assert model.config == old["config"] and model.reward_colors == old["reward_colors"]
    assert model.state_dict().keys() == old["weights"].keys()
    assert all(torch.equal(value, old["weights"][key]) for key, value in model.state_dict().items())
    del model, old
    paths = dict(data=data_path, cache=cache_path, parent=parent, original_global_initial=control,
                 driver=Path(__file__), helper=BASE / "context_pilot.py")
    hashes, sources = {key: digest(path) for key, path in paths.items()}, refine.source_hashes()
    output = BASE / "global_rms_seed0"
    assert not output.exists(), "Use a new output directory"
    settings = dict(trials=2048, seed=0, device="cpu", groups=4, rollouts=4, lr=1e-4,
        normalization="rms", root_fraction=1., context_capacity=3, global_context=True,
        distance_features=False, objective_features=False, retain_graphs=True, epochs=1,
        clip_ratio=None, entropy_weight=.001, target_kl=.02, solve_bonus=0.,
        validation_every=2048, validation_attempts=32, validation_samples=8,
        validation_seconds=30., checkpoint_every=1024)
    receipt = dict(paths={key: str(path) for key, path in paths.items()}, hashes=hashes,
        source_sha256=sources, settings=settings, cpu=sorted(os.sched_getaffinity(0)),
        initial_weights_equal_original_global_pilot=True,
        scope="32 training roots and16 development-validation puzzles; no heldout100",
        differences="Global context with prior RMS scaling; v3 retained-graph control, not bitwise replay of archived v1")
    write(BASE / "global_rms_seed0_before.json", receipt)

    def prepared(current, **unused):
        assert refine.digest(current) == refine.digest(data)
        raw = current["network"]
        base = {**current, "samples": [dict(domains=[list(range(dim)) for dim in raw["dimensions"]])]}
        pools = {row["puzzle"]: [dict(domains=tuple(tuple(values) for values in row["domains"]),
            root=True, known_satisfiable=True)] for row in cached["states"] if row["split"] == "train"}
        initial, _ = instances(current)
        return base, _state(base), pools, [initial[index] for index in current["validation_indices"]]

    original = refine.training_states
    refine.training_states = prepared
    try:
        result = refine.fit(data, output=output, initial=parent, **settings)
    finally:
        refine.training_states = original
    assert sources == refine.source_hashes() and hashes == {key: digest(path) for key, path in paths.items()}
    assert result["trials"] == 2048 and not result["interrupted"]
    write(BASE / "global_rms_seed0_after.json", dict(**receipt, result=result, complete=True,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(output, result, flush=True)


if __name__ == "__main__":
    main()
