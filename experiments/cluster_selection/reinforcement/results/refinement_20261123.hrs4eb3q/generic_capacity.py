"""Fixed generic learning-capacity check; deliberately not a generalization test."""
from itertools import product
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import Policy, load


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run():
    torch.set_num_threads(1)
    output = Path(tempfile.mkdtemp(prefix="generic_capacity_", dir=Path(__file__).parent))
    tables = dict(ab=np.eye(2, dtype=bool), bc=np.eye(2, dtype=bool),
                  ac=np.array([[1, 1], [1, 0]], dtype=bool))
    tables.update({key: np.ones((2, 2), dtype=bool) for key in ("ax", "xy", "ay")})
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key))
        for key, values in tables.items()}, maintain_indices=True)
    raw = network_data(network)
    root = {color: [0, 1] for color in raw["colors"]}
    data = dict(version=1, network=raw, reward_colors=list("abc"), train_indices=[0, 1],
        validation_indices=[2], initial_domains=[root, {**root, "x": [0]}, {**root, "x": [1]}])
    write(output / "data.json", data)
    sources = refine.source_hashes()
    write(output / "protocol.json", dict(training_seed=29, probe_seed=1234, probe_rollouts=128,
        fresh_training_trials=512, learning_rate=.01, groups=2, rollouts=8, root_fraction=1.,
        normalization="group", entropy_weight=0., source_hashes=sources,
        note="Fixed hand-chosen generic capacity fit on a GAC-stalled training root; "
             "not Sudoku evidence, held-out topology transfer, or a comparison with older training."))
    torch.manual_seed(29)
    actor = Policy(raw, width=8, rounds=1, contextual=True, target_first=True, exploration=0.,
        heuristic_candidates=0, decoder="sampling_grow", min_active=2, compact_domains=True,
        cyclic_only=True, max_cores=3, max_open_colors=2)
    actor.reward_colors, actor.beta = list("abc"), 0.
    parent = output / "parent.pt"
    torch.save(dict(version=1, kind=actor.kind, network=raw, config=actor.config,
        weights=actor.state_dict(), reward_colors=actor.reward_colors, beta=0.,
        training=dict(train_instance_hashes=[])), parent)
    state = network.create_mac_state()
    assert state.propagate() and not state.is_solved()
    sample = dict(domains=list(state.domains.values()), root=True, known_satisfiable=True)
    solutions = []
    for values in product((0, 1), repeat=len(raw["colors"])):
        assignment = dict(zip(raw["colors"], values))
        if all(core[{color: assignment[color] for color in core.colors}]
               for core in network.coresDict.values()):
            solutions.append(assignment)
    assert len(solutions) == 4

    def probe(policy, label):
        before = state.snapshot_domains(), tuple(state._cores), tuple(state._trail)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            group, outcomes, distinct = refine.collect(policy, state, sample, rollouts=128)
        assert before == (state.snapshot_domains(), tuple(state._cores), tuple(state._trail))
        rows = []
        for rollout, metrics in zip(group["rollouts"], outcomes):
            assert not metrics["contradictions"]
            assert all(all(assignment[color] in metrics["domains"][color] for color in raw["colors"])
                       for assignment in solutions)
            rows.append(dict(trace=rollout["trace"], old_logp=rollout["old_logp"],
                reward=rollout["reward"], useful=metrics["useful"], gain=metrics["gain"],
                solved=metrics["solved"], domains=metrics["domains"]))
        validation = refine_eval.evaluate(policy, data, attempts=8, samples=8, seconds=2., seed=1234)
        result = dict(useful=sum(row["useful"] for row in rows), trials=len(rows), distinct=distinct,
            mean_reward=sum(row["reward"] for row in rows) / len(rows), solution_preservation=True,
            independent_trial_rollback=True, validation=validation, rows=rows)
        write(output / f"{label}.json", result)
        print(label, result["useful"], "/", result["trials"], flush=True)
        return {key: result[key] for key in ("useful", "trials", "mean_reward", "validation")}

    before = probe(actor, "before")
    refine.fit(data, output=output / "training", initial=parent, trials=512, seed=29,
        groups=2, rollouts=8, lr=.01, normalization="group", root_fraction=1., entropy_weight=0.,
        validation_every=512, checkpoint_every=512, validation_attempts=8,
        validation_samples=8, validation_seconds=2.)
    after = probe(load(output / "training" / "final.pt"), "after")
    assert sources == refine.source_hashes()
    write(output / "summary.json", dict(before=before, after=after, sources_unchanged=True,
        global_solutions_preserved=4, assertions_passed=True,
        interpretation="A deliberate fit test on one fixed generic tensor network. "
            "The free variables prevent root solving; improved sampled pruning is not Sudoku generalization."))
    print(output, flush=True)


if __name__ == "__main__":
    run()
