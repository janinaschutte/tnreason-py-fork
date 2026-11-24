"""Measured rewards, sampled likelihoods and exact CPU training continuation."""
from copy import deepcopy
import json
import random
import signal

import numpy as np
import pytest

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.data import network_data
from tnreason.engine.workload_to_numpy import NumpyCore

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import Policy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def problem(decoys=True):
    # GAC cannot combine a=b=c with the ban on (a,c)=(1,1).
    tables = {"ab": np.eye(2, dtype=bool), "bc": np.eye(2, dtype=bool),
              "ac": np.array([[1, 1], [1, 0]], dtype=bool)}
    if decoys:
        tables.update({key: np.ones((2, 2), dtype=bool) for key in ("ax", "xy", "ay")})
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key))
        for key, values in tables.items()}, maintain_indices=True)
    raw = network_data(network)
    initial = [{color: [0, 1] for color in raw["colors"]}]
    if decoys:
        initial.extend([{**initial[0], "x": [value]} for value in (0, 1)])
    return network, dict(version=1, network=raw, reward_colors=list("abc"), initial_domains=initial,
        train_indices=[0, 1] if decoys else [0], validation_indices=[2] if decoys else [])


def parent_checkpoint(data, path):
    torch.manual_seed(29)
    actor = Policy(data["network"], width=8, rounds=1, contextual=True, target_first=True,
        exploration=0., heuristic_candidates=0, decoder="sampling_grow", min_active=2,
        compact_domains=True, cyclic_only=True, max_cores=3, max_open_colors=2)
    actor.reward_colors, actor.beta = data["reward_colors"], 0.
    torch.save(dict(version=1, kind=actor.kind, network=actor.network, config=actor.config,
        weights=actor.state_dict(), reward_colors=actor.reward_colors, beta=actor.beta,
        training=dict(train_instance_hashes=[])), path)
    return actor


def test_collected_rewards_are_independent_and_likelihoods_match_generation(tmp_path):
    network, data = problem()
    actor = parent_checkpoint(data, tmp_path / "parent.pt")
    state = network.create_mac_state()
    assert state.propagate()
    sample = dict(domains=list(state.domains.values()), root=True, known_satisfiable=True)
    before = state.snapshot_domains(), tuple(state._cores), tuple(state._trail), state.statistics
    group, outcomes, distinct = refine.collect(actor, state, sample, rollouts=32, solve_bonus=1.)
    assert 0 < sum(item["useful"] for item in outcomes) < len(outcomes)
    assert distinct > 1 and all(not item["solved"] for item in outcomes)
    for item, outcome in zip(group["rollouts"], outcomes):
        _, logp, _, _ = actor.rollout(group["domains"], beta=0., trace=item["trace"], policy_trace=True)
        assert float(logp) == item["old_logp"]
        assert item["reward"] == outcome["reward"]
    assert (state.snapshot_domains(), tuple(state._cores), tuple(state._trail)) == before[:3]
    assert state.statistics.propagation_calls == before[3].propagation_calls + 1
    assert state.statistics.revisions == before[3].revisions and not state._queue


def test_solve_bonus_survives_pruning_cap_but_does_not_reward_already_solved_state(tmp_path):
    network, data = problem(decoys=False)
    actor = parent_checkpoint(data, tmp_path / "parent.pt")
    state = network.create_mac_state()
    assert state.propagate() and not state.is_solved()
    sample = dict(domains=list(state.domains.values()), known_satisfiable=True)
    group, outcomes, _ = refine.collect(actor, state, sample, rollouts=4, solve_bonus=1.)
    assert all(item["solved"] and item["reward"] == 1. for item in outcomes)
    assert all(item["reward"] == 2. for item in group["rollouts"])
    solved = dict(domains=[[0]] * 3, known_satisfiable=True)
    group, outcomes, _ = refine.collect(actor, state, solved, rollouts=2, solve_bonus=1.)
    assert all(not item["solved"] and not item["useful"] for item in outcomes)
    assert all(item["reward"] == 0. for item in group["rollouts"])


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_exact_resume_restores_custom_architecture_optimizer_sampling_and_contexts(tmp_path, monkeypatch):
    _, data = problem()
    data["reward_colors"] = ["a"]  # Exercise the learned objective adapter during continuation.
    parent = tmp_path / "parent.pt"
    parent_checkpoint(data, parent)

    def quick_validation(*args, **kwargs):
        return dict(totals=dict(root_solved=0, dfs_complete=1, dfs_nodes=3,
            instances=1, sampled_useful=0, sampled_trials=0))

    # Isolate continuation from validation cost; real validation is tested below.
    monkeypatch.setattr(refine_eval, "evaluate", quick_validation)
    settings = dict(seed=29, groups=2, rollouts=4, lr=.001, global_context=True, objective_features=True,
        retain_graphs=True,
        root_fraction=.5, normalization="group", validation_every=32,
        checkpoint_every=16, entropy_weight=.001, solve_bonus=1.)
    refine.fit(data, output=tmp_path / "full", initial=parent, trials=64, **settings)
    refine.fit(data, output=tmp_path / "part", initial=parent, trials=32, **settings)
    result = refine.fit(data, output=tmp_path / "resumed", resume=tmp_path / "part" / "resume.pt",
                        trials=64, **settings)
    assert result["trials"] == 64 and not result["interrupted"]
    full, resumed = [torch.load(tmp_path / name / "resume.pt", weights_only=True)
                     for name in ("full", "resumed")]
    assert full["optimizer"]["state"]  # Equality must include actual learned Adam moments.
    assert full["weights"]["objective_role.weight"].abs().sum() > 0
    assert any(full["contexts"].values())  # Include policy-created state sampling in the check.
    for key in ("weights", "optimizer", "config", "actor_config", "completed", "batches", "scale",
                "contexts", "pending", "rng", "torch_rng", "cuda_rng"):
        assert_same(full[key], resumed[key])
    for left, right in zip(full["history"], resumed["history"]):
        assert_same({key: value for key, value in left.items() if key != "seconds"},
                    {key: value for key, value in right.items() if key != "seconds"})
    records = [json.loads(line) for line in (tmp_path / "resumed" / "training.jsonl").read_text().splitlines()]
    assert [item["fresh_trials"] for item in records] == [40, 48, 56, 64]
    assert all(item["trials"] == 8 for item in records)
    with pytest.raises(ValueError, match="settings mismatch"):
        refine.fit(data, output=tmp_path / "bad_settings", resume=tmp_path / "part" / "resume.pt",
                   trials=64, **{**settings, "lr": .002})
    hashes = refine.source_hashes()
    monkeypatch.setattr(refine, "source_hashes", lambda: {**hashes, "changed.py": "new-source"})
    with pytest.raises(ValueError, match="source changed"):
        refine.fit(data, output=tmp_path / "bad_sources", resume=tmp_path / "part" / "resume.pt",
                   trials=64, **settings)


def test_real_generic_validation_keeps_model_rng_and_train_split_unchanged(tmp_path):
    _, data = problem()
    actor = parent_checkpoint(data, tmp_path / "parent.pt").train()
    weights = deepcopy(actor.state_dict())
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    result = refine_eval.evaluate(actor, data, attempts=4, samples=4, seconds=2., seed=23, nodes=100)
    assert result["totals"]["instances"] == result["totals"]["dfs_complete"] == 1
    assert result["totals"]["sampled_trials"] == 4
    assert result["totals"]["dfs_nodes"] >= 1
    assert actor.training and torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert_same(weights, actor.state_dict())
    changed = deepcopy(data)
    changed["initial_domains"][2] = changed["initial_domains"][0]
    with pytest.raises(ValueError, match="overlap"):
        refine_eval.evaluate(actor, changed, attempts=4, samples=0, seconds=2.)
    with pytest.raises(ValueError, match="overlap"):
        refine.fit(changed, output=tmp_path / "leak", trials=8, groups=2, rollouts=4)


@pytest.mark.parametrize("terminate", [False, True])
def test_initial_validation_has_resume_before_interrupt_and_restores_handlers(tmp_path, monkeypatch, terminate):
    _, data = problem()
    parent, output = tmp_path / "parent.pt", tmp_path / "run"
    parent_checkpoint(data, parent)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(True)
        assert (output / "resume.pt").exists(), "Checkpoint must precede even the initial validation"
        saved = torch.load(output / "resume.pt", weights_only=True)
        assert saved["completed"] == saved["batches"] == 0
        assert not kwargs["stop_requested"]()
        if terminate:
            signal.raise_signal(signal.SIGTERM)
            assert kwargs["stop_requested"]()
        raise InterruptedError("validation interrupted")

    monkeypatch.setattr(refine_eval, "evaluate", interrupted)
    options = dict(output=output, initial=parent, trials=8, groups=2, rollouts=4,
                   validation_every=8, checkpoint_every=8)
    if terminate:
        result = refine.fit(data, **options)
        assert result["trials"] == 0 and result["interrupted"]
        assert torch.load(output / "resume.pt", weights_only=True)["completed"] == 0
    else:
        # An unrelated InterruptedError is a real failure, not a graceful stop.
        with pytest.raises(InterruptedError, match="validation interrupted"):
            refine.fit(data, **options)
    assert calls == [True]
    assert {number: signal.getsignal(number) for number in handlers} == handlers
    assert not (output / "validation_000000000.json").exists()


def test_root_checkpoint_selection_counts_unsat_resolution(tmp_path, monkeypatch):
    _, data = problem()
    parent = tmp_path / "parent.pt"
    parent_checkpoint(data, parent)
    calls = []

    def validation(*args, **kwargs):
        resolved = int(bool(calls))
        calls.append(resolved)
        return dict(totals=dict(root_solved=0, root_resolved=resolved, dfs_complete=1,
            dfs_nodes=2, instances=1, sampled_useful=0, sampled_trials=0))

    monkeypatch.setattr(refine_eval, "evaluate", validation)
    result = refine.fit(data, output=tmp_path / "run", initial=parent, trials=8,
        groups=2, rollouts=4, validation_every=8, checkpoint_every=8)
    assert calls == [0, 1]
    assert result["best"]["root"]["trials"] == 8
    # Existing quick-validation fixtures omit root_resolved and continue testing
    # the root_solved fallback during exact resume above.


def test_omitted_reward_colors_default_to_all_variables_for_new_and_loaded_policies(tmp_path):
    _, data = problem()
    data["reward_colors"] = data["network"]["colors"]
    parent = tmp_path / "parent.pt"
    parent_checkpoint(data, parent)
    data.pop("reward_colors")
    for initial in (None, parent):
        model = refine.initialize(data, initial=initial)
        assert model.reward_colors == data["network"]["colors"]
        report = refine_eval.evaluate(model, data, attempts=2, samples=0, seconds=2., nodes=100)
        assert report["totals"]["dfs_complete"] == 1


def test_stdout_usefulness_aggregates_the_reporting_window(tmp_path, monkeypatch, capsys):
    _, data = problem()
    parent = tmp_path / "parent.pt"
    parent_checkpoint(data, parent)
    monkeypatch.setattr(refine_eval, "evaluate", lambda *args, **kwargs: dict(totals=dict(
        root_solved=0, dfs_complete=1, dfs_nodes=3, instances=1, sampled_useful=0, sampled_trials=0)))
    refine.fit(data, output=tmp_path / "run", initial=parent, trials=32, groups=2, rollouts=4,
        root_fraction=1., validation_every=32, checkpoint_every=32)
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("batch 4,"))
    assert "window useful root " in line and "/32, context 0/0;" in line
    assert [json.loads(line)["trials"] for line in (tmp_path / "run" / "training.jsonl").read_text().splitlines()] == [8] * 4
