"""Fixed validation preserves training state and distinguishes pruning from DFS."""
from copy import deepcopy
import random
import time

import numpy as np
import pytest
import torch

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement import refine_eval
from experiments.cluster_selection.reinforcement.model import Policy
from tnreason.engine.workload_to_numpy import NumpyCore


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup():
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = closing[3, 3] = False
    network = ConstraintNetwork({
        "ab": NumpyCore(values=np.eye(4, dtype=bool), colors=["a", "b"]),
        "bc": NumpyCore(values=np.eye(4, dtype=bool), colors=["b", "c"]),
        "ac": NumpyCore(values=closing, colors=["a", "c"])}, maintain_indices=True)
    raw = network_data(network)
    data = dict(version=1, network=raw, reward_colors=raw["colors"],
        initial_domains=[{c: list(values) for c in raw["colors"]} for values in ((0, 1), (2, 3))],
        train_indices=[0], validation_indices=[1])
    model = Policy(raw, width=8, rounds=1, contextual=True, target_first=True,
        compact_domains=True, max_cores=3, max_open_colors=3, min_active=2,
        cyclic_only=True, heuristic_candidates=0, decoder="sampling_grow")
    model.reward_colors, model.beta = raw["colors"], 0.
    return data, model


def test_independent_trials_and_root_propagation_preserve_model_rng_and_data(monkeypatch):
    data, model = setup()
    original_data, weights = deepcopy(data), deepcopy(model.state_dict())
    model.train()
    random.seed(42)
    torch.manual_seed(41)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    original = refine_eval.propagate

    def without_dfs(network, *args, **kwargs):
        def forbidden(*args, **kwargs):
            pytest.fail("Root propagation must not call DFS")
        with monkeypatch.context() as patch:
            patch.setattr(network, "solve_mac", forbidden)
            return original(network, *args, **kwargs)

    monkeypatch.setattr(refine_eval, "propagate", without_dfs)
    report = refine_eval.evaluate(model, data, attempts=3, seconds=2., samples=3, seed=17, nodes=100)
    assert report["totals"]["sampled_useful"] == report["totals"]["sampled_solved"] == 3
    assert report["totals"]["sampled_mean_reward"] == 1.
    assert report["totals"]["root_solved"] == report["totals"]["dfs_nodes"] == 1
    assert report["totals"]["root_resolved"] == 1
    assert report["rows"][0]["root"]["assignment"] == dict(a=2, b=2, c=2)
    assert report["rows"][0]["root"]["nodes"] == 0
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert model.training and data == original_data
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())


def test_baselines_use_same_residual_search_and_limits():
    data, model = setup()
    plain = refine_eval.evaluate_baseline(model, data, mode="plain", attempts=3, seconds=2., nodes=100)
    structural = refine_eval.evaluate_baseline(model, data, mode="heuristic", attempts=3, seconds=2., nodes=100)
    assert plain["totals"]["sampled_trials"] == structural["totals"]["sampled_trials"] == 0
    assert plain["totals"]["root_solved"] == 0 and plain["totals"]["dfs_nodes"] == 2
    assert structural["totals"]["root_solved"] == structural["totals"]["dfs_nodes"] == 1


@pytest.mark.parametrize("options", [dict(indices=[0]), dict(indices=[1, 1]), dict(attempts=-1),
    dict(seconds=float("nan")), dict(samples=-1), dict(nodes=True)])
def test_invalid_budgets_indices_and_training_overlap_are_rejected(options):
    data, model = setup()
    with pytest.raises(ValueError):
        refine_eval.evaluate(model, data, **options)


def test_distinct_indices_cannot_hide_identical_training_domains():
    data, model = setup()
    data["initial_domains"][1] = deepcopy(data["initial_domains"][0])
    with pytest.raises(ValueError, match="overlap"):
        refine_eval.evaluate(model, data)


def test_censored_residual_search_is_unknown_not_zero(monkeypatch):
    data, model = setup()
    original = refine_eval.solve

    def censored(*args, **kwargs):
        row = original(*args, **kwargs)
        return {**row, "status": "wall_timeout", "nodes": None, "solved": False}

    monkeypatch.setattr(refine_eval, "solve", censored)
    report = refine_eval.evaluate(model, data, samples=0, attempts=3, seconds=2.)
    assert report["totals"]["dfs_complete"] == 0 and report["totals"]["dfs_nodes"] is None


def test_rng_is_restored_when_rollout_fails(monkeypatch):
    data, model = setup()
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()

    def fail(*args, **kwargs):
        random.random()
        torch.rand(())
        raise RuntimeError("injected failure")

    monkeypatch.setattr(Policy, "rollout", fail)
    with pytest.raises(RuntimeError, match="injected"):
        refine_eval.evaluate(model, data, samples=1)
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("mode", ["plain", "heuristic"])
def test_time_only_requires_learned_proposals(mode):
    data, model = setup()
    with pytest.raises(ValueError, match="positive attempts"):
        refine_eval.evaluate_baseline(model, data, mode=mode, attempts=0)


def test_time_only_refreshes_proposals_and_removes_aggregate_work_caps(monkeypatch):
    closing = np.ones((4, 4), dtype=bool)
    closing[1, 1] = closing[3, 3] = False
    cores = {}
    for a, b, c in ("abc", "uvw"):
        cores[a + b] = NumpyCore(values=np.eye(4, dtype=bool), colors=[a, b])
        cores[b + c] = NumpyCore(values=np.eye(4, dtype=bool), colors=[b, c])
        cores[a + c] = NumpyCore(values=closing.copy(), colors=[a, c])
    raw = network_data(ConstraintNetwork(cores, maintain_indices=True))
    data = dict(version=1, network=raw, reward_colors=raw["colors"],
        initial_domains=[{c: list(values) for c in raw["colors"]} for values in ((0, 1), (2, 3))],
        train_indices=[0], validation_indices=[1])
    model = Policy(raw, width=8, rounds=1, contextual=True, target_first=True,
        compact_domains=True, max_cores=3, max_open_colors=1, min_active=2,
        cyclic_only=True, heuristic_candidates=0, decoder="sampling_grow", max_total_work=1)
    model.reward_colors, model.beta = raw["colors"], 0.
    original_config, original_weights = deepcopy(model.config), deepcopy(model.state_dict())
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    captured, original = [], Policy.proposals

    def observed(self, domains, **options):
        captured.append((sum(len(values) > 1 for values in domains.values()), options))
        return original(self, domains, **options)

    monkeypatch.setattr(Policy, "proposals", observed)
    limited = refine_eval.evaluate(model, data, attempts=1, samples=0, seconds=2.)
    assert limited["totals"]["root_solved"] == 0
    captured.clear()
    report = refine_eval.evaluate(model, data, attempts=0, samples=0, seconds=2.)
    assert report["settings"]["time_only"] and report["settings"]["attempts"] == 0
    assert report["totals"]["root_solved"] == report["totals"]["dfs_nodes"] == 1
    assert report["rows"][0]["root"]["cluster_statistics"]["selected"] == 2
    assert [active for active, _ in captured] == [6, 3]
    assert all(options == dict(max_candidates=1_000_000_000, max_steps=1_000_000_000)
               for _, options in captured)
    assert model.config == original_config and model.training
    assert all(torch.equal(value, original_weights[key]) for key, value in model.state_dict().items())
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)


def test_time_only_deadline_rolls_back_interrupted_summary_and_restores_rng(monkeypatch):
    data, model = setup()
    states, original = [], runtime.apply_candidate
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()

    def delayed(state, candidate, prepared=None, **limits):
        before = state.snapshot_domains()
        states.append((state, before))
        propagate = state.propagate

        def pause():
            assert any(key.startswith("summary_") for key in state._cores)
            time.sleep(1.)
            return propagate()

        state.propagate = pause
        try:
            return original(state, candidate, prepared, **limits)
        finally:
            del state.__dict__["propagate"]

    monkeypatch.setattr(runtime, "apply_candidate", delayed)
    report = refine_eval.evaluate(model, data, attempts=0, samples=0, seconds=.25)
    assert report["rows"][0]["root"]["status"] == "wall_timeout" and states
    assert report["totals"]["dfs_complete"] == 1
    for state, before in states:
        assert state.snapshot_domains() == before
        assert set(state._cores) == {core["key"] for core in data["network"]["cores"]}
        assert not state._added_core_keys and not state._queue and not state._queued
        assert state.decision_level == 0 and "propagate" not in state.__dict__
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("full_mac", [False, True])
@pytest.mark.parametrize("implicit_reward_colors", [False, True])
def test_cli_loads_new_actor_heads_and_checks_original_constraints(tmp_path, full_mac, implicit_reward_colors):
    import json
    from experiments.cluster_selection.reinforcement.actor import Policy as Actor
    from experiments.cluster_selection.reinforcement.experiment import instance_hashes

    data, base = setup()
    model = Actor(data["network"], **base.config, global_context=True, objective_features=True)
    model.load_state_dict(base.state_dict(), strict=False)
    model.reward_colors, model.beta = data["reward_colors"], 0.
    path, checkpoint, output = [tmp_path / name for name in ("data.json", "policy.pt", "report.json")]
    if implicit_reward_colors:
        data.pop("reward_colors")
    path.write_text(json.dumps(data))
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
        actor_config=model.actor_config, weights=model.state_dict(), reward_colors=model.reward_colors,
        beta=0., training=dict(train_instance_hashes=sorted(instance_hashes(data, [data["initial_domains"][0]])))), checkpoint)
    args = [str(path), "--checkpoint", str(checkpoint), "--output", str(output),
        "--attempts", "3" if full_mac else "0", "--seconds", "2"]
    if full_mac:
        args.append("--full-mac")
    report = refine_eval.main(args)
    assert json.loads(output.read_text()) == report
    assert report["totals"]["root_solved"] == report["totals"]["dfs_nodes"] == 1
    assert report["protocol"]["actor_config"]["global_context"]
    assert report["totals"]["root_resolved"] == 1
    assert report["protocol"]["actor_config"]["objective_features"]
    assert report["protocol"]["training_fingerprints"] == 1
    assignment = report["rows"][0]["assignment"] if full_mac else report["rows"][0]["root"]["assignment"]
    assert assignment == dict(a=2, b=2, c=2)


@pytest.mark.parametrize("exposure", [None, "overlap"])
def test_cli_refuses_missing_fingerprints_or_checkpoint_training_overlap(tmp_path, exposure):
    import json
    from experiments.cluster_selection.reinforcement.experiment import instance_hashes

    data, model = setup()
    path, checkpoint, output = [tmp_path / name for name in ("data.json", "policy.pt", "report.json")]
    path.write_text(json.dumps(data))
    metadata = {} if exposure is None else dict(training=dict(train_instance_hashes=sorted(
        instance_hashes(data, [data["initial_domains"][1]]))))
    torch.save(dict(version=1, kind=model.kind, network=model.network, config=model.config,
        weights=model.state_dict(), reward_colors=model.reward_colors, beta=0., **metadata), checkpoint)
    with pytest.raises(ValueError, match="fingerprints|overlap"):
        refine_eval.main([str(path), "--checkpoint", str(checkpoint), "--output", str(output)])
    assert not output.exists()


def test_cli_does_not_overwrite_an_existing_result(tmp_path):
    output = tmp_path / "previous.json"
    output.write_text("previous result\n")
    with pytest.raises(FileExistsError):
        refine_eval.main(["missing.json", "--checkpoint", "missing.pt", "--output", str(output)])
    assert output.read_text() == "previous result\n"


def test_unsat_root_proof_resolves_generic_instance_without_residual_dfs(monkeypatch):
    # Three pairwise inequalities cannot color a triangle with either of these
    # two-color domains. Each binary constraint nevertheless has GAC support.
    unequal = ~np.eye(4, dtype=bool)
    network = ConstraintNetwork({key: NumpyCore(values=unequal.copy(), colors=list(key))
        for key in ("ab", "bc", "ac")}, maintain_indices=True)
    raw = network_data(network)
    data = dict(version=1, network=raw, reward_colors=raw["colors"],
        initial_domains=[{color: list(values) for color in raw["colors"]}
                         for values in ((0, 1), (2, 3))],
        train_indices=[0], validation_indices=[1])
    for domains in data["initial_domains"]:
        state = network.create_mac_state(domains)
        assert state.propagate() and not state.is_solved()
        assert all(len(values) == 2 for values in state.domains.values())
    model = Policy(raw, width=8, rounds=1, contextual=True, target_first=True,
        compact_domains=True, max_cores=3, max_open_colors=1, min_active=2,
        cyclic_only=True, heuristic_candidates=0, decoder="sampling_grow")
    model.reward_colors, model.beta = raw["colors"], 0.

    def forbidden(*args, **kwargs):
        pytest.fail("A verified root contradiction must not start residual DFS")

    propagate = refine_eval.propagate

    def proof(*args, **kwargs):
        result = propagate(*args, **kwargs)
        assert result["status"] == "unsat", "The complete triangle must prove the contradiction"
        return result

    monkeypatch.setattr(refine_eval, "propagate", proof)
    monkeypatch.setattr(refine_eval, "solve", forbidden)
    report = refine_eval.evaluate(model, data, attempts=2, samples=0, seconds=2.)
    row = report["rows"][0]
    assert row["root"]["status"] == "unsat"
    assert row["residual"]["status"] == "unsat" and row["residual"]["nodes"] == 1
    assert row["residual"]["from_root_proof"] is True
    assert row["residual"]["assignment"] == {} and row["residual"]["setup_s"] == 0.
    assert row["residual"]["work"]["contractions"] == 0
    assert report["totals"]["root_solved"] == 0 and report["totals"]["root_resolved"] == 1
    assert report["totals"]["baseline_resolved"] == report["totals"]["baseline_solved"] == 0
    assert report["totals"]["dfs_complete"] == report["totals"]["dfs_nodes"] == 1
    assert report["totals"]["residual_seconds"] == 0.
    data["initial_domains"][1] = {color: [2] for color in raw["colors"]}
    immediate = refine_eval.evaluate(model, data, attempts=2, samples=0, seconds=2.)
    assert immediate["totals"]["baseline_resolved"] == immediate["totals"]["root_resolved"] == 1
    assert immediate["totals"]["baseline_solved"] == immediate["totals"]["root_solved"] == 0


@pytest.mark.parametrize("phase", ["before_puzzle", "between_samples", "after_root"])
def test_requested_validation_stop_restores_rng_model_and_data(monkeypatch, phase):
    data, model = setup()
    original_data, weights = deepcopy(data), deepcopy(model.state_dict())
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    requested, calls = [phase == "before_puzzle"], []
    rollout, propagate = Policy.rollout, refine_eval.propagate

    def forbidden(*args, **kwargs):
        pytest.fail("Requested validation stop must precede residual DFS")

    def sampled(self, *args, **kwargs):
        assert phase == "between_samples" and not requested[0]
        result = rollout(self, *args, **kwargs)
        calls.append("sample")
        requested[0] = True
        return result

    def rooted(*args, **kwargs):
        assert phase == "after_root" and not requested[0]
        result = propagate(*args, **kwargs)
        calls.append("root")
        requested[0] = True
        return result

    monkeypatch.setattr(Policy, "rollout", sampled)
    monkeypatch.setattr(refine_eval, "propagate", rooted)
    monkeypatch.setattr(refine_eval, "solve", forbidden)
    with pytest.raises(InterruptedError):
        refine_eval.evaluate(model, data, samples=3 if phase == "between_samples" else 0,
            attempts=3, seconds=2., stop_requested=lambda: requested[0])
    assert calls == ([] if phase == "before_puzzle" else ["sample" if phase == "between_samples" else "root"])
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert data == original_data and model.training
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
