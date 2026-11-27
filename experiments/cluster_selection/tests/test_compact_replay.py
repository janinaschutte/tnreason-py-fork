"""Compact teachers preserve complete deductions and actual sequential states."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection import runtime
from tnreason.engine.workload_to_numpy import NumpyCore

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement.actor import Policy
from experiments.cluster_selection.reinforcement.compact import compact_success


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def problem(*, second=False, decoy=True):
    tables = {"ab": np.eye(2, dtype=bool), "bc": np.eye(2, dtype=bool),
              "ac": np.array([[1, 1], [1, 0]], dtype=bool)}
    if decoy:
        tables["ax"] = np.ones((2, 2), dtype=bool)
    if second:
        tables.update(uv=np.eye(2, dtype=bool), vw=np.eye(2, dtype=bool),
                      uw=np.array([[1, 1], [1, 0]], dtype=bool),
                      au=np.ones((2, 2), dtype=bool))
    network = ConstraintNetwork({key: NumpyCore(values=value, colors=list(key))
        for key, value in tables.items()}, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    model = Policy(network_data(network), width=8, rounds=1, contextual=True,
        target_first=True, exploration=0., heuristic_candidates=0,
        decoder="sampling_grow", min_active=2, compact_domains=True,
        cyclic_only=True, max_cores=len(tables), max_open_colors=3,
        learned_stopping=True).eval()
    model.reward_colors = ["a"]  # All other variables are auxiliary reward colors.
    candidate = SummaryClusterCandidate("full", tuple(tables), ("a", "u") if second else ("a",))
    return model, state, candidate


def assert_replay(model, state, examples, expected):
    original = state.snapshot_domains()
    state.push_level()
    try:
        for item in examples:
            assert item.domains == tuple(state.domains[color] for color in model.network["colors"])
            with torch.no_grad():
                candidate, logp, _, trace = model.rollout(item.domains, trace=item.trace,
                    policy_trace=True, beta=0.)
            assert tuple(trace) == item.trace and torch.isfinite(logp)
            assert set(candidate.core_keys) == set(item.core_keys)
            assert set(candidate.open_colors) == set(item.open_colors)
            before = state.snapshot_domains()
            runtime.apply_candidate(state, candidate, compact_domains=True)
            assert sum(len(before[color]) - len(state.domains[color]) for color in before) == item.removed_values
            assert item.removed_values > 0
            assert not state._temporary_core_keys and not state._added_core_keys
        assert state.snapshot_domains() == expected
    finally:
        state.pop_level()
    assert state.snapshot_domains() == original


def test_shortening_preserves_auxiliary_pruning_free_policy_support_and_caller_state():
    model, state, candidate = problem()
    before, statistics = state.snapshot_domains(), state.statistics
    originals = {key: core.values.copy() for key, core in state._cores.items()}
    hook = lambda *_: pytest.fail("Simplification invoked a runtime policy hook")
    state.on_closure = hook
    state.push_level()  # Also exercise use within an existing DFS branch.
    target = evaluate_candidate(state, candidate, compact_domains=True)
    result = compact_success(model, state, candidate, max_probes=16)
    assert result.original_cores == 4 and result.final_cores == 3
    assert result.original_removed == 3 and result.rejected == 0
    assert len(result.examples) == 1 and result.examples[0].core_keys == ("ab", "bc", "ac")
    assert result.examples[0].removed_values == 3  # Includes b and c, despite reward_colors=[a].
    assert result.probes <= 16 and result.cost_seconds > 0
    assert state.snapshot_domains() == before and state.statistics == statistics
    assert state.on_closure is hook and state.decision_level == 1
    assert not state._temporary_core_keys and not state._added_core_keys
    for key, values in originals.items():
        np.testing.assert_array_equal(state._cores[key].values, values)
    with pytest.raises(FrozenInstanceError):
        result.examples[0].removed_values = 100
    state.on_closure = None
    assert_replay(model, state, result.examples, target.domains)
    state.pop_level()


def test_disconnected_simplification_splits_into_useful_sequential_teachers():
    model, state, candidate = problem(second=True, decoy=False)
    target = evaluate_candidate(state, candidate, compact_domains=True)
    before, statistics = state.snapshot_domains(), state.statistics
    result = compact_success(model, state, candidate, max_probes=32, split=True)
    assert len(result.examples) == 2 and result.final_cores == 6 < result.original_cores
    assert all(len(item.core_keys) == 3 for item in result.examples)
    assert [item.removed_values for item in result.examples] == [3, 3]
    assert result.examples[0].domains != result.examples[1].domains
    assert result.probes <= 32
    assert state.snapshot_domains() == before and state.statistics == statistics
    assert_replay(model, state, result.examples, target.domains)


@pytest.mark.parametrize("maximum", [1, 2, 3, 6])
def test_shared_probe_limit_is_strict_and_fallback_keeps_full_deduction(maximum):
    model, state, candidate = problem(second=True)
    target = evaluate_candidate(state, candidate, compact_domains=True)
    result = compact_success(model, state, candidate, max_probes=maximum, split=True)
    assert result.probes <= maximum and result.examples
    assert_replay(model, state, result.examples, target.domains)


def test_uninformative_and_unreachable_constructions_do_not_enter_replay():
    model, state, candidate = problem()
    decoy = SummaryClusterCandidate("noop", ("ax",), ("a",))
    result = compact_success(model, state, decoy)
    assert result.examples == () and result.original_removed == 0
    # A complete contracted conjunction can prune even when the constructor's
    # minimum active-core scope makes that construction unreachable.
    model.config["min_active"] = 3
    result = compact_success(model, state, candidate)
    assert result.examples == () and result.original_removed == 3


def test_rejected_trial_is_counted_and_caller_remains_unchanged():
    model, state, candidate = problem()
    before, statistics = state.snapshot_domains(), state.statistics
    model.config["max_work"] = 0
    result = compact_success(model, state, candidate, max_probes=2)
    assert result.examples == () and result.probes == result.rejected == 1
    assert state.snapshot_domains() == before and state.statistics == statistics


def test_necessary_long_cycle_remains_available_and_is_not_truncated():
    colors = tuple("abcdefgh")
    tables = {left + right: NumpyCore(values=np.eye(2, dtype=bool), colors=[left, right])
              for left, right in zip(colors, colors[1:])}
    tables["ah"] = NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool), colors=["a", "h"])
    network = ConstraintNetwork(tables, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate() and all(len(values) == 2 for values in state.domains.values())
    model = Policy(network_data(network), width=8, rounds=1, contextual=True,
        target_first=True, exploration=0., heuristic_candidates=0, decoder="sampling_grow",
        compact_domains=True, cyclic_only=True, max_cores=8, max_open_colors=2,
        learned_stopping=True).eval()
    candidate = SummaryClusterCandidate("long", tuple(tables), ("a",))
    result = compact_success(model, state, candidate, max_probes=32, split=True)
    assert result.original_cores == result.final_cores == 8
    assert len(result.examples) == 1 and result.original_removed == 8
    assert_replay(model, state, result.examples, {color: (0,) for color in colors})


def test_leaf_peeling_saves_probes_but_still_verifies_the_complete_outcome():
    tables = {"ab": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"]),
              "bc": NumpyCore(values=np.eye(2, dtype=bool), colors=["b", "c"]),
              "ac": NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool), colors=["a", "c"])}
    previous = "c"
    for idx in range(20):
        following = f"z{idx}"
        tables[f"tail{idx}"] = NumpyCore(values=np.eye(2, dtype=bool), colors=[previous, following])
        previous = following
    network = ConstraintNetwork(tables, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    model = Policy(network_data(network), width=8, rounds=1, contextual=True,
        target_first=True, exploration=0., heuristic_candidates=0, decoder="sampling_grow",
        compact_domains=True, cyclic_only=True, max_cores=len(tables), max_open_colors=2,
        learned_stopping=True).eval()
    candidate = SummaryClusterCandidate("tree", tuple(tables), ("a",))
    target = evaluate_candidate(state, candidate, compact_domains=True)
    result = compact_success(model, state, candidate, max_probes=8)
    assert result.original_cores == 23 and result.final_cores == 3
    assert result.probes <= 5 and result.original_removed == 23
    assert_replay(model, state, result.examples, target.domains)


def test_contradictions_are_not_mislabelled_as_exact_pruning_teachers():
    network = ConstraintNetwork({key: NumpyCore(values=~np.eye(2, dtype=bool), colors=list(key))
        for key in ("ab", "bc", "ac")}, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    model = Policy(network_data(network), width=8, rounds=1, contextual=True,
        target_first=True, exploration=0., heuristic_candidates=0, decoder="sampling_grow",
        compact_domains=True, cyclic_only=True, max_cores=3, max_open_colors=2,
        learned_stopping=True).eval()
    candidate = SummaryClusterCandidate("odd", ("ab", "bc", "ac"), ("a",))
    before = state.snapshot_domains()
    result = compact_success(model, state, candidate)
    assert not result.examples and result.original_removed == 0
    assert not state.contradiction and state.snapshot_domains() == before


def test_invalid_settings_and_nonclosed_states_are_rejected():
    model, state, candidate = problem()
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            compact_success(model, state, candidate, max_probes=limit)
    model.actor_config["learned_stopping"] = False
    with pytest.raises(ValueError, match="learned stopping"):
        compact_success(model, state, candidate)
    model.actor_config["learned_stopping"] = True
    state.restrict_domain("a", (0,))
    with pytest.raises(ValueError, match="GAC-closed"):
        compact_success(model, state, candidate)
