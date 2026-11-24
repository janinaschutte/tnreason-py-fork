"""Rule-shaped proposals must be generic, affordable, and checked by contraction."""
from copy import deepcopy
import numpy as np
import pytest

from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate, prepare_candidate
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator as LegacyGenerator
from experiments.cluster_selection import rule_heuristic
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.polynomial_handling import PolynomialCore
from tnreason.engine.workload_to_numpy import NumpyCore


def hall_network(size=3, scramble=False):
    """k choices must occupy k category slots; each slot has one extra value."""
    rng = np.random.default_rng(19)
    names = [f"vertex_{i}" for i in range(2 * size + size * size)]
    if scramble:
        rng.shuffle(names)
    rows, targets = names[:size], names[size:2 * size]
    row_values = [rng.permutation(size) if scramble else np.arange(size) for _ in rows]
    target_values = [rng.permutation(size + 1) if scramble else np.arange(size + 1) for _ in targets]
    cores = {}
    for i, row in enumerate(rows):
        for j, target in enumerate(targets):
            bridge = names[2 * size + i * size + j]
            truth = int(rng.integers(2)) if scramble else 1
            for category, dimension, selected in ((row, size, row_values[i][j]),
                                                   (target, size + 1, target_values[j][i])):
                values = ((np.arange(dimension)[:, None] == selected)
                          == (np.arange(2)[None, :] == truth))
                colors = [category, bridge]
                if scramble:
                    values = values.astype(int) * (-3 if rng.integers(2) else 2)
                    if rng.integers(2):
                        values, colors = values.T, colors[::-1]
                cores[f"factor_{len(cores):02d}"] = NumpyCore(values=values, colors=colors)
    if scramble:
        cores = dict(reversed(tuple(cores.items())))
    return ConstraintNetwork(cores, maintain_indices=True), dict(zip(targets, (int(v[-1]) for v in target_values)))


def proposals(network, domains=None, **options):
    limits = dict(max_cores=64, max_open_colors=6, max_candidates=128,
                  max_steps=131072, max_work=1000000, max_size=100000)
    limits.update(options)
    return rule_heuristic.ClusterCandidateGenerator(network, rule_inspired=True, **limits).generate(
        domains, cyclic_only=True, compact_domains=True)


def macro_for(network, targets, state):
    candidates = proposals(network, state.domains)
    return next(candidate for candidate in candidates
                if set(candidate.core_keys) == set(network.coresDict)
                and len(candidate.open_colors) == 1 and candidate.open_colors[0] in targets)


def test_disabled_mode_matches_legacy_without_numerical_support_reads():
    class StructureOnly:
        def __init__(self, colors):
            self.colors, self.shape = colors, (3, 3)

        @property
        def values(self):
            pytest.fail("Disabled rules inspected tensor values")

        def __getitem__(self, assignment):
            pytest.fail("Disabled rules read tensor support")

    cores = {"one": StructureOnly(("u", "v")), "two": StructureOnly(("v", "w")),
             "three": StructureOnly(("w", "u"))}
    options = dict(max_cores=64, max_candidates=16, max_steps=8192)
    expected = LegacyGenerator(cores, **options).generate(cyclic_only=True)
    assert expected
    assert rule_heuristic.ClusterCandidateGenerator(cores, **options).generate(cyclic_only=True) == expected
    assert rule_heuristic.ClusterCandidateGenerator(cores, rule_inspired=False, **options).generate(cyclic_only=True) == expected


def test_dense_hall_triple_strengthens_gac_beyond_default_cycle_proposals():
    network, targets = hall_network()
    state = network.create_mac_state()
    assert state.propagate() and all(len(state.domains[c]) == 4 for c in targets)
    legacy = LegacyGenerator(network, max_cores=64, max_candidates=128, max_steps=131072).generate(
        state.domains, cyclic_only=True, compact_domains=True)
    assert legacy and all(len(c.core_keys) <= 16 for c in legacy)
    assert all(evaluate_candidate(state, c, cost_weight=0).removed_domain_values == 0 for c in legacy)
    candidate = macro_for(network, targets, state)
    assert len(candidate.core_keys) == 18
    before = state.snapshot_domains()
    result = evaluate_candidate(state, candidate, cost_weight=0, compact_domains=True)
    output = candidate.open_colors[0]
    assert targets[output] not in result.domains[output] and result.removed_domain_values > 0
    assert state.snapshot_domains() == before  # Independent evaluation rolls back.


@pytest.mark.parametrize("size,scramble", [(2, False), (3, True), (4, True)])
def test_channel_detection_ignores_names_axes_values_and_boolean_polarity(monkeypatch, size, scramble):
    # Isolate the new detector: existing cycles could otherwise hide a missed pattern.
    monkeypatch.setattr(LegacyGenerator, "generate", lambda *args, **kwargs: ())
    network, targets = hall_network(size, scramble)
    state = network.create_mac_state()
    assert state.propagate()
    candidate = macro_for(network, targets, state)
    result = evaluate_candidate(state, candidate, cost_weight=0, compact_domains=True)
    output = candidate.open_colors[0]
    assert len(candidate.core_keys) == 2 * size * size
    assert targets[output] not in result.domains[output]


def test_sparse_hall_ring_uses_current_domains_with_renamed_flipped_channels(monkeypatch):
    monkeypatch.setattr(LegacyGenerator, "generate", lambda *args, **kwargs: ())
    original, targets = hall_network()
    rename = {color: f"opaque_{14-i}" for i, color in enumerate(original.dimension_dict)}
    cores = {}
    for key, core in original.coresDict.items():
        values = core.values.copy()
        for axis, color in enumerate(core.colors):
            if original.dimension_dict[color] == 2 and int(color.split("_")[-1]) % 2:
                values = np.flip(values, axis)
        cores[key] = NumpyCore(values=values.T, colors=[rename[c] for c in core.colors[::-1]])
    network = ConstraintNetwork(cores, maintain_indices=True)
    domains = {rename[f"vertex_{i}"]: tuple(v for v in range(3) if v != (i+2) % 3) for i in range(3)}
    state = network.create_mac_state(domains)
    assert state.propagate()
    target_names = {rename[c] for c in targets}
    assert all(3 in state.domains[c] for c in target_names)
    candidate = next(c for c in proposals(network, state.domains)
                     if len(c.core_keys) == 12 and set(c.open_colors) <= target_names)
    result = evaluate_candidate(state, candidate, compact_domains=True)
    assert 3 not in result.domains[candidate.open_colors[0]] and result.removed_domain_values > 0


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_nonfunctional_channel_does_not_invent_hall_pruning(missing):
    network, _ = hall_network()
    key = next(iter(network.coresDict))
    if missing:
        del network.coresDict[key]
    else:
        network.coresDict[key].values.fill(True)
    network.rebuild_indices()
    state = network.create_mac_state()
    assert state.propagate()
    candidates = proposals(network, state.domains)
    assert all(len(c.core_keys) < 18 for c in candidates)
    assert all(evaluate_candidate(state, c, cost_weight=0, compact_domains=True).removed_domain_values == 0
               for c in candidates)


def test_all_parallel_original_cores_are_included_and_intersected():
    network, targets = hall_network()
    original = next(iter(network.coresDict.values()))
    network.coresDict["parallel"] = NumpyCore(values=-2 * np.ones(original.shape[::-1]),
                                             colors=list(original.colors[::-1]))
    network.rebuild_indices()
    state = network.create_mac_state()
    assert state.propagate()
    candidate = macro_for(network, targets, state)
    assert "parallel" in candidate.core_keys and len(candidate.core_keys) == 19
    assert evaluate_candidate(state, candidate, compact_domains=True).removed_domain_values > 0
    # Keeping only the all-ones parallel factor would not establish a channel.
    first = next(iter(network.coresDict))
    network.coresDict[first].values.fill(True)
    assert all(len(c.core_keys) < 18 for c in proposals(network))


def test_sparse_cancellation_and_consumed_iterators_are_preserved(monkeypatch):
    network, targets = hall_network()
    state = network.create_mac_state()
    assert state.propagate()
    dense = next(iter(network.coresDict.values()))
    untouched = list(network.coresDict.values())[1]
    next(untouched)
    index = untouched.index
    key = next(iter(network.coresDict))
    entries = [(int(dense.values[idx]), dict(zip(dense.colors, idx))) for idx in np.ndindex(dense.shape)
               if dense.values[idx]]
    zero = tuple(np.argwhere(dense.values == 0)[0])
    cancelled = dict(zip(dense.colors, zero))
    entries += [(5, cancelled), (-5, cancelled)]
    sparse = PolynomialCore(values=deepcopy(entries), colors=list(dense.colors), shape=dense.shape)
    network.coresDict[key] = sparse
    monkeypatch.setattr(PolynomialCore, "__iter__", lambda self: pytest.fail("Consumed sparse iterator"))
    candidate = macro_for(network, targets, state)
    assert key in candidate.core_keys and sparse.values == entries and untouched.index == index
    # Evaluate against the unchanged dense snapshot, not the poisoned sparse backend.
    assert evaluate_candidate(state, candidate, compact_domains=True).removed_domain_values > 0


def test_support_scan_and_contraction_budgets_are_enforced():
    class HugeLookup:
        colors, shape = ("u", "flag"), (100001, 2)
        values = None

        def __getitem__(self, assignment):
            pytest.fail("Oversized input was read")

    assert proposals({"huge": HugeLookup()}, max_size=100000) == ()
    network, _ = hall_network()
    assert proposals(network, max_work=1) == ()
    assert proposals(network, max_size=1) == ()
    candidates = proposals(network, max_candidates=2)
    assert len(candidates) <= 2
    state = network.create_mac_state()
    assert state.propagate()
    for candidate in candidates:
        estimate = prepare_candidate(state, candidate, compact_domains=True)[0]
        assert estimate.work <= 1000000 and estimate.max_size <= 100000


def test_high_degree_indicator_catalog_has_a_pair_budget():
    table = ((np.arange(3)[:, None] == 0) == (np.arange(2)[None, :] == 1))
    cores = {f"factor_{i}": NumpyCore(values=table, colors=[f"category_{i}", "shared_flag"])
             for i in range(100)}
    generator = rule_heuristic.ClusterCandidateGenerator(cores, rule_inspired=True,
        max_steps=16, max_candidates=2, max_work=100000)
    generator.generate(cyclic_only=True)
    assert 0 < generator.rule_statistics["catalog_pairs"] <= 16
    assert generator.rule_statistics["catalog_pairs_capped"] == 1
    assert generator.rule_statistics["catalog_lookup_work"] <= 100000


def test_overlap_cluster_and_backtracking_keep_pruning_without_summaries(monkeypatch):
    monkeypatch.setattr(LegacyGenerator, "generate", lambda *args, **kwargs: ())
    cores = {}
    for value in range(2):
        for category in ("first", "second"):
            table = ((np.arange(4)[:, None] == value) == (np.arange(2)[None, :] == 1))
            cores[f"{category}_{value}"] = NumpyCore(values=table, colors=[category, f"flag_{value}"])
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate()
    root = state.snapshot_domains()
    state.push_level()
    state.restrict_domain("first", (0, 1))
    assert state.propagate() and state.domains["second"] == (0, 1, 2, 3)
    rule_heuristic.attach_heuristic(state, rule_inspired=True, max_cores=64,
                                   max_candidates=64, max_steps=131072, compact_domains=True)
    assert state.propagate() and state.domains["second"] == (0, 1)
    assert set(state._cores) == set(cores) and not state._temporary_core_keys
    state.pop_level()
    state.on_closure = None
    assert state.snapshot_domains() == root and state.decision_level == 0
    assert network.coresDict == cores


def cascade_network():
    # The first projection removes a=0; external propagation fixes z=1.
    # Recontracting the SAME cluster then excludes a=1, which GAC alone cannot.
    ac = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    relation = np.zeros((2, 2, 2), dtype=bool)
    relation[0, 1, :] = relation[1, 0, :] = True
    relation[1, 1, 0] = True
    cores = {"ab": NumpyCore(values=ac, colors=["a", "b"]),
             "ac": NumpyCore(values=ac, colors=["a", "c"]),
             "bcz": NumpyCore(values=relation, colors=["b", "c", "z"]),
             "external": NumpyCore(values=[[1, 0], [0, 1], [0, 1]], colors=["a", "z"])}
    return ConstraintNetwork(cores, maintain_indices=True)


def test_policy_regenerates_before_applying_an_old_batch_remainder(monkeypatch):
    network = cascade_network()
    state = network.create_mac_state()
    assert state.propagate() and state.domains["a"] == (0, 1, 2)
    first = SummaryClusterCandidate("first", ("ab", "ac", "bcz"), ("a",))
    stale = SummaryClusterCandidate("stale", first.core_keys, ("b",))
    fresh = SummaryClusterCandidate("fresh", tuple(network.coresDict), ("a",))
    snapshots, applications = [], []
    policy = rule_heuristic.HeuristicPolicy(state, rule_inspired=False, max_candidates=4,
        compact_domains=True, on_application=lambda s, c, e, before, result: applications.append(c.key))
    def generate(domains, **options):
        snapshots.append(dict(domains))
        return (first, stale) if len(domains["a"]) == 3 else (fresh,) if len(domains["a"]) == 2 else ()
    monkeypatch.setattr(policy.generator, "generate", generate)
    state.on_closure = policy
    assert state.propagate() and state.domains["a"] == (2,)
    assert applications == ["first", "fresh"]
    assert snapshots[1]["a"] == (1, 2) and snapshots[1]["z"] == (1,)
    assert set(state._cores) == set(network.coresDict) and not state._temporary_core_keys


def test_useful_cluster_can_repeat_after_internal_cascade_and_sibling_rollback(monkeypatch):
    network = cascade_network()
    state = network.create_mac_state()
    assert state.propagate()
    root = state.snapshot_domains()
    candidate = SummaryClusterCandidate("repeat", ("ab", "ac", "bcz"), ("a",))
    observations = []
    def observed(s, c, estimate, before, result):
        assert s.decision_level == 1 and set(s._cores) == set(network.coresDict)
        observations.append((before["a"], before["z"], result.removed_values))
    policy = rule_heuristic.HeuristicPolicy(state, max_candidates=4, compact_domains=True,
                                           on_application=observed)
    monkeypatch.setattr(policy.generator, "generate", lambda *a, **k: (candidate,))
    for _ in range(2):
        state.push_level()
        state.on_closure = policy
        assert state.propagate() and state.domains["a"] == (2,) and state.domains["z"] == (1,)
        state.pop_level()
        state.on_closure = None
        assert state.snapshot_domains() == root and not state._temporary_core_keys
    positive = [row for row in observations if row[-1]]
    assert positive == [((0, 1, 2), (0, 1), 2), ((1, 2), (1,), 1)] * 2


def test_refresh_keeps_unexplored_independent_components(monkeypatch):
    monkeypatch.setattr(LegacyGenerator, "generate", lambda *args, **kwargs: ())
    cores = {}
    for prefix in ("left", "right"):
        for value in range(2):
            for category in ("source", "target"):
                table = ((np.arange(4)[:, None] == value) == (np.arange(2)[None, :] == 1))
                cores[f"{prefix}_{category}_{value}"] = NumpyCore(values=table,
                    colors=[f"{prefix}_{category}", f"{prefix}_flag_{value}"])
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state({f"{p}_source": (0, 1) for p in ("left", "right")})
    assert state.propagate()
    rule_heuristic.attach_heuristic(state, rule_inspired=True, max_candidates=32,
                                   max_cores=64, max_steps=131072, compact_domains=True)
    generator = state.on_closure.generator
    generate, calls = generator.generate, []
    def recorded(domains, **options):
        calls.append((dict(domains), set(options["changed_colors"])))
        return generate(domains, **options)
    monkeypatch.setattr(generator, "generate", recorded)
    assert state.propagate()
    assert state.domains["left_target"] == state.domains["right_target"] == (0, 1)
    second_domains, changed = calls[1]
    unexplored = next(p for p in ("left", "right") if len(second_domains[f"{p}_target"]) == 4)
    assert any(color.startswith(unexplored) for color in changed)
