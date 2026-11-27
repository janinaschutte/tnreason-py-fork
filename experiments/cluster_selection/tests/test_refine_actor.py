"""Context corrections preserve construction semantics and have useful gradients."""
import numpy as np
import pytest

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.data import network_data
from tnreason.engine.workload_to_numpy import NumpyCore

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement.actor import Policy, load
from experiments.cluster_selection.reinforcement.model import Policy as BasePolicy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(**options):
    network = ConstraintNetwork({key: NumpyCore(values=np.ones((2, 2), dtype=bool), colors=list(key))
        for key in ("ab", "bc", "cd", "ad", "bd")}, maintain_indices=True)
    return Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
        heuristic_candidates=0, exploration=0., max_cores=4, max_open_colors=2,
        decoder="sampling_grow", **options).eval()


@pytest.fixture
def feature_actor():
    network = ConstraintNetwork({"xy": NumpyCore(values=np.ones((2, 3), dtype=bool),
        colors=["x", "y"])}, maintain_indices=True)
    return Policy(network_data(network), width=8, rounds=1).eval()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
@pytest.mark.parametrize("domains", [
    [[0, 1], [0, 2]], ((1,), (2,)), {"y": (2, 0), "x": (1,)},
    [np.array([1, 0], dtype=np.int64), np.array([2], dtype=np.int32)],
    [[0, 0, 1], [2, 0, 2]], [[], []], [[1], []], [{0, 1}, {1, 2}],
], ids=["list", "tuple", "mapping", "numpy-integers", "duplicates", "empty", "partly-empty", "sets"])
def test_domain_features_preserve_materialized_inputs(feature_actor, domains, device):
    actor = feature_actor.to(device)
    expected = BasePolicy.domain_features(actor, domains)
    actual = actor.domain_features(domains)
    assert actual.device.type == device
    assert actual.dtype == expected.dtype and torch.equal(actual, expected)


@pytest.mark.parametrize("domains,error,message", [
    ([[0]], ValueError, "Expected every variable"),
    ({"x": [0]}, KeyError, "y"),
    ([[True], [0]], ValueError, "outside variable dimension"),
    ([[np.bool_(True)], [0]], ValueError, "outside variable dimension"),
    ([[-1], [0]], ValueError, "outside variable dimension"),
    ([[2], [0]], ValueError, "outside variable dimension"),
    ([[0], [3]], ValueError, "outside variable dimension"),
    ([[1.], [0]], ValueError, "outside variable dimension"),
    ([["1"], [0]], ValueError, "outside variable dimension"),
    ([None, [0]], TypeError, "not iterable"),
])
def test_domain_features_validate_each_value(feature_actor, domains, error, message):
    with pytest.raises(error, match=message):
        feature_actor.domain_features(domains)


@pytest.mark.parametrize("mapping", [False, True])
def test_domain_features_materialize_one_shot_values_before_validation(feature_actor, mapping):
    materialized = [[1, 0, 1], [2, 0]]
    domains = [iter(values) for values in materialized]
    if mapping:
        domains = dict(zip(feature_actor.network["colors"], domains))
    expected = BasePolicy.domain_features(feature_actor, materialized)
    assert torch.equal(feature_actor.domain_features(domains), expected)
    invalid = [iter([0]), iter([2, 3])]
    with pytest.raises(ValueError, match="outside variable dimension"):
        feature_actor.domain_features(invalid)


@pytest.mark.parametrize("global_context,distance_features,objective_features", [
    (False, False, False), (True, False, False), (False, True, False),
    (True, True, False), (False, False, True), (True, True, True)])
def test_zero_corrections_preserve_base_policy_and_masks(global_context, distance_features, objective_features):
    torch.manual_seed(19)
    actor = model(global_context=global_context, distance_features=distance_features,
                  objective_features=objective_features)
    base = BasePolicy(actor.network, **actor.config).eval()
    base.load_state_dict({key: value for key, value in actor.state_dict().items()
                         if not key.startswith(("anchor_context.", "core_distance.", "objective_role."))})
    domains = [[0, 1]] * len(actor.dimensions)
    for keys, opened, phase in (([], [], False), ([0, 1], [0], False), ([0, 1, 2], [0], True)):
        left = actor.distribution(actor.encode(domains), [2] * 4, keys, opened, phase, 0.)
        right = base.distribution(base.encode(domains), [2] * 4, keys, opened, phase, 0.)
        assert left[0] == right[0]
        torch.testing.assert_close(left[1].probs, right[1].probs, atol=2e-7, rtol=0)
    assert "global_context" not in actor.config and "distance_features" not in actor.config
    assert actor.kind == "construction"


def test_objective_roles_are_visible_have_gradients_and_uniform_rewards_remain_unchanged():
    tables = {}
    for a, b, c in ("abc", "uvw"):
        tables.update({a + b: np.eye(2, dtype=bool), b + c: np.eye(2, dtype=bool),
                       a + c: np.array([[1, 1], [1, 0]], dtype=bool)})
    network = ConstraintNetwork({key: NumpyCore(values=values, colors=list(key))
        for key, values in tables.items()}, maintain_indices=True)
    torch.manual_seed(29)
    actor = Policy(network_data(network), width=8, rounds=1, contextual=True,
        target_first=True, objective_features=True, exploration=0., heuristic_candidates=0).eval()
    actor.reward_colors = ["a"]
    domains = [[0, 1]] * 6
    base, encoded = BasePolicy.encode(actor, domains), actor.encode(domains)
    torch.testing.assert_close(base[0], encoded[0], atol=0, rtol=0)
    actions, distribution = actor.distribution(encoded, [2] * 6, [], [], False, 0.)
    a, u = [actions.index(("open", actor.color_idx[color])) for color in ("a", "u")]
    torch.testing.assert_close(distribution.logits[a], distribution.logits[u], atol=0, rtol=0)
    (-distribution.log_prob(torch.tensor(a))).backward()
    assert torch.isfinite(actor.objective_role.weight.grad).all()
    assert actor.objective_role.weight.grad.abs().sum() > 0
    with torch.no_grad():
        actor.objective_role.weight.fill_(1.)
    changed = actor.encode(domains)
    # Primary weight is 1; each of five auxiliary variables has weight 1/5.
    torch.testing.assert_close(changed[0][actor.color_idx["a"]], base[0][actor.color_idx["a"]], atol=0, rtol=0)
    torch.testing.assert_close(changed[0][actor.color_idx["u"]] - base[0][actor.color_idx["u"]],
                               torch.full((8,), float(np.log(.2))), atol=1e-6, rtol=0)
    _, corrected = actor.distribution(changed, [2] * 6, [], [], False, 0.)
    assert abs(float(corrected.logits[a] - corrected.logits[u])) > 1e-8
    for targets in (actor.network["colors"], None):
        actor.reward_colors = targets
        actual = actor.encode(domains)
        torch.testing.assert_close(actual[0], BasePolicy.encode(actor, domains)[0], atol=0, rtol=0)


def test_global_context_changes_relative_anchor_scores_and_receives_gradients():
    actor = model(global_context=True)
    encoded = actor.encode([[0, 1]] * 4)
    actions, distribution = actor.distribution(encoded, [2] * 4, [], [], False, 0.)
    target = actions.index(("open", actor.color_idx["a"]))
    (-distribution.log_prob(torch.tensor(target))).backward()
    assert actor.anchor_context[-1].weight.grad.abs().sum() > 0

    # Hold two candidate embeddings fixed and alter only a remote embedding.
    # The global path must change their relative scores, not just a shared bias.
    with torch.no_grad():
        for parameter in actor.anchor_context.parameters():
            parameter.zero_()
        actor.anchor_context[0].weight[0, 0] = 1
        actor.anchor_context[0].weight[0, 8] = 1
        actor.anchor_context[-1].weight[0, 0] = 1
    variables, cores = [torch.zeros_like(values) for values in encoded]
    a, b, d = (actor.color_idx[color] for color in "abd")
    variables[a, 0], variables[b, 0] = -1, 1
    changed = variables.clone()
    changed[d, 0] = 8
    gaps = []
    for values in (variables, changed):
        actions, distribution = actor.distribution((values, cores), [2] * 4, [], [], False, 0.)
        gaps.append(distribution.logits[actions.index(("open", b))]
                    - distribution.logits[actions.index(("open", a))])
    assert abs(float(gaps[1] - gaps[0])) > .5


def test_distances_follow_selected_active_paths_and_condition_core_scores():
    actor = model(distance_features=True)
    keys = [actor.key_idx[key] for key in ("ab", "bc", "cd")]
    anchor = actor.color_idx["a"]
    assert actor._distances(keys, anchor, [2] * 4) == {
        actor.color_idx[color]: distance for color, distance in zip("abcd", range(4))}
    sizes = [2] * 4
    sizes[actor.color_idx["b"]] = 1
    assert actor.color_idx["d"] not in actor._distances(keys, anchor, sizes)
    encoded = actor.encode([[0, 1]] * 4)
    actions, distribution = actor.distribution(encoded, [2] * 4, keys, [anchor], False, 0.)
    target = actions.index(("core", actor.key_idx["bd"]))
    (-distribution.log_prob(torch.tensor(target))).backward()
    assert actor.core_distance[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        for parameter in actor.core_distance.parameters():
            parameter.zero_()
        actor.core_distance[0].weight[0, 8] = 1  # Minimum anchor distance.
        actor.core_distance[-1].weight[0, 0] = 1
    actions, actual = actor.distribution(encoded, [2] * 4, keys, [anchor], False, 0.)
    _, reference = BasePolicy.distribution(actor, encoded, [2] * 4, keys, [anchor], False, 0.)
    other = actions.index(("core", actor.key_idx["ad"]))
    correction = (actual.logits[target] - actual.logits[other]
                  - reference.logits[target] + reference.logits[other])
    torch.testing.assert_close(correction, torch.tensor(.25), atol=1e-6, rtol=0)


def test_context_corrections_follow_variable_and_core_reordering():
    actor = model(global_context=True, distance_features=True, objective_features=True)
    actor.reward_colors = ["a"]
    raw = actor.network
    colors = list(reversed(raw["colors"]))
    remap = {old: colors.index(color) for old, color in enumerate(raw["colors"])}
    reordered = dict(colors=colors, dimensions=list(reversed(raw["dimensions"])), cores=[
        dict(core, colors=[remap[color] for color in core["colors"]]) for core in reversed(raw["cores"])])
    other = Policy(reordered, **actor.config, **actor.actor_config).eval()
    other.reward_colors = ["a"]
    with torch.no_grad():
        actor.anchor_context[-1].weight.fill_(.1)
        actor.core_distance[-1].weight.fill_(.1)
        actor.objective_role.weight.fill_(.1)
    other.load_state_dict(actor.state_dict())
    domains = {color: (0, 1) for color in raw["colors"]}
    for selected in ((), ("ab", "bc", "cd")):
        scores = []
        for policy in (actor, other):
            keys = [policy.key_idx[key] for key in selected]
            opened = [policy.color_idx["a"]] if selected else []
            actions, distribution = policy.distribution(policy.encode(domains), [2] * 4,
                                                        keys, opened, False, 0.)
            scores.append({(kind, policy.keys[idx] if kind == "core" else policy.network["colors"][idx]
                if kind == "open" else idx): probability for (kind, idx), probability
                in zip(actions, distribution.probs)})
        assert scores[0].keys() == scores[1].keys()
        for action in scores[0]:
            torch.testing.assert_close(scores[0][action], scores[1][action], atol=1e-6, rtol=0)


@pytest.mark.parametrize("learned_stopping", [False, True])
def test_growth_trace_likelihood_matches_actual_sample_masks(learned_stopping):
    torch.manual_seed(17)
    actor = model(global_context=True, distance_features=True, learned_stopping=learned_stopping)
    domains = [[0, 1]] * 4
    candidate, log_prob, _, trace = actor.rollout(domains, beta=0.)
    repeated, replay_log_prob, _, replay_trace = actor.rollout(domains, beta=0., trace=trace, policy_trace=True)
    assert candidate == repeated and replay_trace == trace
    torch.testing.assert_close(log_prob, replay_log_prob, atol=0, rtol=0)
    keys, opened, phase = [], [], False
    encoded = actor.encode(domains)
    expected = log_prob.new_zeros(())
    for action in trace:
        actions, distribution = actor.distribution(encoded, [2] * 4, keys, opened, phase, 0.)
        following = actor._continuations(actions, keys, opened, phase)
        permitted = [actions[idx] for idx in following]
        assert action in permitted
        masked = torch.distributions.Categorical(logits=distribution.logits[following])
        expected = expected + masked.log_prob(torch.tensor(permitted.index(action)))
        kind, idx = action
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif kind == "finish":
            phase = True
    torch.testing.assert_close(log_prob, expected, atol=0, rtol=0)


def test_learned_stopping_allows_short_core_trace_without_changing_anchor_selection():
    actor = model(learned_stopping=True)
    domains = [[0, 1]] * 4
    candidate = SummaryClusterCandidate("short", ("ab",), ("a", "b"))
    trace = actor.candidate_trace(candidate, domains, target_color="a")
    actual, logp, _, repeated = actor.rollout(domains, beta=0., trace=trace, policy_trace=True)
    assert actual.core_keys == candidate.core_keys and actual.open_colors == candidate.open_colors
    assert repeated == trace and torch.isfinite(logp)
    (-logp).backward()
    assert actor.head[-1].weight.grad.abs().sum() > 0
    actor.actor_config["learned_stopping"] = False
    with pytest.raises(ValueError, match="Illegal construction action"):
        actor.rollout(domains, beta=0., trace=trace, policy_trace=True)
    actor.actor_config["learned_stopping"] = True
    for keys, opened, phase in (([], [], False), ([], [0], False)):
        actions = actor.actions(keys, opened, phase, [2] * 4)
        assert actor._continuations(actions, keys, opened, phase) == BasePolicy._continuations(
            actor, actions, keys, opened, phase)


def test_learned_stopping_can_finish_with_fewer_outputs_than_the_limit():
    actor = model(learned_stopping=True)
    domains = [[0, 1]] * 4
    candidate = SummaryClusterCandidate("one_output", ("ab",), ("a",))
    trace = actor.candidate_trace(candidate, domains, target_color="a")
    actual, logp, _, _ = actor.rollout(domains, beta=0., trace=trace, policy_trace=True)
    assert actual.open_colors == ("a",) and torch.isfinite(logp)
    keys, opened = [actor.key_idx["ab"]], [actor.color_idx["a"]]
    actions = actor.actions(keys, opened, True, [2] * 4)
    assert ("open", actor.color_idx["b"]) in actions
    following = [actions[idx] for idx in actor._continuations(actions, keys, opened, True)]
    assert following == actions and ("finish", -1) in following
    actor.actor_config["learned_stopping"] = False
    following = [actions[idx] for idx in actor._continuations(actions, keys, opened, True)]
    assert following == [("open", actor.color_idx["b"])]


def test_learned_stopping_preserves_cycle_mask_and_full_core_limit():
    actor = model(learned_stopping=True, cyclic_only=True)
    for selected, can_finish in ((("ab", "bc"), False), (("ab", "bd", "ad"), True)):
        keys = [actor.key_idx[key] for key in selected]
        actions = actor.actions(keys, [actor.color_idx["a"]], False, [2] * 4)
        following = [actions[idx] for idx in actor._continuations(actions, keys, [0], False)]
        assert (("finish", -1) in following) == can_finish
        assert any(kind == "core" for kind, _ in following)

    # Parallel original tensors let all 64 legal additions remain affordable.
    network = ConstraintNetwork({f"link{idx:02d}": NumpyCore(values=np.ones((2, 2), dtype=bool),
        colors=["a", "b"]) for idx in range(64)}, maintain_indices=True)
    large = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
        learned_stopping=True, heuristic_candidates=0, exploration=0., max_cores=64,
        max_open_colors=1, decoder="sampling_grow").eval()
    candidate = SummaryClusterCandidate("large", tuple(large.keys), ("a",))
    domains = [[0, 1]] * 2
    trace = large.candidate_trace(candidate, domains)
    actual, logp, _, _ = large.rollout(domains, beta=0., trace=trace, policy_trace=True)
    assert actual.core_keys == candidate.core_keys and len(actual.core_keys) == 64
    assert torch.isfinite(logp)


@pytest.mark.parametrize("enabled", [False, True])
def test_strict_checkpoint_load_preserves_old_and_new_policy(tmp_path, enabled):
    actor = model(global_context=enabled, distance_features=enabled, objective_features=enabled,
                  learned_stopping=enabled)
    actor.beta, actor.reward_colors = 0., ["a"]
    if enabled:
        with torch.no_grad():
            actor.anchor_context[-1].weight.fill_(.1)
            actor.core_distance[-1].weight.fill_(.1)
            actor.objective_role.weight.fill_(.1)
    checkpoint = dict(version=1, kind=actor.kind, network=actor.network, config=actor.config,
        weights=actor.state_dict(), reward_colors=actor.reward_colors, beta=actor.beta)
    if enabled:
        checkpoint["actor_config"] = actor.actor_config
    path = tmp_path / "actor.pt"
    torch.save(checkpoint, path)
    restored = load(path)
    assert restored.actor_config == actor.actor_config and not restored.training
    assert restored.config == actor.config and restored.reward_colors == actor.reward_colors
    assert restored.beta == actor.beta
    domains = [[0, 1]] * 4
    assert list(restored.proposals(domains, max_candidates=4)) == list(actor.proposals(domains, max_candidates=4))
    if not enabled:
        checkpoint["actor_config"] = dict(global_context=False, distance_features=False)
        torch.save(checkpoint, path)
        assert load(path).actor_config["objective_features"] is False
        assert load(path).actor_config["learned_stopping"] is False
    if enabled:
        checkpoint["weights"].pop("anchor_context.0.weight")
        torch.save(checkpoint, path)
        with pytest.raises(RuntimeError, match="Missing key"):
            load(path)


@pytest.mark.parametrize("options", [dict(global_context=1), dict(distance_features="yes"),
    dict(objective_features=1), dict(learned_stopping=1),
    dict(global_context=True, contextual=False), dict(distance_features=True, target_first=False),
    dict(global_context=True, q_learning=True), dict(learned_stopping=True),
    dict(learned_stopping=True, decoder="sampling_grow", q_learning=True)])
def test_invalid_context_configuration_is_rejected(options):
    defaults = dict(contextual=True, target_first=True)
    defaults.update(options)
    with pytest.raises(ValueError):
        Policy(model().network, **defaults)
