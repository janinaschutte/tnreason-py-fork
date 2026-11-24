"""Likelihood, gradient direction and clipping checks for construction updates."""
from copy import deepcopy
from math import log

import pytest

torch = pytest.importorskip("torch")
from experiments.cluster_selection.reinforcement.updates import advantages, update


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.))
        self.calls = []

    def rollout(self, domains, *, beta, trace, policy_trace):
        assert beta == 0 and policy_trace
        self.calls.append((domains, trace))
        distribution = torch.distributions.Categorical(logits=torch.stack((self.theta, self.theta * 0)))
        logp = torch.stack([distribution.log_prob(torch.tensor(action[1])) for action in trace]).sum()
        return None, logp, distribution.entropy(), trace


def group(actor, rewards=(1., 0.), *, length=1):
    result = dict(domains={"x": (0, 1)}, rollouts=[])
    for action, reward, advantage in zip((0, 1), rewards, advantages(rewards, mode="none")):
        trace = [("choice", action)] * length
        with torch.no_grad():
            old = float(actor.rollout(result["domains"], beta=0., trace=trace, policy_trace=True)[1])
        result["rollouts"].append(dict(trace=trace, reward=reward, advantage=advantage, old_logp=old))
    actor.calls.clear()
    return result


@pytest.mark.parametrize("mode,expected", [("none", [-1., 1.]), ("rms", [-4., 4.]),
                                           ("group", [-2., 2.])])
def test_advantages_are_leave_one_out_with_explicit_normalization(mode, expected):
    assert advantages([0., 1.], mode=mode, scale=.25) == expected
    assert advantages([.5, .5], mode=mode) == [0., 0.]
    assert advantages([.1] * 6, mode=mode) == [0.] * 6
    assert advantages([0., 0., 3.], mode="none") == [-1.5, -1.5, 3.]


@pytest.mark.parametrize("rewards,options", [([], {}), ([1.], {}), ([0., float("nan")], {}),
    ([0., 1.], dict(mode="unknown")), ([0., 1.], dict(scale=0.))])
def test_invalid_advantage_inputs(rewards, options):
    with pytest.raises(ValueError):
        advantages(rewards, **options)


def test_positive_construction_increases_in_probability_and_sequence_scores_are_summed():
    values = []
    for length in (1, 2):
        actor = Actor()
        measured = group(actor, length=length)
        result = update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1), entropy_weight=0.)
        assert result["optimizer_steps"] == 1 and result["scored_trials"] == 2
        assert result["approx_kl"] == pytest.approx(0., abs=1e-14)
        assert actor.theta > 0 and len(actor.calls) == 2
        assert result["grad_norm"] == pytest.approx(length / 2)
        values.append(float(actor.theta))
    assert values == pytest.approx([.05, .1])


def test_zero_advantage_groups_keep_batch_weight_without_entropy_only_updates():
    actor = Actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.1)
    measured = group(actor, rewards=(0., 0.))
    original = actor.theta.detach().clone()
    result = update(actor, [measured], optimizer, entropy_weight=1.)
    assert result["optimizer_steps"] == result["scored_trials"] == 0
    assert result["skipped_groups"] == 1 and not optimizer.state
    torch.testing.assert_close(actor.theta, original)
    result = update(actor, [measured, group(actor)], torch.optim.SGD(actor.parameters(), lr=.1),
                    entropy_weight=0.)
    assert result["active_groups"] == 1 and result["groups"] == 2
    assert float(actor.theta) == pytest.approx(.025)


@pytest.mark.parametrize("advantage,ratio,clipped", [(1., 1.5, True), (-1., .5, True),
                                                    (1., .5, False), (-1., 1.5, False)])
def test_clipped_surrogate_handles_both_advantage_signs(advantage, ratio, clipped):
    actor = Actor()
    measured = group(actor)
    item = measured["rollouts"][0]
    item.update(advantage=advantage, old_logp=item["old_logp"] - log(ratio))
    measured["rollouts"] = [item]
    result = update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1),
                    clip_ratio=.2, entropy_weight=0., target_kl=None)
    expected = -min(ratio * advantage, max(.8, min(1.2, ratio)) * advantage)
    assert result["policy_loss"] == pytest.approx(expected)
    assert result["clip_fraction"] == 1.
    assert (abs(float(actor.theta)) < 1e-12) == clipped


def test_multiple_epochs_require_importance_clipping_and_kl_stops_before_another_step():
    actor = Actor()
    measured = group(actor)
    optimizer = torch.optim.SGD(actor.parameters(), lr=.1)
    with pytest.raises(ValueError, match="multiple epochs"):
        update(actor, [measured], optimizer, epochs=2)
    result = update(actor, [measured], optimizer, epochs=4, clip_ratio=.2, entropy_weight=0.,
                    target_kl=1e-6)
    assert result["early_stop"] and result["optimizer_steps"] == 1
    assert result["scored_trials"] == 4 and result["approx_kl"] > 1e-6
    assert float(actor.theta) == pytest.approx(.05)
    assert actor.theta.grad is None


def test_nonfinite_measurements_or_scores_do_not_update_parameters():
    actor = Actor()
    measured = group(actor)
    original = actor.theta.detach().clone()
    measured["rollouts"][0]["reward"] = float("nan")
    with pytest.raises(ValueError, match="finite rollout"):
        update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1))
    torch.testing.assert_close(actor.theta, original)
    measured = group(actor)
    measured["rollouts"][0]["old_logp"] = -10000.
    with pytest.raises(FloatingPointError, match="importance ratio"):
        update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1), clip_ratio=.2)
    torch.testing.assert_close(actor.theta, original)
    assert actor.theta.grad is None


def test_gradient_norm_reports_clipping_before_the_step():
    actor = Actor()
    measured = group(actor)
    result = update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1),
                    entropy_weight=0., grad_clip=.1)
    assert result["grad_norm"] == pytest.approx(.5)
    assert result["grad_clipped"] == 1
    assert float(actor.theta) == pytest.approx(.01, abs=3e-8)


def test_nonfinite_gradient_is_cleared_before_any_optimizer_step():
    class SingularActor(Actor):
        def rollout(self, *args, **kwargs):
            return None, self.theta.sqrt(), self.theta * 0, kwargs["trace"]

    actor = SingularActor()
    measured = dict(domains={"x": (0, 1)}, rollouts=[dict(
        trace=[("choice", 0)], old_logp=0., reward=1., advantage=1.)])
    with pytest.raises(RuntimeError, match="non-finite"):
        update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1), entropy_weight=0.)
    assert float(actor.theta) == 0. and actor.theta.grad is None


def test_update_replays_growth_masks_and_the_exact_original_construction_likelihood():
    import numpy as np
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection.data import network_data
    from experiments.cluster_selection.reinforcement.model import Policy

    network = ConstraintNetwork({key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
                                 for key in ("ab", "bc", "ac")}, maintain_indices=True)
    torch.manual_seed(5)
    actor = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
                   max_cores=3, max_open_colors=2, heuristic_candidates=0, decoder="sampling_grow")
    domains = {color: (0, 1, 2) for color in "abc"}
    measured = dict(domains=domains, rollouts=[])
    for reward in (1., 0.):
        with torch.no_grad():
            _, logp, _, trace = actor.rollout(domains, beta=0.)
        measured["rollouts"].append(dict(trace=trace, old_logp=float(logp), reward=reward,
                                          advantage=1. if reward else -1.))
    rng = torch.get_rng_state().clone()
    result = update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.001))
    assert result["approx_kl"] == pytest.approx(0., abs=1e-12)
    assert torch.equal(rng, torch.get_rng_state())
    assert result["optimizer_steps"] == 1 and result["grad_norm"] > 0


def retained_group(actor):
    measured = group(actor)
    for item in measured["rollouts"]:
        _, logp, entropy, _ = actor.rollout(measured["domains"], beta=0.,
                                            trace=item["trace"], policy_trace=True)
        item.update(logp_tensor=logp, entropy_tensor=entropy)
    actor.calls.clear()
    return measured


def test_retained_graphs_avoid_forward_recomputation():
    actor = Actor()
    measured = retained_group(actor)
    result = update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1), entropy_weight=0.)
    assert not actor.calls and float(actor.theta) == pytest.approx(.05)
    assert result["retained_trials"] == result["scored_trials"] == 2
    assert result["rescored_trials"] == 0


@pytest.mark.parametrize("change", ["missing", "nonscalar", "detached", "nonfinite", "old_logp"])
def test_retained_graphs_reject_malformed_tensors_without_updating(change):
    actor = Actor()
    measured = retained_group(actor)
    item = measured["rollouts"][0]
    if change == "missing":
        del item["entropy_tensor"]
    elif change == "nonscalar":
        item["logp_tensor"] = item["logp_tensor"].unsqueeze(0)
    elif change == "detached":
        item["logp_tensor"] = item["logp_tensor"].detach()
    elif change == "nonfinite":
        item["entropy_tensor"] = item["entropy_tensor"] * float("nan")
    else:
        item["old_logp"] += .1
    with pytest.raises(ValueError, match="[Rr]etained|[Tt]ensors|graphs"):
        update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1))
    assert float(actor.theta) == 0. and actor.theta.grad is None and not actor.calls


@pytest.mark.parametrize("options", [dict(epochs=2, clip_ratio=.2), dict(clip_ratio=.2)])
def test_retained_graphs_require_a_single_vanilla_epoch(options):
    actor = Actor()
    measured = retained_group(actor)
    with pytest.raises(ValueError, match="one vanilla epoch"):
        update(actor, [measured], torch.optim.SGD(actor.parameters(), lr=.1), **options)
    assert float(actor.theta) == 0. and actor.theta.grad is None


def test_retained_and_rescored_graphs_cannot_be_mixed_and_zero_advantage_graphs_skip():
    actor = Actor()
    measured = retained_group(actor)
    with pytest.raises(ValueError, match="mix retained/rescored"):
        update(actor, [measured, group(actor)], torch.optim.SGD(actor.parameters(), lr=.1))
    actor.calls.clear()
    for item in measured["rollouts"]:
        item["reward"] = item["advantage"] = 0.
    result = update(actor, [measured], torch.optim.Adam(actor.parameters(), lr=.1))
    assert result["optimizer_steps"] == result["retained_trials"] == 0 and not actor.calls
    assert float(actor.theta) == 0. and actor.theta.grad is None


def test_retained_and_recomputed_real_policy_updates_are_bitwise_identical(monkeypatch):
    import numpy as np
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection.data import network_data
    from experiments.cluster_selection.reinforcement.model import Policy

    network = ConstraintNetwork({key: NumpyCore(values=~np.eye(3, dtype=bool), colors=list(key))
                                 for key in ("ab", "bc", "ac")}, maintain_indices=True)
    torch.manual_seed(5)
    retained = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
                      max_cores=3, max_open_colors=2, heuristic_candidates=0, decoder="sampling_grow",
                      exploration=0.)
    recomputed = deepcopy(retained)
    domains = {color: (0, 1, 2) for color in "abc"}
    cached, rescored = dict(domains=domains, rollouts=[]), dict(domains=domains, rollouts=[])
    for reward, advantage in zip((1., 0., 1., 0.), advantages([1., 0., 1., 0.], "none")):
        _, logp, entropy, trace = retained.rollout(domains, beta=0.)
        item = dict(trace=trace, old_logp=float(logp.detach()), reward=reward, advantage=advantage)
        rescored["rollouts"].append(dict(item))
        cached["rollouts"].append(dict(item, logp_tensor=logp, entropy_tensor=entropy))
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(retained, "rollout", lambda *args, **kwargs: pytest.fail("Cached update re-scored a trace"))
    left_optimizer = torch.optim.Adam(retained.parameters(), lr=.001)
    right_optimizer = torch.optim.Adam(recomputed.parameters(), lr=.001)
    left = update(retained, [cached], left_optimizer)
    right = update(recomputed, [rescored], right_optimizer)
    assert torch.equal(rng, torch.get_rng_state())
    assert left["retained_trials"] == right["rescored_trials"] == 4
    assert left["grad_norm"] > 0 and left["grad_norm"] == right["grad_norm"]
    assert left["policy_loss"] == right["policy_loss"] and left["entropy"] == right["entropy"]
    for left_parameter, right_parameter in zip(retained.parameters(), recomputed.parameters()):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=0, atol=0)
        torch.testing.assert_close(left_parameter.grad, right_parameter.grad, rtol=0, atol=0)
    for left_state, right_state in zip(left_optimizer.state.values(), right_optimizer.state.values()):
        for name in left_state:
            torch.testing.assert_close(left_state[name], right_state[name], rtol=0, atol=0)
