"""Two actual Sudoku traces; synthetic advantages test gradient paths, not reward."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import perf_counter

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))
os.sched_setaffinity(0, {0})

import torch
from context_pilot import digest, write
from experiments.cluster_selection.reinforcement import refine
from experiments.cluster_selection.reinforcement.updates import update


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = json.loads((BASE / "pilot_data.json").read_text())
    cached = json.loads((BASE / "pilot_states.json").read_text())
    sample = next(row for row in cached["states"] if row["puzzle"] == 24)
    domains = {color: tuple(values) for color, values in zip(data["network"]["colors"], sample["domains"])}
    parent = BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    paths = dict(data=BASE / "pilot_data.json", cache=BASE / "pilot_states.json", parent=parent,
                 driver=Path(__file__), helper=BASE / "context_pilot.py")
    hashes, sources = {name: digest(path) for name, path in paths.items()}, refine.source_hashes()
    model = refine.initialize(data, seed=0, initial=parent, global_context=True)
    rescored = deepcopy(model)
    cached_group, plain_group, scores = [], [], []
    started = perf_counter()
    for advantage in (-1., 1.):
        _, logp, entropy, trace = model.rollout(domains, beta=0.)
        item = dict(trace=trace, old_logp=float(logp.detach()), reward=(advantage + 1) / 2,
                    advantage=advantage)
        cached_group.append(dict(**item, logp_tensor=logp, entropy_tensor=entropy))
        plain_group.append(item)
        with torch.no_grad():
            _, other, _, _ = rescored.rollout(domains, beta=0., trace=trace, policy_trace=True)
        scores.append(dict(decisions=len(trace), sampled_logp=float(logp.detach()),
                           replay_logp=float(other), absolute_difference=abs(float(logp.detach()) - float(other))))
    rng = torch.get_rng_state().clone()
    optimizers = [torch.optim.Adam(actor.parameters(), lr=1e-4) for actor in (model, rescored)]
    cache_metrics = update(model, [dict(domains=domains, rollouts=cached_group)], optimizers[0])
    replay_metrics = update(rescored, [dict(domains=domains, rollouts=plain_group)], optimizers[1])
    assert torch.equal(rng, torch.get_rng_state())
    gradient_equal, parameter_equal, gradient_error, parameter_error = True, True, 0., 0.
    error_squared, baseline_squared, parameter_differences = 0., 0., []
    for (name, left), (right_name, right) in zip(model.named_parameters(), rescored.named_parameters()):
        assert name == right_name and (left.grad is None) == (right.grad is None)
        parameter_equal &= torch.equal(left, right)
        local_error = float((left - right).abs().max().detach())
        parameter_error = max(parameter_error, local_error)
        if local_error:
            parameter_differences.append(dict(name=name, max_absolute_difference=local_error))
        if left.grad is not None:
            difference = left.grad - right.grad
            gradient_equal &= torch.equal(left.grad, right.grad)
            gradient_error = max(gradient_error, float(difference.abs().max()))
            error_squared += float(difference.double().square().sum())
            baseline_squared += float(right.grad.double().square().sum())
    with torch.no_grad():
        probabilities = []
        sizes = [len(domains[color]) for color in model.network["colors"]]
        for actor in (model, rescored):
            actions, distribution = actor.distribution(actor.encode(domains), sizes, [], [], False, 0.)
            allowed = actor._continuations(actions, [], [], False)
            probabilities.append(distribution.logits[allowed].softmax(0))
        anchor_tv = float((probabilities[0] - probabilities[1]).abs().sum()) / 2
    assert sources == refine.source_hashes() and hashes == {name: digest(path) for name, path in paths.items()}
    result = dict(source_sha256=sources, hashes=hashes, cpu=sorted(os.sched_getaffinity(0)),
        scope="One existing training root, two policy-sampled traces; no oracle contraction and no heldout100 data",
        synthetic_advantages=[-1., 1.], synthetic_rewards=[0., 1.],
        interpretation="Tests numerical gradient equivalence only; rewards are deliberately synthetic, not pruning evidence",
        score_checks=scores, cached=cache_metrics, rescored=replay_metrics,
        gradients_bitwise_equal=gradient_equal, parameters_bitwise_equal=parameter_equal,
        max_gradient_abs_difference=gradient_error, gradient_relative_l2=(error_squared / max(baseline_squared, 1e-300)) ** .5,
        max_parameter_abs_difference=parameter_error, rng_unchanged=True,
        largest_parameter_differences=sorted(parameter_differences, key=lambda item: -item["max_absolute_difference"])[:8],
        actual_anchor_total_variation=anchor_tv,
        seconds=perf_counter() - started, source_and_inputs_unchanged=True)
    output = Path(mkdtemp(prefix="sudoku_gradient_", dir=BASE))
    write(output / "summary.json", result)
    print(output, json.dumps({key: value for key, value in result.items() if key not in {"source_sha256", "hashes"}}, indent=2))


if __name__ == "__main__":
    main()
