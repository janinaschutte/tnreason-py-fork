"""Temporary objective-role prototype; not a production checkpoint format.

Run without arguments for lightweight representation/gradient checks. An
optional --trials 512 runs a deliberate fit check on the symmetric witness.
Neither experiment uses Sudoku or establishes generalization.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.reinforcement.actor import Policy as BasePolicy
from experiments.cluster_selection.reinforcement import refine
from experiments.cluster_selection.reinforcement.refine import collect
from experiments.cluster_selection.reinforcement.updates import advantages, update


class Policy(BasePolicy):
    """Twenty-line candidate addition: expose the generic reward's own weights."""
    def __init__(self, *args, objective_features=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prototype_config = dict(objective_features=objective_features)
        if objective_features:
            self.objective_role = nn.Linear(1, self.config["width"], bias=False)
            nn.init.zeros_(self.objective_role.weight)

    def encode(self, domains):
        encoded = super().encode(domains)
        if not self.prototype_config["objective_features"]:
            return encoded
        colors = self.network["colors"]
        targets = set(self.reward_colors or colors)
        auxiliary = min(1., len(targets) / max(1, len(colors) - len(targets)))
        weights = encoded[0].new_tensor([1. if color in targets else auxiliary for color in colors])
        return (encoded[0] + self.objective_role(weights.log()[:, None]), *encoded[1:])


def problem():
    cores = {}
    for a, b, c in ("abc", "uvw"):
        for key, values in ((a + b, np.eye(2, dtype=bool)), (b + c, np.eye(2, dtype=bool)),
                            (a + c, np.array([[1, 1], [1, 0]], dtype=bool))):
            cores[key] = NumpyCore(values=values, colors=list(key))
    network = ConstraintNetwork(cores, maintain_indices=True)
    state = network.create_mac_state()
    assert state.propagate() and not state.is_solved()
    torch.manual_seed(29)
    model = Policy(network_data(network), width=8, rounds=1, contextual=True, target_first=True,
        exploration=0., heuristic_candidates=0, decoder="sampling_grow", min_active=2,
        compact_domains=True, cyclic_only=True, max_cores=3, max_open_colors=1)
    model.reward_colors, model.beta = ["a"], 0.
    sample = dict(domains=list(state.domains.values()), known_satisfiable=True)
    return model, state, sample


def anchor_scores(model, domains, *, base_encoding=False):
    encoded = BasePolicy.encode(model, domains) if base_encoding else model.encode(domains)
    actions, distribution = model.distribution(encoded, [len(domains[c]) for c in model.network["colors"]],
                                              [], [], False, 0.)
    return actions, distribution


def checks(model, state):
    actions, before = anchor_scores(model, state.domains, base_encoding=True)
    _, initial = anchor_scores(model, state.domains)
    assert torch.equal(before.logits, initial.logits)
    a, u = (actions.index(("open", model.color_idx[color])) for color in ("a", "u"))
    assert torch.equal(initial.logits[a], initial.logits[u])
    (-initial.log_prob(torch.tensor(a))).backward()
    gradient = model.objective_role.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    result = dict(zero_initialization_preserves_logits_bitwise=True,
                  initial_a_u_logit_gap=float(initial.logits[a] - initial.logits[u]),
                  role_gradient_l1=float(gradient.abs().sum()))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.objective_role.weight.zero_()
        model.objective_role.weight[0, 0] = 1.
        for parameter in model.head.parameters():
            parameter.zero_()
        model.head[0].weight[0, 3 * model.config["width"]] = 1.
        model.head[0].bias[0] = 10.
        model.head[-1].weight[0, 0] = 1.
    _, changed = anchor_scores(model, state.domains)
    result["constructed_a_u_logit_gap"] = float(changed.logits[a] - changed.logits[u])
    assert result["constructed_a_u_logit_gap"] > 1.
    model.reward_colors = model.network["colors"]
    _, all_primary = anchor_scores(model, state.domains)
    _, all_primary_base = anchor_scores(model, state.domains, base_encoding=True)
    assert torch.equal(all_primary.logits, all_primary_base.logits)
    result["all_variables_rewarded_means_exact_zero_feature"] = True
    model.load_state_dict(original)
    model.reward_colors = ["a"]
    model.zero_grad(set_to_none=True)
    return result


def capacity(initial, state, sample, trials):
    results = {}
    for enabled in (False, True):
        # Use the integrated production actor, not the temporary class above.
        model = BasePolicy(initial.network, **initial.config, objective_features=enabled)
        weights = {key: value for key, value in initial.state_dict().items()
                   if enabled or not key.startswith("objective_role.")}
        model.load_state_dict(weights)
        model.reward_colors, model.beta = ["a"], 0.
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)

        def probe():
            torch.manual_seed(1234)
            group, outcomes, _ = collect(model, state, sample, rollouts=128)
            actions, distribution = anchor_scores(model, state.domains)
            left = [idx for idx, (kind, color) in enumerate(actions)
                    if kind == "open" and model.network["colors"][color] in "abc"]
            anchor_indices = [idx for idx, (kind, _) in enumerate(actions) if kind == "open"]
            return dict(mean_reward=sum(row["reward"] for row in group["rollouts"]) / 128,
                left_component_samples=sum(outcome["domains"]["a"] == (0,) for outcome in outcomes),
                left_component_probability=float(distribution.probs[left].sum()
                    / distribution.probs[anchor_indices].sum()),
                reward_histogram={str(value): sum(row["reward"] == value for row in group["rollouts"])
                                  for value in (.3, .7)})

        before, updates = probe(), []
        for batch in range(trials // 8):
            torch.manual_seed(29 + batch)  # Common random numbers before each arm's sampled batch.
            group, _, _ = collect(model, state, sample, rollouts=8)
            rewards = [item["reward"] for item in group["rollouts"]]
            for item, advantage in zip(group["rollouts"], advantages(rewards)):
                item["advantage"] = advantage
            updates.append(update(model, [group], optimizer, entropy_weight=0.))
        results["objective" if enabled else "base"] = dict(before=before, after=probe(),
            fresh_trials=trials, optimizer_steps=sum(row["optimizer_steps"] for row in updates))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 0 or args.trials % 8:
        parser.error("Trials must be nonnegative and divisible by eight")
    torch.set_num_threads(1)
    output = Path(tempfile.mkdtemp(prefix="objective_feature_", dir=Path(__file__).parent))
    model, state, sample = problem()
    sources = refine.source_hashes()
    report = dict(temporary_harness=True, capacity_uses_production_actor=bool(args.trials),
        dependency_source_sha256=sources, training_or_test_data_used=False,
        prototype_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks(model, state),
        note="Generic objective observability/capacity test, not Sudoku or generalization evidence")
    if args.trials:
        report["capacity"] = capacity(model, state, sample, args.trials)
    report["dependency_sources_unchanged"] = sources == refine.source_hashes()
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "dependency_source_sha256"}, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
