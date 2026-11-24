"""Optional context corrections for the existing constructive policy.

The shared encoder, action masks and proposal decoder are unchanged. Domain
masks are built in one indexed write. Optional heads start at zero, preserving
a loaded policy before optimization.
"""
from collections import defaultdict, deque
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn
from torch.distributions import Categorical

from ..encoder import _mlp
from .model import Policy as BasePolicy


class Policy(BasePolicy):
    def __init__(self, network, *args, global_context=False, distance_features=False,
                 objective_features=False, **kwargs):
        if any(not isinstance(flag, bool) for flag in (global_context, distance_features, objective_features)):
            raise ValueError("Actor context options must be boolean")
        super().__init__(network, *args, **kwargs)
        self.actor_config = dict(global_context=global_context, distance_features=distance_features,
                                 objective_features=objective_features)
        if (global_context or distance_features) and (
                not self.contextual or not self.config.get("target_first") or self.config.get("q_learning")):
            raise ValueError("Actor context requires a contextual target-first policy without Q-learning")
        width = self.config["width"]
        if global_context:
            self.anchor_context = _mlp(3 * width + 3, width, 1)
        if distance_features:
            self.core_distance = _mlp(width + 5, width, 1)
        if objective_features:
            self.objective_role = nn.Linear(1, width, bias=False)
            nn.init.zeros_(self.objective_role.weight)
        for name in ("anchor_context", "core_distance"):
            if hasattr(self, name):
                nn.init.zeros_(getattr(self, name)[-1].weight)
                nn.init.zeros_(getattr(self, name)[-1].bias)

    def domain_features(self, domains):
        """Keep the original features without one tiny device write per variable."""
        if isinstance(domains, Mapping):
            domains = [domains[color] for color in self.network["colors"]]
        if len(domains) != len(self.dimensions):
            raise ValueError("Expected every variable's domain in network order")
        rows, columns = [], []
        for idx, values in enumerate(domains):
            values = list(values)
            if any(isinstance(value, bool) or not isinstance(value, Integral)
                   or not 0 <= value < self.network["dimensions"][idx] for value in values):
                raise ValueError("Domain value outside variable dimension")
            rows.extend([idx] * len(values))
            columns.extend(values)
        mask = self.dimensions.new_zeros(len(domains), self.max_dim)
        if columns:
            mask[torch.tensor(rows, dtype=torch.long, device=mask.device),
                 torch.tensor(columns, dtype=torch.long, device=mask.device)] = 1
        scalars = torch.stack((self.dimensions / self.max_dim, mask.sum(1) / self.dimensions,
                               (self.dimensions == 2).to(mask.dtype), self.degree.log1p()), dim=1)
        return torch.cat((mask, self.valid.to(mask.dtype), scalars), dim=1)

    def encode(self, domains):
        encoded = super().encode(domains)
        if not self.actor_config["objective_features"]:
            return encoded
        colors = self.network["colors"]
        targets = set(self.reward_colors or colors)
        auxiliary = min(1., len(targets) / max(1, len(colors) - len(targets)))
        weights = encoded[0].new_tensor([1. if color in targets else auxiliary for color in colors])
        # These are exactly the entropy weights used by learn.measure. Uniform
        # weights give log(1)=0 even after the adapter has learned nonzero weights.
        return (encoded[0] + self.objective_role(weights.log()[:, None]), *encoded[1:])

    def _distances(self, keys, anchor, sizes):
        """Shortest selected-core paths, ignoring singleton bridge variables."""
        incident = defaultdict(list)
        for key in keys:
            scope = [color for color in self.scopes[key] if sizes[color] > 1]
            for color in scope:
                incident[color].append(scope)
        distances, queue = {anchor: 0}, deque([anchor])
        while queue:
            color = queue.popleft()
            for scope in incident.pop(color, ()):
                for neighbor in scope:
                    if neighbor not in distances:
                        distances[neighbor] = distances[color] + 1
                        queue.append(neighbor)
        return distances

    def distribution(self, encoded, sizes, keys, opened, output_phase, beta, targets=()):
        actions, distribution = super().distribution(encoded, sizes, keys, opened, output_phase, beta, targets)
        variables, cores = encoded[:2]
        corrections = distribution.logits.new_zeros(len(actions))
        corrected = False
        initial = not keys and not opened and not output_phase
        if self.actor_config["global_context"] and initial:
            indices = [idx for idx, (kind, _) in enumerate(actions) if kind == "open"]
            if indices:
                selected = [actions[idx][1] for idx in indices]
                counts = variables.new_tensor(sizes)
                fractions = counts / self.dimensions
                scalars = torch.stack(((counts > 1).to(variables.dtype).mean(), fractions.mean(),
                    (counts.clamp_min(1).log() / self.dimensions.clamp_min(2).log()).mean()))
                context = torch.cat((variables.mean(0), cores.mean(0), scalars))
                inputs = torch.cat((variables[selected], context.expand(len(selected), -1)), dim=1)
                corrections = corrections.index_copy(0, torch.tensor(indices, device=variables.device),
                                                      self.anchor_context(inputs).squeeze(1))
                corrected = True
        if self.actor_config["distance_features"] and opened and not output_phase:
            indices = [idx for idx, (kind, _) in enumerate(actions) if kind == "core"]
            if indices:
                distances = self._distances(keys, opened[0], sizes)
                scale, features = max(1, self.config["max_cores"]), []
                selected = [actions[idx][1] for idx in indices]
                for key in selected:
                    active = [color for color in self.scopes[key] if sizes[color] > 1]
                    reached = [distances[color] for color in active if color in distances]
                    near = reached or [scale]
                    features.append([min(near) / scale, max(near) / scale,
                        sum(near) / (len(near) * scale), len(reached) / max(1, len(active)),
                        float(opened[0] in active)])
                inputs = torch.cat((cores[selected], cores.new_tensor(features)), dim=1)
                corrections = corrections.index_copy(0, torch.tensor(indices, device=cores.device),
                                                      self.core_distance(inputs).squeeze(1))
                corrected = True
        if not corrected:
            return actions, distribution
        return actions, Categorical(logits=distribution.logits + corrections)


def load(path):
    """Strictly load old construction weights or a checkpoint with actor_config."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint.get("kind") != Policy.kind:
        raise ValueError("Expected a version-1 construction-policy checkpoint")
    config = dict(checkpoint["config"])
    config.setdefault("normalize", False)
    model = Policy(checkpoint["network"], **config, **checkpoint.get("actor_config", {}))
    model.reward_colors, model.beta = checkpoint["reward_colors"], checkpoint["beta"]
    model.load_state_dict(checkpoint["weights"], strict=True)
    return model.eval()
