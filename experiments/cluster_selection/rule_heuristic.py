"""Optional overlap and small Hall clusters recognized from tensor supports.

No variable names, global value labels or puzzle rules are assumed. Recognition
only proposes original cores; domain-conditioned contraction and GAC perform all
pruning. Recreate the generator after changing core values as well as structure.
"""
from collections import Counter, defaultdict
from hashlib import sha1
from itertools import combinations, product, zip_longest
from math import prod
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from . import runtime as _runtime
from .candidate_generation import ClusterCandidateGenerator as _Base
from .candidate_evaluation_tnreason import SummaryClusterCandidate
from .contraction_cost import estimate_candidate_cost
from .runtime import attach_heuristic as _attach_heuristic


class ClusterCandidateGenerator(_Base):
    """Keep the legacy generator unchanged unless rule_inspired is enabled.

    Enabled calls split traversal steps between legacy search and motif search,
    interleaving their proposals with legacy first. A bounded, cached support
    catalog recognizes (indicator == bit) iff (category == value), including
    reversed axes, literal polarity and parallel premises. Targets may have
    additional values; sources must cover every currently allowed choice.
    """

    def __init__(self, network, *, rule_inspired=False, **options):
        if not isinstance(rule_inspired, bool):
            raise ValueError("rule_inspired must be Boolean")
        super().__init__(network, **options)
        self.rule_inspired, self._bridges = rule_inspired, None
        self.rule_statistics = Counter()

    def _catalog(self):
        groups = defaultdict(list)
        for key, core in self.cores.items():
            if len(core.colors) == 2 and 2 in core.shape:
                groups[tuple(sorted(core.colors))].append(key)
        channels, remaining = defaultdict(list), self.max_work
        for colors, keys in sorted(groups.items()):
            shape = tuple(self.dimensions[c] for c in colors)
            size = prod(shape)
            if size > self.max_size:
                self.rule_statistics["catalog_skipped_inputs"] += 1
                continue  # Do not inspect unaffordable input tables.
            # Sparse polynomial lookup scans stored slices; custom lookup costs
            # cannot be inferred, so otherwise charge realized entries read.
            reads = sum(size * max(1, len(values)*2) if isinstance(values := getattr(self.cores[key], "values", None), list)
                        else size for key in keys)
            if reads > remaining:
                self.rule_statistics["catalog_skipped_inputs"] += 1
                continue
            remaining -= reads
            self.rule_statistics["catalog_support_entries"] += size * len(keys)
            self.rule_statistics["catalog_lookup_work"] += reads
            support = np.ones(shape, dtype=bool)
            valid = True
            for key in keys:
                core = self.cores[key]
                values = getattr(core, "values", None)
                if isinstance(values, np.ndarray):
                    if tuple(values.shape) != tuple(core.shape):
                        valid = False
                        break
                    table = values.transpose(tuple(core.colors.index(c) for c in colors)) != 0
                else:
                    table = np.asarray([core[dict(zip(colors, at))] != 0
                                        for at in np.ndindex(shape)], dtype=bool).reshape(shape)
                support &= table
            if not valid:
                continue
            for axis, indicator in enumerate(colors):
                if self.dimensions[indicator] != 2:
                    continue
                category = colors[1-axis]
                table = support if axis == 1 else support.T  # category x indicator
                if not np.all(table.sum(axis=1) == 1):
                    continue
                for bit in (0, 1):  # Both interpretations are valid for dimension two.
                    selected = np.flatnonzero(table[:, bit])
                    if len(selected) == 1:
                        channels[indicator, bit].append((category, int(selected[0]), tuple(sorted(keys))))
        bridges = defaultdict(lambda: defaultdict(list))
        pair_budget = min(self.max_steps, self.max_work)
        possible_pairs = sum(len(row)*(len(row)-1)//2 for row in channels.values())
        self.rule_statistics["catalog_pairs_capped"] += int(possible_pairs > pair_budget)
        for literal, occurrences in channels.items():
            for left, right in combinations(occurrences, 2):
                if not pair_budget:
                    break
                pair_budget -= 1
                self.rule_statistics["catalog_pairs"] += 1
                if left[0] == right[0]:
                    continue
                keys = tuple(sorted(set(left[2]) | set(right[2])))
                bridges[left[0]][left[1]].append((right[0], right[1], literal, keys))
                bridges[right[0]][right[1]].append((left[0], left[1], literal, keys))
            if not pair_budget:
                break
        self._bridges = {c: {v: tuple(sorted(set(links))) for v, links in slots.items()}
                         for c, slots in bridges.items()}

    def _affordable(self, keys, output, domains, compact):
        keys, opened = tuple(sorted(keys)), (output,)
        if len(keys) > self.max_cores or self.dimensions[output] > self.max_size:
            return None
        candidate = SummaryClusterCandidate("rule_" + sha1(repr((keys, opened)).encode()).hexdigest(), keys, opened)
        inputs = ({key: SimpleNamespace(colors=self.cores[key].colors,
                   shape=tuple(len(domains[c]) for c in self.cores[key].colors)) for key in keys}
                  if compact else self.cores)
        cost = estimate_candidate_cost(inputs, candidate)
        if cost.work <= self.max_work and max(cost.max_size, self.dimensions[output]) <= self.max_size:
            return candidate
        return None

    def _rules(self, domains, eligible, changed, compact, budget):
        if self._bridges is None:
            self._catalog()
        remaining, emitted = budget, set()

        def spend():
            nonlocal remaining
            if remaining <= 0:
                self.rule_statistics["search_capped"] += 1
                return False
            remaining -= 1
            self.rule_statistics["search_steps"] += 1
            return True

        def proposals(keys, images, family):
            if not set().union(*(self.scopes[key] for key in keys)) & changed:
                return
            for target in sorted(images.keys() & eligible):
                if not spend():
                    return
                if not set(domains[target]) - images[target]:
                    continue
                signature = tuple(sorted(keys)), (target,)
                if signature in emitted:
                    continue
                emitted.add(signature)
                candidate = self._affordable(keys, target, domains, compact)
                if candidate is not None:
                    yield SummaryClusterCandidate(candidate.key.replace("rule_", "rule_"+family+"_", 1),
                                                  candidate.core_keys, candidate.open_colors)
                else:
                    self.rule_statistics["cost_rejected"] += 1

        rows = []
        # Reserve most motif traversal for joining rows rather than enumerating
        # many alternative encodings of the same source choice set.
        row_budget = max(1, budget // 3)
        source_options = []
        for category, slots in sorted(self._bridges.items(), key=lambda item: (len(domains[item[0]]), item[0])):
            values = domains[category]
            if len(values) < 2:
                continue
            options = [tuple(link for link in slots.get(value, ())
                             if link[1] in domains[link[0]] and link[2][1] in domains[link[2][0]])
                       for value in values]
            if any(not option for option in options):
                continue
            common = set.intersection(*(set(link[0] for link in option) for option in options))
            for target in sorted(common):
                if not spend():
                    return
                selected = [next(link for link in option if link[0] == target) for option in options]
                keys = set().union(*(set(link[3]) for link in selected))
                yield from proposals(keys, {target: {link[1] for link in selected}}, "overlap")
            if len(values) <= 4:
                source_options.append((category, product(*options)))
        # Round robin across source categories before exploring extra encodings.
        for selections in zip_longest(*(choices for _, choices in source_options)):
            for (category, _), selected in zip(source_options, selections):
                if selected is None:
                    continue
                if row_budget <= 0 or not spend():
                    break
                row_budget -= 1
                targets = {link[0]: link[1] for link in selected}
                if len(targets) == len(selected):
                    keys = frozenset().union(*(frozenset(link[3]) for link in selected))
                    rows.append((category, targets, keys, frozenset(link[2] for link in selected)))
            if row_budget <= 0 or remaining <= 0:
                break
        by_target = defaultdict(set)
        for idx, (_, targets, _, _) in enumerate(rows):
            for target in targets:
                by_target[target].add(idx)
        for start, (category, targets, keys, literals) in enumerate(rows):
            stack = [((start,), {category}, {t: {v} for t, v in targets.items()}, keys, literals)]
            while stack:
                chosen, sources, images, keys, literals = stack.pop()
                if 2 <= len(chosen) == len(images) <= 4:
                    yield from proposals(keys, images, "hall")
                if len(chosen) == 4:
                    continue
                neighbors = set().union(*(by_target[t] for t in images))
                for idx in sorted(neighbors, reverse=True):
                    if idx <= chosen[-1]:
                        continue
                    if not spend():
                        return
                    category, targets, extra, extra_literals = rows[idx]
                    if (category in sources or category in images or sources & targets.keys()
                            or literals & extra_literals
                            or len(images.keys() | targets.keys()) > 4
                            or any(value in images.get(target, ()) for target, value in targets.items())):
                        continue
                    combined = {t: set(values) for t, values in images.items()}
                    for target, value in targets.items():
                        combined.setdefault(target, set()).add(value)
                    stack.append((chosen+(idx,), sources | {category}, combined, keys | extra, literals | extra_literals))

    def generate(self, domains=None, **options):
        if not self.rule_inspired:
            return super().generate(domains, **options)
        steps = self.max_steps
        legacy_steps = max(1, steps // 2)
        try:
            self.max_steps = legacy_steps
            legacy = super().generate(domains, **options)
        finally:
            self.max_steps = steps
        limit = options.get("max_candidates")
        limit = self.max_candidates if limit is None else limit
        if not limit or steps <= legacy_steps:
            return legacy
        current = {c: tuple(sorted(set((domains or {}).get(c, range(dim))))) for c, dim in self.dimensions.items()}
        if any(not values for values in current.values()):
            return legacy
        requested = options.get("open_colors")
        eligible = (set(requested) if requested is not None else
                    {c for c, d in self.dimensions.items() if d > 2} or set(self.dimensions))
        eligible &= {c for c, values in current.items() if len(values) > 1}
        changed = set(self.dimensions if options.get("changed_colors") is None else options["changed_colors"])
        rules = self._rules(current, eligible, changed, options.get("compact_domains", False), steps-legacy_steps)
        result, seen = [], set()
        for pair in zip_longest(legacy, rules):
            for candidate in pair:
                if candidate is None:
                    continue
                signature = candidate.core_keys, candidate.open_colors
                if signature not in seen:
                    seen.add(signature)
                    result.append(candidate)
                    if candidate.key.startswith("rule_"):
                        self.rule_statistics["emitted_" + candidate.key.split("_")[1]] += 1
                if len(result) >= limit:
                    return tuple(result)
        return tuple(result)


class HeuristicPolicy(_runtime.HeuristicPolicy):
    """Refresh proposals after pruning; cache only exact no-op inputs.

    Explicit construction with rule_inspired=False provides the same refreshed
    schedule for legacy-proposal controls. The default attach helper stays legacy.
    """

    def __init__(self, state, *, rule_inspired=False, **options):
        super().__init__(state, **options)
        previous = self.generator
        limits = {key: getattr(previous, key) for key in
                  ("max_cores", "max_open_colors", "max_cycle_length", "max_candidates", "max_work", "max_size", "max_steps")}
        self.generator = ClusterCandidateGenerator(state._cores, rule_inspired=rule_inspired, **limits)

    def __call__(self, state, changed):
        if state is not self.state:
            raise ValueError("A heuristic policy is bound to one state")
        snapshot = tuple(state.domains[c] for c in self.colors)
        if state.contradiction or state.is_solved() or snapshot == self.last:
            return
        started, attempts, proposals = perf_counter(), 0, None
        self.statistics["calls"] += 1
        try:
            while not state.contradiction and not state.is_solved() and attempts < self.generator.max_candidates:
                if proposals is None:
                    proposals = iter(self.generator.generate(state.domains, changed_colors=changed,
                        cyclic_only=True, compact_domains=self.compact_domains))
                candidate = next(proposals, None)
                if candidate is None:
                    break
                self.statistics["proposed"] += 1
                signature = candidate.core_keys, candidate.open_colors
                scope = sorted(set().union(*(self.generator.scopes[key] for key in candidate.core_keys)))
                fingerprint = tuple(state.domains[c] for c in scope)
                if self.used.get(signature) == fingerprint:
                    self.statistics["skipped"] += 1
                    continue
                prepared = _runtime._prepare(state, candidate, compact_domains=self.compact_domains, **self.limits)
                if prepared is None:
                    self.statistics["rejected"] += 1
                    continue
                before = state.snapshot_domains() if self.on_application is not None else None
                result = _runtime.apply_candidate(state, candidate, prepared)
                attempts += 1
                useful = bool(result.removed_values or result.contradiction)
                self.statistics["applied"] += 1
                self.statistics["useful"] += int(useful)
                self.statistics["removed_values"] += result.removed_values
                self.statistics["contradictions"] += int(result.contradiction)
                if useful:
                    proposals = None
                    changed = set(self.colors)  # Do not lose unexplored regions on restart.
                else:
                    if len(self.used) >= self.generator.max_steps:
                        self.used.pop(next(iter(self.used)))
                    self.used[signature] = fingerprint
                if self.on_application is not None:
                    self.on_application(state, candidate, prepared[0], before, result)
            self.last = None if state.contradiction else tuple(state.domains[c] for c in self.colors)
        finally:
            self.statistics["time_s"] += perf_counter()-started


def attach_heuristic(state, *, rule_inspired=False, **options):
    """Optional adapter; the default calls the original runtime entry point."""
    if not isinstance(rule_inspired, bool):
        raise ValueError("rule_inspired must be Boolean")
    if not rule_inspired:
        return _attach_heuristic(state, **options)
    if state.on_closure is not None:
        raise ValueError("Heuristic clusters cannot be combined with another closure hook")
    state.on_closure = HeuristicPolicy(state, rule_inspired=True, **options)
    return state
