"""Verified, bounded simplification for success-imitation examples.

These are edited teacher sequences, never on-policy REINFORCE samples. Every
accepted replacement preserves the complete GAC outcome, including auxiliary
variables. Greedy deletion gives no global minimum guarantee.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
from time import perf_counter

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from ..candidate_evaluation_mac import evaluate_candidate
from ..candidate_evaluation_tnreason import SummaryClusterCandidate


@dataclass(frozen=True)
class CompactExample:
    domains: tuple[tuple[int, ...], ...]
    trace: tuple[tuple[str, int], ...]
    core_keys: tuple[str, ...]
    open_colors: tuple[str, ...]
    removed_values: int


@dataclass(frozen=True)
class CompactResult:
    examples: tuple[CompactExample, ...]
    probes: int
    rejected: int
    original_cores: int
    final_cores: int
    original_removed: int
    cost_seconds: float


def _candidate(state, keys, outputs):
    scope = {color for key in keys for color in state._cores[key].colors}
    outputs = tuple(color for color in outputs if color in scope)
    keys = tuple(keys)
    name = "compact_" + sha1(repr((keys, outputs)).encode()).hexdigest()
    return SummaryClusterCandidate(name, keys, outputs)


def _trace(model, candidate, domains):
    """Check free-policy support without an encoder/distribution evaluation."""
    try:
        trace = model.candidate_trace(candidate, domains)
    except ValueError:
        return None
    sizes = [len(domains[color]) for color in model.network["colors"]]
    keys, opened, phase = [], [], False
    for action in trace:
        actions = model.actions(keys, opened, phase, sizes)
        if model.config.get("decoder") == "sampling_grow":
            actions = [actions[idx] for idx in model._continuations(actions, keys, opened, phase)]
        if action not in actions:
            return None
        kind, idx = action
        if kind == "core":
            keys.append(idx)
        elif kind == "open":
            opened.append(idx)
        elif kind == "finish":
            phase = True
    return tuple(trace)


def _components(state, candidate):
    """Active-variable components, matching the constructor's frontier mask."""
    remaining, result = set(candidate.core_keys), []
    scopes = {key: {color for color in state._cores[key].colors if len(state.domains[color]) > 1}
              for key in remaining}
    while remaining:
        component = {next(key for key in candidate.core_keys if key in remaining)}
        remaining -= component
        reached = set().union(*(scopes[key] for key in component))
        while True:
            following = {key for key in remaining if scopes[key] & reached}
            if not following:
                break
            component |= following
            remaining -= following
            reached.update(*(scopes[key] for key in following))
        result.append(tuple(key for key in candidate.core_keys if key in component))
    return result


def _peel(state, candidate):
    """Remove GAC-satisfied leaf factors; the complete result is still checked."""
    keys, outputs = list(candidate.core_keys), set(candidate.open_colors)
    while keys:
        scopes = {key: {color for color in state._cores[key].colors if len(state.domains[color]) > 1}
                  for key in keys}
        counts = Counter(color for scope in scopes.values() for color in scope)
        leaves = {key for key in keys
                  if sum(color in outputs or counts[color] > 1 for color in scopes[key]) <= 1}
        if not leaves:
            break
        keys = [key for key in keys if key not in leaves]
    return _candidate(state, keys, candidate.open_colors)


def compact_success(model, state, candidate, *, max_probes=64, split=False):
    """Return legal compact teachers, leaving the supplied state untouched.

    ``max_probes`` counts all contracted trials, including the original outcome
    and rejected over-budget plans. Splitting first uses active components; if
    connected, it tries separate retained outputs. Pieces are measured from
    their actual sequential input domains and accepted only if each prunes and
    their final full-domain snapshot equals the original outcome. A legal
    single-cluster reduction remains the fallback.
    """
    if not isinstance(max_probes, int) or isinstance(max_probes, bool) or max_probes < 1:
        raise ValueError("Expected a positive integer simplification probe limit")
    if not isinstance(split, bool) or not model.actor_config.get("learned_stopping"):
        raise ValueError("Compact replay requires boolean split and learned stopping")
    if state.contradiction or state._queue or state._queued:
        raise ValueError("Compact replay requires a noncontradictory GAC-closed state")
    started, before = perf_counter(), state.snapshot_domains()
    colors = model.network["colors"]
    limits = {key: model.config[key] for key in ("max_work", "max_size")}
    limits["compact_domains"] = model.config.get("compact_domains", False)
    probes = rejected = 0
    removed = 0

    def finish(examples=()):
        return CompactResult(tuple(examples), probes, rejected, len(candidate.core_keys),
                             sum(len(item.core_keys) for item in examples), removed,
                             perf_counter() - started)

    def evaluate(current, proposed):
        nonlocal probes, rejected
        if probes >= max_probes or not proposed.core_keys:
            return None
        probes += 1
        try:
            return evaluate_candidate(current, proposed, cost_weight=0., **limits)
        except ValueError as error:
            if str(error) not in ("Candidate exceeds the contraction work or tensor-size budget",
                                  "NumPy contraction supports at most 52 distinct colors"):
                raise
            rejected += 1
            return None

    def example(current, proposed, outcome):
        trace = _trace(model, proposed, current.domains)
        if trace is None:
            return None
        count = sum(len(current.domains[color]) - len(outcome.domains[color]) for color in colors)
        return CompactExample(tuple(tuple(current.domains[color]) for color in colors), trace,
                              proposed.core_keys, proposed.open_colors, count)

    target = evaluate(state, candidate)
    if target is None or target.contradiction or target.domains == before:
        return finish()
    removed = sum(len(before[color]) - len(target.domains[color]) for color in colors)
    current, best = candidate, example(state, candidate, target)
    peeled = _peel(state, candidate)
    if peeled.core_keys and len(peeled.core_keys) < len(current.core_keys):
        outcome = evaluate(state, peeled)
        if outcome is not None and not outcome.contradiction and outcome.domains == target.domains:
            current = peeled
            legal = example(state, current, outcome)
            if legal is not None:
                best = legal
    # Reserve enough independent checks for a small decomposition. No trial
    # gets another construction's reward; edits only populate imitation replay.
    reserve = min(len(candidate.open_colors) + 1, max(0, (max_probes - 1) // 3)) if split else 0
    for key in tuple(current.core_keys):
        if probes >= max_probes - reserve or len(current.core_keys) <= 1:
            break
        proposed = _candidate(state, [item for item in current.core_keys if item != key], current.open_colors)
        outcome = evaluate(state, proposed)
        if outcome is not None and not outcome.contradiction and outcome.domains == target.domains:
            current = proposed
            legal = example(state, current, outcome)
            if legal is not None:
                best = legal
    fallback = () if best is None else (best,)
    if not split or probes >= max_probes:
        return finish(fallback)

    components = _components(state, current)
    if len(components) > 1:
        pieces = [_candidate(state, keys, current.open_colors) for keys in components]
    elif len(current.open_colors) > 1:
        pieces = [_candidate(state, current.core_keys, (color,)) for color in current.open_colors]
    else:
        return finish(fallback)
    # The scratch state prevents even temporary sequential restrictions from
    # touching caller domains, backtracking trails, hooks or counters.
    scratch = ConstraintNetwork(state._cores, maintain_indices=True).create_mac_state(before)
    if not scratch.propagate() or scratch.snapshot_domains() != before:
        raise AssertionError("Expected an unchanged GAC-closed scratch state")
    examples = []
    for position, piece in enumerate(pieces):
        if scratch.snapshot_domains() == target.domains:
            break
        if not piece.open_colors:
            continue
        outcome = evaluate(scratch, piece)
        if outcome is None or outcome.contradiction or outcome.domains == scratch.snapshot_domains():
            continue
        if any(not set(target.domains[color]) <= set(outcome.domains[color]) for color in colors):
            continue
        # Component cores already went through deletion above. Retrying them
        # first can spend the budget before later useful pieces are checked.
        deletions = () if len(components) > 1 else tuple(piece.core_keys)
        for key in deletions:
            remaining_checks = len(pieces) - position  # Other pieces plus a residual check.
            if probes >= max_probes - remaining_checks or len(piece.core_keys) <= 1:
                break
            proposed = _candidate(scratch, [item for item in piece.core_keys if item != key], piece.open_colors)
            smaller = evaluate(scratch, proposed)
            if (smaller is not None and not smaller.contradiction and smaller.domains == outcome.domains
                    and _trace(model, proposed, scratch.domains) is not None):
                piece, outcome = proposed, smaller
        teacher = example(scratch, piece, outcome)
        if teacher is None:
            continue
        examples.append(teacher)
        for color in colors:
            scratch.restrict_domain(color, outcome.domains[color])
        if not scratch.propagate() or scratch.snapshot_domains() != outcome.domains:
            raise AssertionError("A verified temporary summary left domains outside GAC closure")
    if scratch.snapshot_domains() != target.domains and probes < max_probes:
        residual = evaluate(scratch, current)
        if residual is not None and not residual.contradiction and residual.domains == target.domains:
            teacher = example(scratch, current, residual)
            if teacher is not None and teacher.removed_values > 0:
                examples.append(teacher)
                for color in colors:
                    scratch.restrict_domain(color, residual.domains[color])
                scratch.propagate()
    total = sum(len(item.core_keys) for item in examples)
    ceiling = len(candidate.core_keys) if best is None else len(best.core_keys)
    if (len(examples) > 1 and total <= ceiling and scratch.snapshot_domains() == target.domains
            and max(len(item.core_keys) for item in examples) < ceiling):
        return finish(examples)
    return finish(fallback)
