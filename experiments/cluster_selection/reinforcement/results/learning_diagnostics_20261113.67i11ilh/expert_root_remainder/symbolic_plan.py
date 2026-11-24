"""Trace the expert executor's exact scope plan without allocating joins."""
from dataclasses import replace
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import signal
from time import perf_counter

from demonstrations.sudoku.inferenceClusters import core

OUT = Path(__file__).resolve().parent


class ShapeOnly:
    def transpose(self, _):
        return self


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    baseline = json.loads((OUT / "expert_root.json").read_text())
    actual = json.loads((OUT / "complete_support.json").read_text())["contractions"][0]
    data_path = next(p for p in protocol["references"] if p.endswith("unique_test.json"))
    data = json.loads(Path(data_path).read_text())
    raw = data["network"]
    scopes = [tuple(raw["colors"][i] for i in c["colors"]) for c in raw["cores"]]
    colors = tuple(dict.fromkeys(c for scope in scopes for c in scope))
    dimensions = {c: len(baseline["domains"][c]) for c in colors}
    limits = core.RuleLimits(**protocol["complete_limits"])
    unbounded = replace(limits, max_size=2**4096, max_work=2**4096,
                        max_planning_steps=2**4096)
    budget = core._Work(unbounded)
    for c in raw["cores"]:
        budget.charge(prod(c["shape"]), prod(c["shape"]))
    factors = []
    for scope in scopes:
        size = prod(dimensions[c] for c in scope)
        budget.charge(size, size)
        factors.append(core._Factor(tuple(c for c in scope if dimensions[c] != 1), ShapeOnly()))
    constants = [f for f in factors if not f.colors]
    if constants:
        budget.charge(len(constants), 1)
        factors = [f for f in factors if f.colors] + [core._Factor((), ShapeOnly())]
    joins, order = [], []
    original_join, original_eliminate = core._join_reduce, core._eliminate

    def join(factors, eliminate, dimensions, budget):
        scope = tuple(dict.fromkeys(c for f in factors for c in f.colors))
        size = prod(dimensions[c] for c in scope)
        joins.append(dict(scope=scope, eliminate=tuple(eliminate), entries=size,
            factors=len(factors), work_before=budget.work, planning_steps=budget.planning_steps))
        budget.charge(size * (len(factors) + bool(eliminate)), size)
        return core._Factor(tuple(c for c in scope if c not in eliminate), ShapeOnly())

    def eliminate(factors, color, dimensions, budget):
        order.append(color)
        return original_eliminate(factors, color, dimensions, budget)

    def timeout(*_):
        raise TimeoutError("600-second symbolic planning guard")

    core._join_reduce, core._eliminate = join, eliminate
    previous = signal.signal(signal.SIGALRM, timeout)
    started = perf_counter()
    try:
        signal.setitimer(signal.ITIMER_REAL, 600)
        for color in colors:
            if color.startswith("a_") and color != actual["output"]:
                budget.plan(len(factors))
                if sum(color in f.colors for f in factors) <= 2:
                    factors = core._eliminate(factors, color, dimensions, budget)
        budget.charge(9, 9)  # Full categorical unary output, as in the executor.
        core._project(factors, (actual["output"],), dimensions, budget)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        core._join_reduce, core._eliminate = original_join, original_eliminate
    first = next(j for j in joins if j["entries"] > limits.max_size)
    assert first["entries"] == actual["max_size"]
    assert first["work_before"] == actual["work"]
    assert first["planning_steps"] == actual["planning_steps"]
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in {**protocol["source_sha256"], **protocol["references"]}.items())
    result = dict(kind="Shape-only replay of exact executor greedy path; no tensor allocation or solving",
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), seconds=perf_counter()-started,
        core_keys=[c["key"] for c in raw["cores"]], open_colors=[actual["output"]],
        estimated_work=budget.work, maximum_entries=budget.max_size,
        planning_steps=budget.planning_steps, maximum_rank=max(len(j["scope"]) for j in joins),
        elimination_order=order, joins=joins, first_actual_budget_failure=first,
        actual_execution_prefix_reproduced_exactly=True,
        limitation="One deterministic greedy path; not a lower bound on every possible contraction order")
    with (OUT / "symbolic_plan.json").open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print({k: result[k] for k in ["estimated_work", "maximum_entries", "planning_steps", "maximum_rank", "seconds"]})


if __name__ == "__main__":
    main()
