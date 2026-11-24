"""One shared root-propagation budget, with frequent time-balanced source mixing."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
from random import Random
import sys
from time import perf_counter

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
SOURCES = ("heuristic", "teacher", "reinforce", "replay", "q")
spec = importlib.util.spec_from_file_location("frozen_matched_time_helpers", PARENT / "run.py")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
helper.OUT = OUT
read, digest, write = helper.read, helper.digest, helper.write


class Combined:
    def __init__(self, models):
        self.models = models
        self.reset()

    def reset(self):
        self.start = None
        self.spent = {source: 0.0 for source in SOURCES}
        self.events, self.progress = [], []
        self.generation, self.active = 0, None

    def clock(self):
        now = perf_counter()
        if self.start is None:
            self.start = now
        return now

    def finish(self, event, now):
        with helper.atomic():
            if "end_s" not in event:
                event["end_s"] = now-self.start
                event["charged_s"] = event["end_s"]-event["start_s"]
                assert event["charged_s"] >= 0
                self.spent[event["source"]] += event["charged_s"]
            if self.active is event:
                self.active = None

    def proposals(self, domains, **unused):
        generation = self.generation
        self.generation += 1
        streams, exhausted = {}, set()
        try:
            while len(exhausted) < len(SOURCES):
                source = min((s for s in SOURCES if s not in exhausted), key=self.spent.get)
                with helper.atomic():
                    event = dict(source=source, generation=generation, start_s=perf_counter()-self.start)
                    self.events.append(event)
                    self.active = event
                try:
                    if source not in streams:
                        if source == "heuristic":
                            info = dict(generation=generation, stages=[])
                            self.progress.append(info)
                            stream = helper.structural(domains, info["stages"])
                        else:
                            bound = helper.CONTEXT["plan"]["technical_iterator_bound"]
                            stream = self.models[source].proposals(domains, max_candidates=bound, max_steps=bound)
                        streams[source] = iter(stream)
                    candidate = next(streams[source], None)
                    if candidate is None:
                        event["exhausted"] = True
                        exhausted.add(source)
                        continue
                    event["candidate_key"] = candidate.key
                    yield candidate
                finally:
                    self.finish(event, perf_counter())
        finally:
            for stream in streams.values():
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

    def publish(self, path, row):
        if row.get("method") == "combined" and "prefixes" in row:
            if self.active is not None:
                self.finish(self.active, self.start+row["wall_s"])
            by_generation = {}
            for event in self.events:
                if "candidate_key" in event:
                    by_generation[event["generation"]] = event
            for proof in row["proofs"]:
                event = by_generation[proof["before"]]
                assert event["candidate_key"] == proof["candidate_key"]
                proof["source"] = event["source"]
            useful = Counter(proof["source"] for proof in row["proofs"])
            row["source_totals"] = {source: dict(charged_s=self.spent[source],
                turns=sum(event["source"] == source for event in self.events), useful=useful[source])
                for source in SOURCES}
            row["unattributed_overhead_s"] = row["wall_s"]-sum(self.spent.values())
            assert row["unattributed_overhead_s"] >= -1e-8
            row["source_events"] = self.events
            row["structural_progress"] = self.progress
        return write(path, row)


def freeze():
    parent = read(PARENT / "protocol.json")
    assert all(digest(path) == expected for path, expected in
               {**parent["source_sha256"], **parent["reference_sha256"]}.items())
    assert digest(parent["data"]) == parent["data_sha256"]
    assert all(digest(info["path"]) == info["sha256"] for info in parent["checkpoints"].values())
    plan = deepcopy(parent)
    order = [("combined", position) for position in range(16)]
    Random(2026111502).shuffle(order)
    plan.update(kind="matched_time_shared_generic_portfolio", methods=["combined"], sources=SOURCES,
        order=order, source_order=SOURCES, scheduling="least cumulative charged source wall time; one candidate turn",
        parent_protocol=str(PARENT / "protocol.json"), parent_protocol_sha256=digest(PARENT / "protocol.json"))
    plan["source_sha256"].update({str(path): digest(path) for path in OUT.glob("*.py")})
    plan["source_sha256"][str(PARENT / "audit.py")] = digest(PARENT / "audit.py")
    plan["reference_sha256"][str(PARENT / "protocol.json")] = digest(PARENT / "protocol.json")
    plan["interpretation"] = (
        "Same reused16 puzzles/checkpoints/hardware/per-cluster limits and exact timed evaluate/residual "
        "helpers as the parent. One aggregate60s budget includes ordinary GAC and every source's proposal, "
        "planning, contraction and GAC; no private60s allowance. Sources heuristic/teacher/REINFORCE/replay/Q "
        "target equal cumulative wall shares by advancing the least-charged source one candidate at a time. "
        "A turn includes cache/rejection screening and any application/GAC/accounting while its adapter "
        "yield is suspended. Turns are nonpreemptive except for the aggregate deadline; a slow proposal "
        "or bounded atomic contraction can exceed its ideal share. There is no hard per-source quota. "
        "All source iterators restart after useful pruning from shared updated domains; cumulative time "
        "balances persist across these generations. Global exact scoped no-op caching remains unchanged. "
        "Per-source timing excludes small adapter gaps and initial GAC, recorded as unattributed overhead. "
        "No random constructor, mask decoder, domain-specific expert, DFS, assumptions or probes in mix. "
        "Prefixes1/5/15/60s credit only fully completed contraction+GAC snapshots by their recorded deadline. "
        "Per-source useful-cluster attribution records the actual constructor that emitted that cluster; "
        "it does not estimate counterfactual causal contribution.")
    write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir()
    print("FROZEN 16 shared-time portfolio roots; no solving queries", flush=True)


def initialize():
    helper.initialize()
    model = Combined(helper.CONTEXT["models"])
    helper.CONTEXT["models"]["combined"] = model
    helper.perf_counter, helper.write = model.clock, model.publish
    return model


def report():
    plan = read(OUT / "protocol.json")
    paths = sorted((OUT / "rows").glob("combined_[0-9][0-9].json"))
    rows = [read(path) for path in paths]
    assert len(rows) == 16 and {row["position"] for row in rows} == set(range(16))
    assert all(row["protocol_sha256"] == digest(OUT / "protocol.json") for row in rows)
    result = dict(instances=16, solved_by_budget={str(b): sum(row["prefixes"][str(b)]["solved"] for row in rows)
        for b in plan["budgets_s"]}, wall_s=sum(row["wall_s"] for row in rows),
        cpu_s=sum(row["cpu_s"] for row in rows), statuses=dict(Counter(row["status"] for row in rows)),
        solved_times={row["position"]: row["solved_s"] for row in rows if row["valid"]},
        source_totals={source: {metric: sum(row["source_totals"][source][metric] for row in rows)
            for metric in ("charged_s", "turns", "useful")} for source in SOURCES})
    write(OUT / "summary.json", dict(protocol_sha256=digest(OUT / "protocol.json"), results={"combined": result},
        row_sha256={path.name: digest(path) for path in sorted((OUT / "rows").glob("*.json"))}))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "initialize", "evaluate", "residual", "report"))
    parser.add_argument("--positions", type=int, nargs="+")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "report":
        report()
    else:
        model = initialize()
        if args.phase != "initialize":
            for method, position in helper.CONTEXT["plan"]["order"]:
                if args.positions is None or position in args.positions:
                    model.reset()
                    (helper.evaluate if args.phase == "evaluate" else helper.residual)(method, position)
