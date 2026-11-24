"""Frozen, paired MAC comparison of optional generic rule-inspired proposals."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
from random import Random
import signal
import sys
from types import SimpleNamespace
from uuid import uuid4

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))
METHODS = ("plain", "cycles8", "cycles12", "cycles8_rules", "expert51")
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze():
    from demonstrations.sudoku.inferenceClusters import RuleLimits, available_rules
    prior = read(BASE / "frequent_mix/protocol.json")
    assert all(digest(p) == h for p, h in prior["source_sha256"].items())
    assert digest(prior["data"]) == prior["data_sha256"]
    sources = dict(prior["source_sha256"])
    files = [Path(__file__), ROOT / "experiments/cluster_selection/rule_heuristic.py",
             ROOT / "experiments/cluster_selection/tests/test_rule_heuristic.py",
             ROOT / "experiments/sudoku_rule_clusters.py"]
    files += sorted((ROOT / "demonstrations/sudoku/inferenceClusters").glob("*.py"))
    sources.update({str(p): digest(p) for p in files})
    order = [(method, i) for i in range(16) for method in METHODS]
    Random(2026111317).shuffle(order)
    write(OUT / "protocol.json", dict(kind="optional_generic_rule_heuristic_paired_mac",
        data=prior["data"], data_sha256=prior["data_sha256"], source_sha256=sources,
        methods=METHODS, order=order, workers=4, seconds=120, nodes=10000,
        branching="name", candidates=64, steps=16384,
        limits=dict(max_cores=64, max_open_colors=6, max_work=1000000, max_size=100000,
                    compact_domains=True), expert_families=available_rules(),
        expert_limits=asdict(RuleLimits()),
        note="Exploratory reuse of the same16 reserved puzzles, no training/checkpoint selection. "
             "Compare ordinary MAC, cycle8 heuristic, cycle12 control, and cycle8 with "
             "optional generic overlap/Hall proposals. Same generic candidate/step/core/output/tensor "
             "limits; enabled detector support reads are extra bounded preparation, so equal caps "
             "are not equal compute. All three structural arms use the SAME refreshed policy: "
             "regenerate after useful contraction, cache only no-op input snapshots. "
             "The expert has its own larger limits and is a separate reference. "
             "Complete MAC solving with name tie-break; summaries temporary, normal reversible domains. "
             "Time includes initial GAC, proposal setup and solver work, excludes network/state creation "
             "and final validation; concurrent single-thread CPU workers. Useful new-rule applications "
             "record exact pre/post domains and original cores for independent branch-snapshot replay; "
             "their protected transactions include trace capture, so a pending guard can overshoot one "
             "bounded application. Trace overhead is included in solving time. An unqueried pre-trace "
             "protocol is retained separately; all final settings are frozen before this study's queries. "
             "Frozen training sources untouched."))
    (OUT / "rows").mkdir()
    print("FROZEN80 paired MAC cases; no solver queries", flush=True)


def initialize():
    import numpy as np
    import torch
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in plan["source_sha256"].items())
    assert digest(plan["data"]) == plan["data_sha256"]
    data = read(plan["data"])
    domains, _ = instances(data)
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    original_solve = network.solve_mac
    network.solve_mac = lambda **kw: original_solve(variable_tie_break="name", **kw)
    CONTEXT.update(plan=plan, network=network, domains=domains)


def evaluate(task):
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.rule_heuristic import HeuristicPolicy
    from experiments.cluster_selection.reinforcement.experiment import solve
    from experiments.sudoku_rule_clusters import ClusterClosure, RuleLimits
    method, position = task
    path = OUT / "rows" / f"{method}_{position:02d}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    plan = CONTEXT["plan"]
    original_attach, original_apply = runtime.attach_heuristic, runtime.apply_candidate
    capture, applications, rule_traces = {}, [], []

    class Expert:
        def __init__(self, state):
            self.policy = ClusterClosure(CONTEXT["network"], plan["expert_families"],
                                         RuleLimits(**plan["expert_limits"]))

        def __call__(self, state, changed):
            return self.policy(state, changed)

        @property
        def statistics(self):
            return self.policy.statistics()

    def attach(state, **options):
        if method == "expert51":
            assert state.on_closure is None
            state.on_closure = Expert(state)
        else:
            if method == "cycles12":
                options["max_cycle_length"] = 12
            assert state.on_closure is None
            state.on_closure = HeuristicPolicy(state, rule_inspired=method == "cycles8_rules", **options)
        capture["policy"] = state.on_closure
        return state

    def applied(state, candidate, prepared=None, **options):
        root = state.decision_level == 0
        is_rule = candidate.key.startswith("rule_")
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            before = state.snapshot_domains() if is_rule else None
            result = original_apply(state, candidate, prepared, **options)
            family = candidate.key.split("_")[1] if is_rule else "cycle"
            applications.append(dict(family=family, root=root, removed=result.removed_values,
                                     contradiction=result.contradiction))
            if is_rule and (result.removed_values or result.contradiction):
                rule_traces.append(dict(core_keys=candidate.core_keys, open_colors=candidate.open_colors,
                    before=before, after=state.snapshot_domains(), level=state.decision_level,
                    removed=result.removed_values, contradiction=result.contradiction,
                    estimated_work=prepared[0].work, max_tensor_size=prepared[0].max_size))
            return result
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    runtime.attach_heuristic, runtime.apply_candidate = attach, applied
    try:
        row = solve(CONTEXT["network"], CONTEXT["domains"][position],
            "plain" if method == "plain" else "heuristic", SimpleNamespace(config=plan["limits"]),
            nodes=plan["nodes"], seconds=plan["seconds"], total_work=None,
            baseline_candidates=plan["candidates"], baseline_steps=plan["steps"])
    finally:
        runtime.attach_heuristic, runtime.apply_candidate = original_attach, original_apply
    assert row["status"] not in ("invalid", "unsat")
    if method == "expert51":
        families = row["cluster_statistics"]["by_family"].values()
        row["work"] = dict(contractions=sum(c.get("contracted", 0) for c in families),
            estimated_work=None, max_tensor_size=None,
            charged_work=sum(c.get("contraction_work", 0) for c in families),
            largest_considered_entries=max((c.get("max_tensor_entries", 0) for c in families), default=0),
            skipped_clusters=sum(c.get("skipped_budget", 0) for c in families),
            accounting_note="Exact executor charges reads/joins, including skipped prefixes; generic estimated work is not measured.")
    counts = defaultdict(Counter)
    for event in applications:
        counts[event["family"]].update(contractions=1, root_contractions=int(event["root"]),
            useful=int(bool(event["removed"] or event["contradiction"])), removed=event["removed"])
    policy = capture.get("policy")
    generator = getattr(policy, "generator", None)
    row.update(method=method, position=position, protocol_sha256=digest(OUT / "protocol.json"),
        application_counts=dict(counts), rule_statistics=dict(getattr(generator, "rule_statistics", {})),
        useful_rule_traces=rule_traces)
    write(path, row)
    print(method, position, row["status"], row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
    return row


def report():
    paths = sorted((OUT / "rows").glob("*.json"))
    rows = [read(path) for path in paths]
    assert len(rows) == 80 and len({(r["method"], r["position"]) for r in rows}) == 80
    results = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        complete = all(r["status"] == "solved" and r["valid"] for r in group)
        results[method] = dict(solved=sum(r["solved"] for r in group),
            dfs_nodes=sum(r["nodes"] for r in group) if complete else None,
            branch_free=sum(r["solved"] and r["nodes"] == 1 for r in group),
            root_positions=[r["position"] for r in group if r["solved"] and r["nodes"] == 1],
            solve_seconds=sum(r["solve_s"] for r in group),
            censored=[r["position"] for r in group if r["status"] != "solved"])
    write(OUT / "summary.json", dict(protocol_sha256=digest(OUT / "protocol.json"),
        results=results, row_sha256={p.name: digest(p) for p in paths}))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate", "report"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "report":
        report()
    else:
        plan = read(OUT / "protocol.json")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"),
                                 initializer=initialize) as pool:
            for job in as_completed([pool.submit(evaluate, task) for task in plan["order"]]):
                job.result()
