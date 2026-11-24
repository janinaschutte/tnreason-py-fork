"""Add preselected supervised masks and the 51-family expert to the reserved test."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import signal
from time import perf_counter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
HELDOUT = OUT.parent / "heldout"
SOURCE = ROOT / "experiments/cluster_selection/results/mask_sudoku_300ep.6lb43_yj/train"
METHODS = {"mask1": (1, False), "mask8": (8, False),
           "mask1_compact": (1, True), "mask8_compact": (8, True), "expert51": None}
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def freeze():
    from demonstrations.sudoku.inferenceClusters import RuleLimits, available_rules
    import torch
    prior = read(HELDOUT / "protocol.json")
    assert all(digest(p) == h for p, h in prior["source_sha256"].items())
    assert digest(prior["data"]) == prior["data_sha256"]
    selection = read(SOURCE / "selection.json")
    assert selection["checkpoint"] == "epoch_0020.pt" and selection["no_test_evaluation"]
    source = SOURCE / selection["checkpoint"]
    assert digest(source) == "669b317520c199c4cbba9259a302ff9481e9e06216bea73e63415f4483529e19"
    saved = torch.load(source, map_location="cpu", weights_only=True)
    assert saved["kind"] == "mask" and saved["network"] == read(prior["data"])["network"]
    assert saved["config"].get("compact_domains", False) is False
    with (OUT / "mask.pt").open("xb") as stream:
        stream.write(source.read_bytes())
    sources = {p: h for p, h in prior["source_sha256"].items()}
    for folder in ["demonstrations/sudoku/inferenceClusters", "demonstrations/sudoku/constraints"]:
        sources.update({str(p): digest(p) for p in (ROOT / folder).glob("*.py")})
    for path in [Path(__file__), ROOT / "experiments/sudoku_rule_clusters.py",
                 ROOT / "demonstrations/__init__.py", ROOT / "demonstrations/sudoku/__init__.py"]:
        if path.exists():
            sources[str(path)] = digest(path)
    references = [HELDOUT / name for name in ["protocol.json", "summary.json", "audit.json"]]
    references += list((HELDOUT / "rows").glob("*.json"))
    references += [OUT.parent / "continuations/final_test" / name
                   for name in ["protocol.json", "summary.json", "audit.json"]]
    references += [SOURCE / "selection.json"]
    order = [(method, i) for i in range(16) for method in METHODS]
    Random(2026111301).shuffle(order)
    plan = dict(kind="additional_exploratory_expert_and_supervised_comparison",
        data=prior["data"], data_sha256=prior["data_sha256"],
        checkpoint=str(OUT / "mask.pt"), checkpoint_sha256=digest(source),
        checkpoint_source=str(source), checkpoint_config=saved["config"],
        selection="Previously selected by old55 validation; no new test selection or tuning",
        source_sha256=sources, reference_sha256={str(p): digest(p) for p in references},
        methods=METHODS, order=order, nodes=10000, seconds=600, workers=4,
        branching="name", allowance=32, decoder_steps=8192, total_work=None,
        expert_families=list(available_rules()), expert_limits=asdict(RuleLimits()),
        expert_episode_cap=None,
        note="Same already queried16 test puzzles; exploratory. Frozen before any new solver queries. "
             "K1/K8 retain saved full-shape planning; predeclared compact variants match other constructors' "
             "domain-compacted planning. Weights identical. Expert has larger per-cluster bounds, unary "
             "projections and no generic episode cap; domain-specific reference, not matched computation.",
        reporting_preference="Always include the Sudoku-specific expert in comparison tables; mark unmeasured when absent")
    assert len(plan["expert_families"]) == 51
    write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir()
    print("FROZEN80cases; selectedmaskepoch20; expert51; four declared mask variants", flush=True)


def initialize():
    import numpy as np
    import torch
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in {**plan["source_sha256"], **plan["reference_sha256"]}.items())
    assert digest(plan["data"]) == plan["data_sha256"]
    assert digest(plan["checkpoint"]) == plan["checkpoint_sha256"]
    data = read(plan["data"])
    domains, _ = instances(data)
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=True)
    assert not instance_hashes(data, domains) & set(saved["training"]["train_instance_hashes"])
    model = runtime.load(plan["checkpoint"]).cpu().eval()
    assert model.network == data["network"] and model.config == plan["checkpoint_config"]
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = partial(network.solve_mac, variable_tie_break="name")
    CONTEXT.update(plan=plan, model=model, network=network, domains=domains)


def evaluate(task):
    import torch
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import solve, _valid_assignment
    method, position = task
    path = OUT / "rows" / f"{method}_{position:02d}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    plan, network, model = (CONTEXT[k] for k in ["plan", "network", "model"])
    domains = CONTEXT["domains"][position]
    if method == "expert51":
        from demonstrations.sudoku.inferenceClusters import RuleLimits
        from experiments.sudoku_rule_clusters import ClusterClosure
        from experiments.constraint_networks.mac import ConstraintNetworkSolver
        state = network.create_mac_state(domains)
        state.on_closure = policy = ClusterClosure(network, plan["expert_families"], RuleLimits(**plan["expert_limits"]))
        solver = ConstraintNetworkSolver(network, max_nodes=plan["nodes"], variable_tie_break="name")
        def timeout(*_):
            raise TimeoutError("Per-puzzle guard")
        previous = signal.signal(signal.SIGALRM, timeout)
        started = perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, plan["seconds"])
            result = solver.solve(state=state)
            valid = result.solved and _valid_assignment(network, domains, result.assignment or {})
            row = dict(status="solved" if valid else "invalid" if result.solved else "node_limit"
                       if result.node_limit_reached else "unsat", nodes=result.nodes,
                       solved=bool(valid), valid=bool(valid), assignment=result.assignment or {})
        except TimeoutError:
            row = dict(status="wall_timeout", nodes=None, solved=False, valid=False)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        stats = policy.statistics()
        families = stats["by_family"].values()
        row.update(solve_s=perf_counter()-started, cluster_statistics=stats,
            mac_statistics=asdict(state.statistics), work=dict(
                contractions=sum(f.get("contracted", 0) for f in families),
                estimated_work=sum(f.get("contraction_work", 0) for f in families),
                max_tensor_size=max((f.get("max_tensor_entries", 0) for f in families), default=0)))
        assert state._cores.keys() == network.coresDict.keys() and not state.decision_level and not state._added_core_keys
    else:
        anchors, compact = plan["methods"][method]
        model.config.update(anchor_candidates=anchors, compact_domains=compact)
        original_apply, original_attach = runtime.apply_candidate, runtime.attach
        applications, episodes = [], []
        def applied(state, candidate, prepared=None, **options):
            root = state.decision_level == 0
            result = original_apply(state, candidate, prepared, **options)
            applications.append(dict(root=root, removed=result.removed_values, contradiction=result.contradiction))
            return result
        def attached(state, *args, **options):
            result = original_attach(state, *args, **options)
            policy = state.on_closure
            def closure(state, changed):
                before = len(applications)
                try:
                    return policy(state, changed)
                finally:
                    count = len(applications)-before
                    assert count <= plan["allowance"]
                    episodes.append(count)
            closure.statistics = policy.statistics
            state.on_closure = closure
            return result
        runtime.apply_candidate, runtime.attach = applied, attached
        try:
            with torch.no_grad():
                row = solve(network, domains, "trained", model, nodes=plan["nodes"], seconds=plan["seconds"],
                    total_work=None, baseline_candidates=32, cluster_attempts=32, stop_on_noop=False, baseline_steps=8192)
        finally:
            runtime.apply_candidate, runtime.attach = original_apply, original_attach
        row.update(episode_contractions=episodes, root_contractions=sum(a["root"] for a in applications))
        assert row["work"]["contractions"] == len(applications) or row["status"] == "wall_timeout"
    assert row["status"] not in ("invalid", "unsat"), row
    row.update(method=method, position=position, protocol_sha256=digest(OUT / "protocol.json"))
    write(path, row)
    print(path.name, row["status"], row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
    return row


def report():
    rows = [read(p) for p in sorted((OUT / "rows").glob("*.json"))]
    assert len(rows) == 80 and len({(r["method"], r["position"]) for r in rows}) == 80
    totals = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        complete = sum(r["status"] == "solved" and r["valid"] for r in group)
        totals[method] = dict(instances=16, completed=complete,
            nodes=sum(r["nodes"] for r in group) if complete == 16 else None,
            branch_free=sum(r["status"] == "solved" and r["nodes"] == 1 for r in group),
            contractions=sum(r["work"]["contractions"] for r in group),
            estimated_work=sum(r["work"]["estimated_work"] for r in group),
            solve_seconds=sum(r["solve_s"] for r in group))
    write(OUT / "summary.json", dict(protocol_sha256=digest(OUT / "protocol.json"), results=totals,
        row_sha256={p.name: digest(p) for p in sorted((OUT / "rows").glob("*.json"))}))
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "evaluate", "report"])
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    elif phase == "evaluate":
        plan = read(OUT / "protocol.json")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"), initializer=initialize) as pool:
            for job in as_completed([pool.submit(evaluate, task) for task in plan["order"]]):
                job.result()
    else:
        report()
