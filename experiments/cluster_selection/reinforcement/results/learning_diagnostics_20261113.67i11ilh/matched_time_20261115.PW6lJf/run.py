"""Matched-time, single-thread root propagation; no decisions or DFS in the timer."""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from random import Random
import signal
import sys
from time import perf_counter, process_time
from types import SimpleNamespace
from uuid import uuid4

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))
METHODS = ("plain", "heuristic", "initial", "teacher", "reinforce", "replay", "q", "expert")
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Exclusive atomic publication outside every measured interval."""
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


def forbid(*args, **kwargs):
    raise AssertionError("No assignment or DFS in matched-time root propagation")


@contextmanager
def atomic():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def freeze():
    import torch
    from demonstrations.sudoku.inferenceClusters import RuleLimits, available_rules
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
    old = read(BASE / "expert_supervised/protocol.json")
    assert all(digest(path) == hash_ for path, hash_ in old["source_sha256"].items())
    assert digest(old["data"]) == old["data_sha256"]
    data = read(old["data"])
    domains, _ = instances(data)
    assert len(domains) == 16
    checkpoints = {}
    paths = {name: BASE / f"heldout/{name}.pt" for name in ("initial", "teacher", "reinforce")}
    paths.update(replay=BASE / "continuations/final_test/self_imitation.pt", q=BASE / "generic_portfolio/q.pt")
    for name, path in paths.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["network"] == data["network"]
        assert not instance_hashes(data, domains) & set(saved["training"]["train_instance_hashes"])
        checkpoints[name] = dict(path=str(path), sha256=digest(path), epoch=saved["training"]["epoch"],
            config=saved["config"], beta=saved["beta"], training_overlap=0)
    assert checkpoints["q"]["config"].get("decoder", "depth_first") == "depth_first"
    references = {i: str(BASE / f"expert_supervised/rows/expert51_{i:02d}.json") for i in range(16)}
    available = sorted(os.sched_getaffinity(0))
    topology = {}
    for cpu in available:
        folder = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        topology[cpu] = {name: (folder / name).read_text().strip()
                         for name in ("core_id", "physical_package_id", "thread_siblings_list")}
    cpu = next((c for c in available if topology[c]["core_id"] == "3"), available[0])
    order = [(method, position) for method in METHODS for position in range(16)]
    Random(2026111501).shuffle(order)
    sources = {**old["source_sha256"], str(Path(__file__)): digest(__file__)}
    plan = dict(kind="matched_time_root_propagation", data=old["data"], data_sha256=old["data_sha256"],
        methods=METHODS, positions=list(range(16)), budgets_s=[1, 5, 15, 60], seconds=60,
        order=order, seed=2026111501, workers=1, cpu=cpu, original_affinity=available, cpu_topology=topology,
        source_sha256=sources, checkpoints=checkpoints, reference_assignments=references,
        reference_sha256={path: digest(path) for path in references.values()},
        limits=dict(max_cores=64, max_open_colors=6, max_work=1_000_000, max_size=100_000),
        compact_domains=True, contraction_cap=None, cumulative_work_cap=None,
        technical_iterator_bound=2**31-1, structural_initial_quota=32, structural_initial_steps=8192,
        expert_families=list(available_rules()), expert_limits=asdict(RuleLimits()),
        environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
            torch_threads=1, blas_threads=1), residual_dfs=dict(max_nodes=10000, seconds=30, branching="name"),
        timing="One serial pinned CPU process; wall-clock budget includes initial ordinary GAC, proposal "
            "adapter/context setup, generation/decoding, planning, contraction and GAC. Model/network/state "
            "loading, source hashing, final validation, disk publication and residual DFS are excluded. "
            "Closed-domain snapshots are copied in RAM inside the timed interval. Completion timestamps "
            "are sampled after full contraction+GAC and lightweight accounting, before snapshot copying. Both wall and CPU timestamps "
            "are recorded. SIGALRM interrupts read-only proposals/planning; blocked during initial GAC "
            "and domain mutation/contraction+GAC. Atomic completion after a budget earns no earlier credit.",
        interpretation="Exploratory reused16 puzzles; no training or checkpoint selection. Single predeclared "
            "trajectory per method/puzzle supplies all time prefixes. Original frozen decoder configurations "
            "are preserved: random/teacher/REINFORCE/replay sampling_grow, Q depth_first. Heuristic uses "
            "progressive nested32/64/128/... quotas with8192/16384/32768/... traversal steps, canonical "
            "deduplication and refresh after pruning. No aggregate contraction cap. Technical iterator limits "
            "are declared and counted if reached, not assumed mathematical exhaustion. Exact no-op cache "
            "is keyed by signature and PRE-contraction scope domains; every useful closure restarts proposals. "
            "Expert preserves51 ordered Sudoku-specific detectors and larger historical limits, a separate "
            "domain-specific reference. Only original-core contractions prune; no DFS, probes or assumptions. "
            "Plain MAC only runs ordinary GAC. A separate unmeasured phase can finish each saved closed "
            "prefix with identical plain MAC to count remaining DFS nodes, including the root.")
    assert len(plan["expert_families"]) == 51
    write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir()
    print(f"FROZEN 128 roots; budgets1/5/15/60s; serial CPU{cpu}; no solver queries", flush=True)


def initialize():
    import numpy as np
    import torch
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from tnreason.engine.workload_to_numpy import NumpyCore
    plan = read(OUT / "protocol.json")
    assert all(digest(path) == hash_ for path, hash_ in
               {**plan["source_sha256"], **plan["reference_sha256"]}.items())
    assert digest(plan["data"]) == plan["data_sha256"]
    os.sched_setaffinity(0, {plan["cpu"]})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["seed"])
    models = {}
    for name, checkpoint in plan["checkpoints"].items():
        assert digest(checkpoint["path"]) == checkpoint["sha256"]
        models[name] = runtime.load(checkpoint["path"]).cpu().eval()
        assert models[name].config == checkpoint["config"]
    data = read(plan["data"])
    raw = data["network"]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    network.solve_mac = forbid
    CONTEXT.update(plan=plan, models=models, network=network, domains=instances(data)[0], runtime=runtime)


def structural(domains, progress):
    from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
    plan = CONTEXT["plan"]
    raw = CONTEXT["models"]["teacher"].network
    cores = {core["key"]: SimpleNamespace(shape=core["shape"],
             colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]}
    quota, steps, seen = plan["structural_initial_quota"], plan["structural_initial_steps"], set()
    while quota <= plan["technical_iterator_bound"]:
        info = dict(quota=quota, traversal_steps=steps, emitted=0, completed=False)
        progress.append(info)
        generator = ClusterCandidateGenerator(cores, max_candidates=quota, max_steps=steps, **plan["limits"])
        for candidate in generator.generate(domains, cyclic_only=True, compact_domains=True):
            signature = tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))
            if signature not in seen:
                seen.add(signature)
                info["emitted"] += 1
                yield candidate
            if len(seen) >= quota:
                break
        info["completed"] = True
        quota, steps = quota*2, steps*2


def expert(domains, statistics):
    from demonstrations.sudoku.inferenceClusters import RuleContext, RuleLimits, RuleRegistry
    plan = CONTEXT["plan"]
    context = RuleContext(CONTEXT["network"], domains, num=3, limits=RuleLimits(**plan["expert_limits"]))
    context.options = lru_cache(maxsize=None)(context.options)
    seen = set()
    for family in plan["expert_families"]:
        detection = RuleRegistry((family,)).detect(context)[family]
        statistics["family_scans"] += 1
        statistics["detection_steps"] += detection.steps
        statistics["truncated_family_scans"] += int(detection.truncated)
        for cluster in detection.clusters:
            signature = cluster.core_keys, cluster.open_colors
            if family == "singles" or signature in seen:
                continue
            seen.add(signature)
            yield context, cluster


def evaluate(method, position):
    import torch
    from demonstrations.sudoku.inferenceClusters import contract_cluster
    from experiments.cluster_selection.reinforcement.experiment import _valid_assignment
    plan, network, models, runtime = (CONTEXT[k] for k in ("plan", "network", "models", "runtime"))
    path = OUT / "rows" / f"{method}_{position:02d}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    input_domains = CONTEXT["domains"][position]
    state = network.create_mac_state(input_domains)
    state.assign = forbid
    original_keys = set(state._cores)
    counts, snapshots, proofs, progress = Counter(), [], [], []
    current, used, stream = None, {}, None
    active_phase, status, solved_s, baseline_s = "ordinary_gac", "iterator_exhausted", None, None
    start_wall, start_cpu = perf_counter(), process_time()

    def elapsed():
        return perf_counter()-start_wall, process_time()-start_cpu

    def checkpoint(reason):
        nonlocal current, solved_s
        assert set(state._cores) == original_keys and not state._added_core_keys
        assert not state._queue and not state._queued and state.decision_level == 0 and state.on_closure is None
        solved = state.is_solved()
        wall_s, cpu_s = elapsed()
        if solved and solved_s is None:
            solved_s = wall_s
        current = state.snapshot_domains()
        snapshots.append(dict(reason=reason, wall_s=wall_s, cpu_s=cpu_s, solved=solved, domains=current,
            closed=True, no_added_cores=True, decision_level=0))

    def timeout(*_):
        raise TimeoutError("Root propagation budget reached")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, plan["seconds"])
    try:
        with atomic():
            state.propagate()
            checkpoint("ordinary_gac")
            baseline_s = snapshots[-1]["wall_s"]
        with torch.no_grad():
            while method != "plain" and not state.is_solved() and not state.contradiction:
                if elapsed()[0] >= plan["seconds"]:
                    raise TimeoutError("Root propagation budget reached")
                active_phase, phase_start = "proposal", perf_counter()
                counts["proposal_calls"] += 1
                try:
                    if stream is None:
                        counts["streams_created"] += 1
                        if method == "heuristic":
                            generation = dict(snapshot=len(snapshots)-1, stages=[])
                            progress.append(generation)
                            stream = iter(structural(current, generation["stages"]))
                        elif method == "expert":
                            stream = iter(expert(current, counts))
                        else:
                            stream = iter(models[method].proposals(current,
                                max_candidates=plan["technical_iterator_bound"], max_steps=plan["technical_iterator_bound"]))
                    proposal = next(stream, None)
                finally:
                    counts["proposal_s"] += perf_counter()-phase_start
                if proposal is None:
                    status = "expert_bounded_detectors_exhausted" if method == "expert" else "technical_iterator_exhausted"
                    break
                counts["proposals"] += 1
                candidate = proposal[1].candidate if method == "expert" else proposal
                signature = tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))
                scope = sorted({color for key in candidate.core_keys for color in state._cores[key].colors})
                fingerprint = tuple(current[color] for color in scope)
                if used.get(signature) == fingerprint:
                    counts["cache_hits"] += 1
                    continue
                active_phase, phase_start = "planning_and_expert_contraction" if method == "expert" else "planning", perf_counter()
                try:
                    if method == "expert":
                        result = contract_cluster(*proposal)
                        prepared = result if result.status == "executed" else None
                    else:
                        assert len(candidate.core_keys) <= 64 and len(candidate.open_colors) <= 6
                        prepared = runtime._prepare(state, candidate, compact_domains=True,
                            max_work=plan["limits"]["max_work"], max_size=plan["limits"]["max_size"])
                finally:
                    counts["planning_s"] += perf_counter()-phase_start
                if prepared is None:
                    counts["rejected"] += 1
                    continue
                if elapsed()[0] >= plan["seconds"]:
                    raise TimeoutError("No atomic application may begin beyond the deadline")
                active_phase, phase_start = "contraction_and_gac", perf_counter()
                with atomic():
                    counts["contractions"] += 1
                    if method == "expert":
                        for color, allowed in result.supported_domains.items():
                            state.restrict_domain(color, allowed)
                        if result.contradiction:
                            raise AssertionError("Unexpected root contradiction")
                        state.propagate()
                        work, maximum = result.work, result.max_size
                    else:
                        runtime.apply_candidate(state, candidate, prepared)
                        work, maximum = prepared[0].work, prepared[0].max_size
                    counts["apply_s"] += perf_counter()-phase_start
                    assert not state.contradiction
                    removed = sum(len(current[color])-len(state.domains[color]) for color in current)
                    counts["estimated_work"] += work
                    counts["max_tensor_size"] = max(counts["max_tensor_size"], maximum)
                    counts["removed_values"] += removed
                    if removed:
                        counts["useful"] += 1
                        before = len(snapshots)-1
                        checkpoint("useful_contraction")
                        proofs.append(dict(before=before, after=len(snapshots)-1, candidate_key=candidate.key, core_keys=candidate.core_keys,
                            open_colors=candidate.open_colors, removed=removed, estimated_work=work, max_size=maximum,
                            family=proposal[1].family if method == "expert" else None))
                        stream = None
                    else:
                        used[signature] = fingerprint
                    active_phase = None
    except TimeoutError:
        status = "resource_guard"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    wall_s, cpu_s = elapsed()
    if state.is_solved():
        status = "solved" if solved_s <= plan["seconds"] else "solved_after_deadline"
    elif method == "plain":
        status = "ordinary_gac_stalled"
    assert current == state.snapshot_domains() and not state.contradiction
    reference = read(plan["reference_assignments"][str(position)])["assignment"]
    assert _valid_assignment(network, input_domains, reference)
    assert all(reference[color] in values for snap in snapshots for color, values in snap["domains"].items())
    assignment = state.assignment() if state.is_solved() else {}
    valid = bool(assignment and _valid_assignment(network, input_domains, assignment))
    assert not state.is_solved() or valid
    assert set(state._cores) == original_keys and not state._added_core_keys and state.decision_level == 0
    assert not state._queue and not state._queued and state.on_closure is None
    prefixes = {}
    for budget in plan["budgets_s"]:
        eligible = [i for i, snap in enumerate(snapshots) if snap["wall_s"] <= budget]
        prefixes[str(budget)] = dict(snapshot=eligible[-1] if eligible else None,
            solved=bool(valid and solved_s is not None and solved_s <= budget))
    row = dict(method=method, position=position, protocol_sha256=digest(OUT / "protocol.json"), status=status,
        solved_s=solved_s, solved_cpu_s=next((s["cpu_s"] for s in snapshots if s["solved"]), None),
        valid=valid, assignment=assignment, wall_s=wall_s, cpu_s=cpu_s, baseline_s=baseline_s,
        prefixes=prefixes, snapshots=snapshots, proofs=proofs, totals=counts, structural_progress=progress,
        interrupted_phase=active_phase if status == "resource_guard" else None,
        decision_level=state.decision_level, no_added_cores=True, no_pending_queue=True,
        dfs_called=False, reference_retained=True,
        affinity=sorted(os.sched_getaffinity(0)))
    write(path, row)
    print(f"ROOT {method:9} {position:02d} {status:32} {wall_s:7.3f}s CPU{cpu_s:7.3f}s "
          f"useful{counts['useful']}/{counts['contractions']}", flush=True)
    return row


def residual(method, position):
    from experiments.cluster_selection.reinforcement.experiment import _valid_assignment
    from experiments.constraint_networks.mac import ConstraintNetworkSolver
    row_path = OUT / "rows" / f"{method}_{position:02d}.json"
    row = read(row_path)
    path = OUT / "rows" / f"{method}_{position:02d}.dfs.json"
    if path.exists():
        existing = read(path)
        assert existing["root_row_sha256"] == digest(row_path)
        return existing
    plan, network = CONTEXT["plan"], CONTEXT["network"]
    results, completed = {}, {}
    for budget in plan["budgets_s"]:
        index = row["prefixes"][str(budget)]["snapshot"]
        if index not in completed:
            domains = row["snapshots"][index]["domains"] if index is not None else CONTEXT["domains"][position]
            state = network.create_mac_state(domains)
            solver = ConstraintNetworkSolver(network, max_nodes=plan["residual_dfs"]["max_nodes"], variable_tie_break="name")
            previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Residual DFS guard")))
            started = perf_counter()
            try:
                signal.setitimer(signal.ITIMER_REAL, plan["residual_dfs"]["seconds"])
                result = solver.solve(state=state)
                valid = bool(result.solved and _valid_assignment(network, CONTEXT["domains"][position], result.assignment))
                assert not result.solved or valid
                completed[index] = dict(nodes=result.nodes if valid else None, solved=valid, valid=valid,
                    status="solved" if valid else "node_limit" if result.node_limit_reached else "unsat",
                    solve_s=perf_counter()-started, snapshot=index)
            except TimeoutError:
                completed[index] = dict(nodes=None, solved=False, valid=False, status="timeout", snapshot=index)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        results[str(budget)] = completed[index]
    record = dict(method=method, position=position, root_row_sha256=digest(row_path),
        protocol_sha256=digest(OUT / "protocol.json"), budgets=results, timed_separately=True)
    write(path, record)
    return record


def report():
    plan = read(OUT / "protocol.json")
    rows = [read(path) for path in sorted((OUT / "rows").glob("*.json")) if not path.name.endswith(".dfs.json")]
    assert len(rows) == 128 and {(row["method"], row["position"]) for row in rows} == set(map(tuple, plan["order"]))
    results = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        results[method] = dict(instances=16, solved_by_budget={str(budget): sum(row["prefixes"][str(budget)]["solved"]
            for row in group) for budget in plan["budgets_s"]}, wall_s=sum(row["wall_s"] for row in group),
            cpu_s=sum(row["cpu_s"] for row in group), statuses=dict(Counter(row["status"] for row in group)),
            solved_times={row["position"]: row["solved_s"] for row in group if row["valid"]})
    record = dict(protocol_sha256=digest(OUT / "protocol.json"), results=results,
        row_sha256={path.name: digest(path) for path in sorted((OUT / "rows").glob("*.json"))})
    write(OUT / "summary.json", record)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate", "residual", "report"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--positions", type=int, nargs="+")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "report":
        report()
    else:
        initialize()
        for method, position in CONTEXT["plan"]["order"]:
            if (args.method is None or method == args.method) and (args.positions is None or position in args.positions):
                (evaluate if args.phase == "evaluate" else residual)(method, position)
