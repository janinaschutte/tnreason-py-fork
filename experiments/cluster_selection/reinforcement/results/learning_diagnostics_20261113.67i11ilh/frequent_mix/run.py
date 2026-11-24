"""Frequently interleave generic cluster constructors, without decisions or DFS."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from multiprocessing import get_context
import os
from pathlib import Path
from random import Random
import signal
import sys
from time import perf_counter
from uuid import uuid4

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
PARENT = BASE / "generic_portfolio"
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("frequent_mix_frozen_helpers", PARENT / "run.py")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
SOURCES = helper.SOURCES
VARIANTS = {"all": SOURCES, **{f"without_{s}": tuple(t for t in SOURCES if t != s) for s in SOURCES}}
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def domain_hash(domains):
    return sha256(json.dumps(domains, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def blocked_alarm():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def write(path, value):
    """Publish a complete immutable JSON atomically; hard links never overwrite."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with blocked_alarm():
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
    prior = read(PARENT / "protocol.json")
    assert all(digest(p) == h for p, h in {**prior["source_sha256"], **prior["reference_sha256"]}.items())
    assert digest(prior["data"]) == prior["data_sha256"]
    assert all(digest(info["path"]) == info["sha256"] for info in prior["checkpoints"].values())
    assert prior["checkpoints"]["q"]["config"].get("decoder", "depth_first") == "depth_first"
    plan = deepcopy(prior)
    plan.pop("methods")
    plan.update(kind="frequent_generic_root_interleaving", variants=VARIANTS, workers=4,
        seconds=600, decoder_steps=131072, decoder_candidates=8192, mask_anchors=32,
        structural_stages=[[32, 8192], [64, 16384], [128, 32768], [256, 65536], [512, 131072]],
        structural_worst_stage_steps=253952, source_order=SOURCES, decisions=0, total_work=None)
    plan["order"] = list(range(16))
    Random(2026111305).shuffle(plan["order"])
    plan["source_sha256"][str(Path(__file__))] = digest(__file__)
    refs = [PARENT / name for name in ("protocol.json", "summary.json", "audit.json")]
    refs += [BASE / "root_first" / name for name in ("protocol.json", "summary.json", "audit.json")]
    refs += [BASE / "expert_root_remainder/root_benchmark" / name for name in ("protocol.json", "summary.json", "audit.json")]
    plan["reference_sha256"].update({str(p): digest(p) for p in refs})
    plan["note"] = ("Same reused16 puzzles; main and six leave-one-source-out variants fixed before queries. "
        "No teacher-first block, cluster allowance or no-op patience. Lazy persistent source iterators; "
        "exactly one emitted candidate is examined per source turn, even on cache hit/rejection; advance "
        "cursor before applying. Every domain reduction invalidates all streams/exhaustion flags, preserving "
        "next-source cursor and exact scoped no-op cache. Stop solved/contradiction, all bounded iterators "
        "exhausted since last pruning, or600s after initial ordinary GAC. Learned stream131072construction "
        "states/8192emissions per domain generation; Q retains saved depth_first decoder. Structural stages "
        "32/64/128/256/512 retain canonical prefixes, stopping each stage at its cumulative unique quota; "
        "worst traversal sum253952steps. Mask32anchors. One next() may have unequal time; eager masks and "
        "structural stages run only on their own turn. No equal wall/compute claim. SIGALRM active during "
        "proposals/planning, blocked across the whole contraction/GAC/counter/proof/checkpoint transaction; "
        "pending timeout is delivered after coherent publication, so guard can overshoot by one bounded "
        "contraction plus checkpoint I/O. Atomic hard-link JSON publication preflighted; no overwrite. "
        "Time excludes model/network/state loading, initial GAC and final assignment verification, includes "
        "active-loop checkpoint I/O. No DFS/assign/assumptions. Existing per-cluster64cores/6outputs/1e6work/1e5size "
        "with compact domains, original cores only, temporary summaries removed immediately.")
    write(OUT / "publication_preflight.json", dict(atomic_exclusive_hardlink=True))
    try:
        write(OUT / "publication_preflight.json", dict(must_not_overwrite=True))
        raise AssertionError("Existing destination was overwritten")
    except FileExistsError:
        pass
    write(OUT / "protocol.json", plan)
    (OUT / "cases").mkdir()
    print("FROZEN main+six leave-one-out variants; no propagation/model queries", flush=True)


def initialize():
    helper.OUT = OUT
    helper.initialize()
    CONTEXT.update(helper.CONTEXT)
    CONTEXT["models"]["mask"].config["anchor_candidates"] = 32
    def forbidden(*args, **kwargs):
        raise AssertionError("DFS/assignment forbidden in root-only inference")
    CONTEXT["network"].solve_mac = forbidden
    CONTEXT["forbidden"] = forbidden


def structural_stream(domains, info):
    seen = set()
    for quota, steps in CONTEXT["plan"]["structural_stages"]:
        info["stages_started"].append(dict(quota=quota, steps=steps))
        model = helper.Structural(CONTEXT["models"]["teacher"], quota)
        for candidate in model.proposals(domains, max_candidates=quota, max_steps=steps):
            signature = tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))
            if signature in seen:
                continue
            seen.add(signature)
            yield candidate
            if len(seen) >= quota:
                break


def evaluate(task):
    import numpy as np
    import torch
    from experiments.cluster_selection.reinforcement.experiment import _valid_assignment
    variant, position = task
    plan, models, network = CONTEXT["plan"], CONTEXT["models"], CONTEXT["network"]
    sources = plan["variants"][variant]
    case = OUT / "cases" / variant / f"{position:02d}"
    case.mkdir(parents=True, exist_ok=True)
    row_path = case / "row.json"
    if row_path.exists():
        row = read(row_path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    if list(case.glob("checkpoint_*.json")):
        raise RuntimeError(f"Partial case preserved; explicit reviewed recovery required: {case}")
    state = network.create_mac_state(CONTEXT["domains"][position])
    state.assign = CONTEXT["forbidden"]
    state.propagate()
    original_cores = set(state._cores)
    assert not state.contradiction and not state._queue and state.decision_level == 0
    colors = models["teacher"].reward_colors
    current_domains = state.snapshot_domains()
    baseline_domains = current_domains
    current_hash = domain_hash(current_domains)
    baseline_hash = current_hash
    entropy = lambda ds: float(sum(np.log(len(ds[c])) for c in colors)) if all(ds[c] for c in colors) else None
    baseline_entropy = entropy(current_domains)
    statistics = {s: Counter() for s in sources}
    sequence, traces, stage_progress = [], [], []
    streams, exhausted, used = {}, set(), {}
    generation, cursor, checkpoint_number, turn = 0, 0, 0, 0
    finishing_source, status, resource_reached, active_event = None, "bounded_iterators_exhausted", False, None
    started = perf_counter()
    def checkpoint(reason):
        nonlocal checkpoint_number
        assert set(state._cores) == original_cores and not state._queue and state.decision_level == 0 and state.on_closure is None
        write(case / f"checkpoint_{checkpoint_number:05d}.json", dict(reason=reason, variant=variant,
            position=position, protocol_sha256=digest(OUT / "protocol.json"), generation=generation,
            cursor=cursor, domains=current_domains, domain_hash=current_hash, totals=statistics,
            sequence=sequence, useful_traces=traces, exhausted_sources=sorted(exhausted),
            structural_progress=stage_progress, elapsed_s=perf_counter()-started))
        checkpoint_number += 1
    checkpoint("ordinary_gac")
    def timeout(*args):
        raise TimeoutError("Post-GAC root resource guard")
    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        remaining = plan["seconds"]-(perf_counter()-started)
        if remaining <= 0:
            raise TimeoutError("Baseline checkpoint consumed root resource guard")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        with torch.no_grad():
            while not state.is_solved() and not state.contradiction and len(exhausted) < len(sources):
                source = sources[cursor]
                cursor = (cursor+1) % len(sources)
                if source in exhausted:
                    continue
                turn += 1
                counts = statistics[source]
                event = dict(turn=turn, source=source, generation=generation, before=current_hash)
                active_event = dict(event, stage="proposal")
                counts["turns"] += 1
                counts["proposal_calls_started"] += 1
                proposal_started = perf_counter()
                proposal_finished = False
                try:
                    if source not in streams:
                        counts["streams_created"] += 1
                        if source == "heuristic":
                            info = dict(generation=generation, stages_started=[])
                            stage_progress.append(info)
                            streams[source] = iter(structural_stream(current_domains, info))
                        else:
                            streams[source] = iter(models[source].proposals(current_domains,
                                max_candidates=plan["decoder_candidates"], max_steps=plan["decoder_steps"]))
                    candidate = next(streams[source], None)
                    proposal_finished = True
                finally:
                    with blocked_alarm():
                        counts["proposal_s"] += perf_counter()-proposal_started
                        counts["proposal_calls_completed"] += int(proposal_finished)
                if candidate is None:
                    with blocked_alarm():
                        exhausted.add(source)
                        counts["bounded_iterator_exhaustions"] += 1
                        sequence.append(dict(event, kind="bounded_iterator_exhausted", after=current_hash))
                        active_event = None
                    continue
                with blocked_alarm():
                    counts["proposals"] += 1
                    active_event["stage"] = "candidate_screening"
                signature = tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))
                event["candidate_signature"] = sha256(repr(signature).encode()).hexdigest()
                scope = sorted({c for key in candidate.core_keys for c in state._cores[key].colors})
                fingerprint = tuple(state.domains[c] for c in scope)
                if used.get(signature) == fingerprint:
                    with blocked_alarm():
                        counts["cache_skips"] += 1
                        sequence.append(dict(event, kind="cache_hit", after=current_hash))
                        active_event = None
                    continue
                active_event["stage"] = "prepare"
                counts["prepare_calls_started"] += 1
                prepare_started = perf_counter()
                prepare_finished = False
                try:
                    prepared = helper.runtime._prepare(state, candidate, compact_domains=True,
                        max_work=1_000_000, max_size=100_000)
                    prepare_finished = True
                finally:
                    with blocked_alarm():
                        counts["prepare_s"] += perf_counter()-prepare_started
                        counts["prepare_calls_completed"] += int(prepare_finished)
                if prepared is None:
                    with blocked_alarm():
                        counts["rejected"] += 1
                        sequence.append(dict(event, kind="unaffordable", after=current_hash))
                        active_event = None
                    continue
                assert len(candidate.core_keys) <= 64 and len(candidate.open_colors) <= 6
                with blocked_alarm():
                    counts["contractions_started"] += 1
                    counts["estimated_work"] += prepared[0].work
                    counts["max_tensor_size"] = max(counts["max_tensor_size"], prepared[0].max_size)
                    apply_started = perf_counter()
                    result = helper.runtime.apply_candidate(state, candidate, prepared)
                    counts["apply_s"] += perf_counter()-apply_started
                    counts["contractions_completed"] += 1
                    after_domains = state.snapshot_domains()
                    removed = sum(len(current_domains[c])-len(after_domains[c]) for c in current_domains)
                    primary_removed = sum(len(current_domains[c])-len(after_domains[c]) for c in colors)
                    assert removed == result.removed_values
                    useful = bool(removed or result.contradiction)
                    counts["removed_all"] += removed
                    counts["removed_primary"] += primary_removed
                    counts["useful"] += int(useful)
                    counts["noops"] += int(not useful)
                    before_entropy, after_entropy = entropy(current_domains), entropy(after_domains)
                    gain = before_entropy-after_entropy if after_entropy is not None else None
                    if gain is not None:
                        counts["entropy_gain"] += gain
                    if useful:
                        after_hash = domain_hash(after_domains)
                        traces.append(dict(event, core_keys=candidate.core_keys, open_colors=candidate.open_colors,
                            after=after_hash, domains=after_domains, removed_all=removed,
                            removed_primary=primary_removed, entropy_gain=gain, contradiction=result.contradiction,
                            estimated_work=prepared[0].work, max_tensor_size=prepared[0].max_size))
                        sequence.append(dict(event, kind="useful", after=after_hash, trace_index=len(traces)-1))
                        current_domains, current_hash = after_domains, after_hash
                        generation += 1
                        streams, exhausted = {}, set()
                        if state.is_solved():
                            finishing_source = source
                        active_event = None
                        checkpoint_started = perf_counter()
                        checkpoint("useful_contraction")
                        counts["checkpoint_s"] += perf_counter()-checkpoint_started
                    else:
                        used[signature] = fingerprint
                        sequence.append(dict(event, kind="noop", after=current_hash))
                        active_event = None
                    assert set(state._cores) == original_cores and not state._queue and state.decision_level == 0
    except TimeoutError:
        status, resource_reached = "resource_guard", True
        if active_event is not None:
            sequence.append(dict(active_event, kind="resource_guard_interrupted_turn", after=current_hash))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    if state.is_solved():
        status = "solved"
    elif state.contradiction:
        status = "contradiction"
    elapsed = perf_counter()-started
    assignment = state.assignment() if state.is_solved() else {}
    valid = bool(state.is_solved() and _valid_assignment(network, CONTEXT["domains"][position], assignment))
    assert not state.is_solved() or valid
    assert current_domains == state.snapshot_domains()
    checkpoint(status)
    row = dict(variant=variant, position=position, protocol_sha256=digest(OUT / "protocol.json"),
        status=status, solved=state.is_solved(), valid=valid, assignment=assignment, finishing_source=finishing_source,
        domains=current_domains, domain_hash=current_hash, baseline_domain_hash=baseline_hash,
        baseline_domains=baseline_domains, baseline_entropy=baseline_entropy, remaining_entropy=entropy(current_domains),
        sequence=sequence, useful_traces=traces, totals=statistics, generations=generation,
        structural_progress=stage_progress, final_exhausted_sources=sorted(exhausted),
        final_created_sources=list(streams), solve_s=elapsed, resource_guard_reached=resource_reached,
        decisions=0, dfs_called=False, next_source=sources[cursor],
        timing_note="Post-GAC; includes active-loop checkpoint I/O, excludes model/state loading and final validation/publication.")
    write(row_path, row)
    print("ROOT", variant, position, status, "last", finishing_source, f"{elapsed:.2f}s",
        "contractions", sum(c["contractions_completed"] for c in statistics.values()), flush=True)
    return row


def report(variant):
    paths = sorted((OUT / "cases" / variant).glob("*/row.json"))
    rows = [read(path) for path in paths]
    assert len(rows) == 16 and {r["position"] for r in rows} == set(range(16))
    result = dict(variant=variant, protocol_sha256=digest(OUT / "protocol.json"),
        puzzles=16, solved=sum(r["solved"] and r["valid"] for r in rows),
        root_positions=[r["position"] for r in rows if r["solved"] and r["valid"]],
        statuses=dict(Counter(r["status"] for r in rows)), solve_seconds=sum(r["solve_s"] for r in rows),
        unfinished=[r["position"] for r in rows if not r["solved"]],
        row_sha256={str(p.relative_to(OUT)): digest(p) for p in paths})
    write(OUT / f"summary_{variant}.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate", "report"))
    parser.add_argument("--variant", choices=VARIANTS, default="all")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "evaluate":
        plan = read(OUT / "protocol.json")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"), initializer=initialize) as pool:
            for job in as_completed([pool.submit(evaluate, (args.variant, position)) for position in plan["order"]]):
                job.result()
    else:
        report(args.variant)
