"""Root-only, teacher-first saturation diagnostic; no solver decisions or DFS."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import signal
import sys
from time import perf_counter

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
PARENT = BASE / "generic_portfolio"
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("root_first_frozen_portfolio", PARENT / "run.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def snapshot(state):
    return {color: list(values) for color, values in state.domains.items()}


def domain_hash(domains):
    return sha256(json.dumps(domains, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze():
    import torch
    prior = read(PARENT / "protocol.json")
    assert all(digest(p) == h for p, h in {**prior["source_sha256"], **prior["reference_sha256"]}.items())
    assert digest(prior["data"]) == prior["data_sha256"]
    assert all(digest(info["path"]) == info["sha256"] for info in prior["checkpoints"].values())
    plan = deepcopy(prior)
    plan.update(kind="root_only_teacher_first_saturation", workers=4, seconds=600,
        initial_phase=dict(source="teacher", slots=1024, steps=262144, pass_number=-1),
        sweep_sources=parent.SOURCES, sweep_phase=dict(slots=512, steps=131072), noop_patience=128,
        structural_list_cap=32, mask_anchors=32, decisions=0, total_work=None)
    plan.pop("methods")
    plan["order"] = list(range(16))
    Random(2026111304).shuffle(plan["order"])
    plan["source_sha256"][str(Path(__file__))] = digest(__file__)
    refs = [PARENT / name for name in ("protocol.json", "summary.json", "audit.json")]
    refs += [PARENT / "structural_cap32" / name for name in ("run.py", "protocol.json", "summary.json", "audit.json")]
    refs += [BASE / "expert_supervised" / name for name in ("protocol.json", "comparison.json", "audit.json")]
    plan["reference_sha256"].update({str(p): digest(p) for p in refs})
    plan["note"] = ("Exploratory saturation on the same reused16 puzzles. Ordinary GAC then teacher1024 first, "
        "then repeat complete six-source passes while domains shrink, in frozen source order. Each source phase retains all "
        "earlier pruning. Stop at solution, contradiction, full six-source pass without domain reduction, "
        "or600s per puzzle; no domain assumptions, assign, solve_mac or DFS. Per-source contraction and "
        "decoder limits are guards, not equal-compute claims. All decoders regenerate after every useful "
        "contraction. Shared no-op cache is keyed by canonical cluster plus current domains on its scope; "
        "rotate source after128 consecutive actual affordable contraction+GAC no-ops; reset streak after any "
        "domain reduction, including auxiliary variables. Skips/rejections do not count as no-op contractions. "
        "1024/512contraction limits are emergency source-rotation guards, not a stop for the entire puzzle. "
        "Structural traversal always has32proposal-list breadth, mask tries32anchors. Core64/output6/work1e6/size1e5 "
        "and compact domains match earlier generic bounds. Immutable checkpoints after baseline, every useful "
        "contraction and every phase preserve domains and proofs. Interrupted workers resume the current phase "
        "from their last checkpoint with fresh no-op cache; any unrecorded no-op work is unknown, never lost pruning.")
    write(OUT / "protocol.json", plan)
    (OUT / "cases").mkdir()
    print("Frozen16 root-only cases; no propagation/model queries before freeze", flush=True)


def phase_settings(plan, number):
    if number == 0:
        return plan["initial_phase"]
    sweep, offset = divmod(number-1, len(plan["sweep_sources"]))
    return dict(plan["sweep_phase"], source=plan["sweep_sources"][offset], pass_number=sweep)


def initialize():
    import numpy as np
    import torch
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection.reinforcement.experiment import instances
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in {**plan["source_sha256"], **plan["reference_sha256"]}.items())
    assert digest(plan["data"]) == plan["data_sha256"]
    models = {}
    for name, info in plan["checkpoints"].items():
        assert digest(info["path"]) == info["sha256"]
        models[name] = parent.runtime.load(info["path"]).cpu().eval()
        models[name].config["compact_domains"] = True
        if name == "mask":
            models[name].config["anchor_candidates"] = plan["mask_anchors"]
    data = read(plan["data"])
    domains, _ = instances(data)
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("DFS is forbidden in this root-only diagnostic")
    network.solve_mac = forbidden
    CONTEXT.update(plan=plan, models=models, network=network, domains=domains)


def evaluate(position):
    import numpy as np
    import torch
    from experiments.cluster_selection.reinforcement.experiment import _valid_assignment
    plan, models, network = CONTEXT["plan"], CONTEXT["models"], CONTEXT["network"]
    case = OUT / "cases" / f"{position:02d}"
    case.mkdir(exist_ok=True)
    row_path = case / "row.json"
    if row_path.exists():
        row = read(row_path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    checkpoints = sorted(case.glob("checkpoint_*.json"))
    resumed = read(checkpoints[-1]) if checkpoints else None
    state = network.create_mac_state(resumed["domains"] if resumed else CONTEXT["domains"][position])
    original_cores = set(state._cores)
    state.assign = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Assignments forbidden"))
    state.propagate()
    assert state.on_closure is None and state.decision_level == 0 and not state._queue
    begin_phase = resumed["next_phase"] if resumed else 0
    phases = deepcopy(resumed["phases"]) if resumed else []
    totals = defaultdict(Counter, {s: Counter(v) for s, v in resumed["totals"].items()}) if resumed else defaultdict(Counter)
    checkpoint_number = len(checkpoints)
    started = perf_counter()
    useful_traces = list(resumed["useful_traces"]) if resumed else []
    used, pass_start = {}, dict(resumed.get("pass_start", {})) if resumed else {}
    status, current, phase_number, noop_streak, attempts = "stalled", None, begin_phase, 0, 0
    original_contract = parent.runtime.contract_candidate
    def measured(candidate, prepared):
        estimate = prepared[0]
        totals[current["source"]].update(contractions=1, estimated_work=estimate.work)
        totals[current["source"]]["max_tensor_size"] = max(totals[current["source"]]["max_tensor_size"], estimate.max_size)
        return original_contract(candidate, prepared)
    def checkpoint(next_phase, reason):
        nonlocal checkpoint_number
        assert set(state._cores) == original_cores and state.decision_level == 0 and state.on_closure is None
        value = dict(position=position, protocol_sha256=digest(OUT / "protocol.json"), reason=reason,
            next_phase=next_phase, domains=snapshot(state), domain_hash=domain_hash(snapshot(state)),
            phases=phases, totals={s: dict(v) for s, v in totals.items()}, useful_traces=useful_traces,
            pass_start=pass_start, elapsed_s=perf_counter()-started, resumed=bool(resumed))
        write(case / f"checkpoint_{checkpoint_number:05d}.json", value)
        checkpoint_number += 1
    if not resumed:
        checkpoint(0, "ordinary_gac")
    else:
        assert snapshot(state) == resumed["domains"]
    def timeout(*args):
        raise TimeoutError("Root saturation wall guard")
    previous_signal = signal.signal(signal.SIGALRM, timeout)
    parent.runtime.contract_candidate = measured
    teacher_solved = any(p["source"] == "teacher" and p["pass_number"] == -1 and p["solved"] for p in phases)
    try:
        signal.setitimer(signal.ITIMER_REAL, plan["seconds"])
        with torch.no_grad():
            while True:
                if state.is_solved() or state.contradiction:
                    break
                current = phase_settings(plan, phase_number)
                source = current["source"]
                if source == "heuristic":
                    model = parent.Structural(models["teacher"], plan["structural_list_cap"])
                else:
                    model = models[source]
                if current["pass_number"] >= 0 and source == parent.SOURCES[0]:
                    pass_start[str(current["pass_number"])] = sum(len(v) for v in state.domains.values())
                before = snapshot(state)
                before_totals = Counter(totals[source])
                phase_started, attempts, stream, noop_streak, reason = perf_counter(), 0, None, 0, "contraction_guard"
                while not state.is_solved() and not state.contradiction and attempts < current["slots"]:
                    if stream is None:
                        totals[source]["refreshes"] += 1
                        stream = iter(model.proposals(state.domains, max_candidates=8192, max_steps=current["steps"]))
                    candidate = next(stream, None)
                    if candidate is None:
                        reason = "decoder_exhausted"
                        break
                    totals[source]["candidates"] += 1
                    signature = tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))
                    scope = sorted({c for key in candidate.core_keys for c in state._cores[key].colors})
                    fingerprint = tuple(state.domains[c] for c in scope)
                    if used.get(signature) == fingerprint:
                        totals[source]["cache_skips"] += 1
                        continue
                    prepared = parent.runtime._prepare(state, candidate, compact_domains=True,
                        max_work=1_000_000, max_size=100_000)
                    if prepared is None:
                        totals[source]["rejected"] += 1
                        continue
                    assert len(candidate.core_keys) <= 64 and len(candidate.open_colors) <= 6
                    input_domains = snapshot(state)
                    result = parent.runtime.apply_candidate(state, candidate, prepared)
                    attempts += 1
                    totals[source]["completed"] += 1
                    totals[source]["removed_values"] += result.removed_values
                    useful = bool(result.removed_values or result.contradiction)
                    totals[source]["useful"] += int(useful)
                    assert state.decision_level == 0 and set(state._cores) == original_cores and not state._queue
                    if useful:
                        noop_streak = 0
                        output_domains = snapshot(state)
                        trace = dict(source=source, phase_number=phase_number, core_keys=candidate.core_keys,
                            open_colors=candidate.open_colors, before=domain_hash(input_domains),
                            after=domain_hash(output_domains), removed=result.removed_values,
                            contradiction=result.contradiction, domains=output_domains,
                            estimated_work=prepared[0].work, max_tensor_size=prepared[0].max_size)
                        useful_traces.append(trace)
                        checkpoint(phase_number, "useful_contraction")
                        stream = None
                    else:
                        used[signature] = fingerprint
                        noop_streak += 1
                        if noop_streak >= plan["noop_patience"]:
                            reason = "noop_patience"
                            break
                after = snapshot(state)
                phase_row = dict(current, phase_number=phase_number, elapsed_s=perf_counter()-phase_started,
                    before=domain_hash(before), after=domain_hash(after), solved=state.is_solved(),
                    contradiction=state.contradiction, remaining_values=sum(len(v) for v in after.values()),
                    stop_reason="solved" if state.is_solved() else "contradiction" if state.contradiction else reason,
                    noop_streak=noop_streak, counts={k: v for k, v in
                        (Counter(totals[source])-before_totals).items() if k != "max_tensor_size"})
                phases.append(phase_row)
                if phase_number == 0:
                    teacher_solved = state.is_solved()
                checkpoint(phase_number+1, "phase_complete")
                print(position, phase_number, source, "solved" if state.is_solved() else "stalled",
                    phase_row["counts"], f"{phase_row['elapsed_s']:.2f}s", flush=True)
                if source == parent.SOURCES[-1] and current["pass_number"] >= 0:
                    if sum(len(v) for v in state.domains.values()) == pass_start[str(current["pass_number"])]:
                        status = "full_pass_stalled"
                        break
                phase_number += 1
    except TimeoutError:
        status = "wall_timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_signal)
        parent.runtime.contract_candidate = original_contract
    if state.is_solved():
        status = "solved"
    elif state.contradiction:
        status = "contradiction"
    checkpoint(len(phases), status)
    domains = snapshot(state)
    assignment = state.assignment() if state.is_solved() else {}
    valid = bool(state.is_solved() and _valid_assignment(network, CONTEXT["domains"][position], assignment))
    assert not state.is_solved() or valid
    colors = models["teacher"].reward_colors
    entropy = float(sum(np.log(len(domains[c])) for c in colors)) if all(domains[c] for c in colors) else None
    row = dict(position=position, protocol_sha256=digest(OUT / "protocol.json"), status=status,
        solved=state.is_solved(), valid=valid, assignment=assignment, domains=domains,
        domain_hash=domain_hash(domains), decisions=0, dfs_called=False, teacher_solved=teacher_solved,
        phases=phases, totals={s: dict(v) for s, v in totals.items()}, useful_traces=useful_traces,
        solve_s=perf_counter()-started, remaining_entropy=entropy, mac_statistics=asdict(state.statistics),
        initial_domain_hash=domain_hash(CONTEXT["domains"][position]), resumed=bool(resumed))
    write(row_path, row)
    print("ROOT", position, status, "teacher", teacher_solved, f"{row['solve_s']:.2f}s", flush=True)
    return row


def report():
    paths = sorted((OUT / "cases").glob("*/row.json"))
    rows = [read(p) for p in paths]
    assert len(rows) == 16 and {r["position"] for r in rows} == set(range(16))
    totals = defaultdict(Counter)
    for row in rows:
        for source, counts in row["totals"].items():
            totals[source].update({k: v for k, v in counts.items() if k != "max_tensor_size"})
            totals[source]["max_tensor_size"] = max(totals[source]["max_tensor_size"], counts.get("max_tensor_size", 0))
    result = dict(protocol_sha256=digest(OUT / "protocol.json"), puzzles=16,
        teacher_solved=sum(r["teacher_solved"] for r in rows),
        teacher_root_positions=[r["position"] for r in rows if r["teacher_solved"]],
        combined_solved=sum(r["solved"] and r["valid"] for r in rows),
        combined_root_positions=[r["position"] for r in rows if r["solved"] and r["valid"]],
        solve_seconds=sum(r["solve_s"] for r in rows), statuses=dict(Counter(r["status"] for r in rows)),
        unresolved=[r["position"] for r in rows if not r["solved"]],
        source_statistics={s: dict(v) for s, v in totals.items()},
        row_sha256={str(p.relative_to(OUT)): digest(p) for p in paths})
    write(OUT / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate", "report"))
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    elif phase == "evaluate":
        plan = read(OUT / "protocol.json")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"), initializer=initialize) as pool:
            for job in as_completed([pool.submit(evaluate, position) for position in plan["order"]]):
                job.result()
    else:
        report()
