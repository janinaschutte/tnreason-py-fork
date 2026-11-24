"""Root-only expert timing after ordinary GAC, with every search entry forbidden."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import signal
from time import perf_counter

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
DATA = BASE / "heldout/unique_test.json"
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def forbid(*args, **kwargs):
    raise AssertionError("DFS/search/assignment forbidden in root-only benchmark")


def freeze():
    from demonstrations.sudoku.inferenceClusters import RuleLimits, available_rules
    prior = read(BASE / "expert_supervised/protocol.json")
    assert all(digest(p) == h for p, h in prior["source_sha256"].items())
    references = [DATA, BASE / "expert_supervised/protocol.json"]
    references += [BASE / f"expert_supervised/rows/expert51_{i:02d}.json" for i in range(16)]
    plan = dict(data=str(DATA), positions=list(range(16)), seconds=600, workers=4,
        families=list(available_rules()), limits=asdict(RuleLimits()),
        source_sha256={**prior["source_sha256"], str(Path(__file__)): digest(__file__)},
        references={str(p): digest(p) for p in references},
        timing="Ordinary GAC timed separately; expert timer begins after ordinary closure and "
               "includes repeated exact cluster projection/GAC only. Network setup/loading and validation excluded.",
        forbidden="state.assign and network.solve_mac both raise; no solver or probe called",
        interpretation="Exploratory root-only reference on the same already queried 16 unique puzzles")
    write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir()
    print("Frozen 16 root-only expert cases", flush=True)


def initialize():
    import numpy as np
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from tnreason.engine.workload_to_numpy import NumpyCore
    plan = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in {**plan["source_sha256"], **plan["references"]}.items())
    data = read(plan["data"])
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = forbid
    CONTEXT.update(plan=plan, network=network, domains=instances(data)[0], data=data)


def evaluate(position):
    from demonstrations.sudoku.inferenceClusters import RuleLimits
    from experiments.cluster_selection.reinforcement.experiment import _valid_assignment
    from experiments.sudoku_rule_clusters import ClusterClosure
    path = OUT / "rows" / f"{position:02d}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    plan, network, domains = CONTEXT["plan"], CONTEXT["network"], CONTEXT["domains"][position]
    state = network.create_mac_state(domains)
    state.assign = forbid
    baseline_seconds, expert_started, policy = None, None, None

    def timeout(*_):
        raise TimeoutError("600-second case guard")

    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, plan["seconds"])
        started = perf_counter()
        state.propagate()
        baseline_seconds = perf_counter() - started
        baseline_domains = dict(state.domains)
        baseline_solved = state.is_solved()
        state.on_closure = policy = ClusterClosure(network, plan["families"], RuleLimits(**plan["limits"]))
        expert_started = perf_counter()
        state.propagate()
        expert_seconds = perf_counter() - expert_started
        status = "contradiction" if state.contradiction else "solved" if state.is_solved() else "stalled"
    except TimeoutError:
        expert_seconds = perf_counter() - expert_started if expert_started is not None else None
        status = "wall_timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert status != "contradiction"
    reference = read(BASE / f"expert_supervised/rows/expert51_{position:02d}.json")["assignment"]
    assert _valid_assignment(network, domains, reference)
    assert all(reference[c] in values for c, values in state.domains.items())
    assert all(reference[c] in values for c, values in baseline_domains.items())
    assignment = {c: next(iter(v)) for c, v in state.domains.items()} if state.is_solved() else {}
    if assignment:
        assert _valid_assignment(network, domains, assignment)
    assert not state.decision_level and not state._added_core_keys
    assert state._cores.keys() == network.coresDict.keys()
    row = dict(position=position, status=status, branch_free=status == "solved",
        baseline_phase_seconds=baseline_seconds, expert_phase_seconds=expert_seconds,
        baseline_solved=baseline_solved, baseline_domains=baseline_domains, domains=dict(state.domains),
        unresolved_positions=sum(len(state.domains[c]) > 1 for c in CONTEXT["data"]["reward_colors"]),
        assignment=assignment, cluster_statistics=policy.statistics() if policy is not None else {},
        mac_statistics=asdict(state.statistics), valid_reference_retained=True,
        no_added_cores=True, decision_level=0, protocol_sha256=digest(OUT / "protocol.json"))
    write(path, row)
    print(position, status, row["unresolved_positions"], expert_seconds, flush=True)
    return row


def main(phase):
    if phase == "freeze":
        freeze()
        return
    plan = read(OUT / "protocol.json")
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"), initializer=initialize) as pool:
        for result in as_completed([pool.submit(evaluate, i) for i in plan["positions"]]):
            result.result()
    rows = [read(OUT / "rows" / f"{i:02d}.json") for i in plan["positions"]]
    assert all(digest(p) == h for p, h in {**plan["source_sha256"], **plan["references"]}.items())
    summary = dict(instances=16, branch_free=sum(r["branch_free"] for r in rows),
        status_counts={s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        baseline_phase_seconds=sum(r["baseline_phase_seconds"] for r in rows),
        expert_phase_seconds=sum(r["expert_phase_seconds"] for r in rows),
        unresolved_positions={r["position"]: r["unresolved_positions"] for r in rows if not r["branch_free"]},
        protocol_sha256=digest(OUT / "protocol.json"),
        row_sha256={p.name: digest(p) for p in sorted((OUT / "rows").glob("*.json"))})
    write(OUT / "summary.json", summary)
    write(OUT / "audit.json", dict(rows=16, original_assignments_and_givens_checked=True,
        no_dfs_or_assign=True, no_added_cores=True, no_training=True,
        reference_retained_every_domain=True, frozen_hashes_unchanged=True,
        timing_excludes_network_loading_and_validation=True))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "evaluate"])
    main(parser.parse_args().phase)
