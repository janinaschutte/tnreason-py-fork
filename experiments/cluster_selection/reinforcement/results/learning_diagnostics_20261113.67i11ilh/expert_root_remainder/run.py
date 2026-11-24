"""Diagnose the expert's one unsolved root, without any search decisions."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import signal
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
PREVIOUS = OUT.parent / "expert_supervised"
DATA = OUT.parent / "heldout/unique_test.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def timeout(*_):
    raise TimeoutError("600-second root-only guard")


def main():
    from demonstrations.sudoku.inferenceClusters import (
        RuleContext, RuleLimits, available_rules, contract_cluster,
    )
    from demonstrations.sudoku.inferenceClusters.projections import describe_complete_support
    from experiments.cluster_selection.reinforcement.experiment import instances, _valid_assignment
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    import experiments.sudoku_rule_clusters as expert
    from tnreason.engine.workload_to_numpy import NumpyCore

    prior = json.loads((PREVIOUS / "protocol.json").read_text())
    assert all(digest(p) == h for p, h in prior["source_sha256"].items())
    references = [DATA, PREVIOUS / "rows/expert51_08.json", PREVIOUS / "protocol.json"]
    limits = RuleLimits()
    expanded = RuleLimits(max_cores=2916, max_work=100_000_000,
                          max_planning_steps=100_000_000)
    protocol = dict(position=8, seconds_per_stage=600, default_limits=asdict(limits),
        complete_limits=asdict(expanded), families=list(available_rules()),
        references={str(p): digest(p) for p in references},
        source_sha256={**prior["source_sha256"], str(Path(__file__)): digest(__file__)},
        selection="Only expert51 root not solved on previously queried16; exploratory diagnostic",
        complete_output="First unresolved position by current domain size and variable name",
        algorithm="Root propagation only; state.assign raises; no solver or probe invoked. "
                  "Try full original-network unary support after default expert stalls. "
                  "Reference assignment is used exclusively in post-execution validation.")
    write(OUT / "protocol.json", protocol)
    data = json.loads(DATA.read_text())
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    domains = instances(data)[0][8]
    state = network.create_mac_state(domains)
    assignments = []

    def no_assign(*args):
        assignments.append(args)
        raise AssertionError("Branch assignment forbidden")

    state.assign = no_assign
    calls, skipped = [], Counter()

    def observed(context, cluster, **options):
        result = contract_cluster(context, cluster, **options)
        calls.append(dict(family=cluster.family, cores=len(cluster.core_keys),
            outputs=len(cluster.open_colors), status=result.status, reason=result.reason,
            work=result.work, max_size=result.max_size, planning_steps=result.planning_steps))
        if result.status != "executed":
            skipped[(cluster.family, result.reason)] += 1
        return result

    original_contract = expert.contract_cluster
    expert.contract_cluster = observed
    state.on_closure = policy = expert.ClusterClosure(network, protocol["families"], limits)
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    started = perf_counter()
    try:
        signal.setitimer(signal.ITIMER_REAL, 600)
        state.propagate()
        status = "solved" if state.is_solved() else "stalled"
    except TimeoutError:
        status = "wall_timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        expert.contract_cluster = original_contract
    baseline = dict(status=status, seconds=perf_counter()-started,
        unresolved_positions=sum(len(state.domains[c]) > 1 for c in data["reward_colors"]),
        domains=dict(state.domains), statistics=policy.statistics(), contractions=calls,
        skipped=[dict(family=f, reason=r, count=n) for (f, r), n in skipped.items()])
    write(OUT / "expert_root.json", baseline)
    print("Default root:", status, baseline["unresolved_positions"], baseline["seconds"], flush=True)
    state.on_closure = None
    completed, status = [], "stalled"
    started = perf_counter()
    try:
        signal.setitimer(signal.ITIMER_REAL, 600)
        for _ in range(81):
            unresolved = [c for c in data["reward_colors"] if len(state.domains[c]) > 1]
            if not unresolved:
                status = "solved"
                break
            output = min(unresolved, key=lambda c: (len(state.domains[c]), c))
            context = RuleContext(network, state.domains, num=3, limits=expanded)
            cluster = describe_complete_support(context, outputs=(output,))
            result = contract_cluster(context, cluster)
            row = dict(output=output, domain_before=state.domains[output], status=result.status,
                reason=result.reason, work=result.work, max_size=result.max_size,
                planning_steps=result.planning_steps, supported=result.supported_domains)
            completed.append(row)
            print("Complete support:", row, flush=True)
            if result.status != "executed":
                status = "skipped_budget"
                break
            before = tuple(state.domains.items())
            for color, allowed in result.supported_domains.items():
                state.restrict_domain(color, allowed)
            assert not result.contradiction
            state.propagate()
            if tuple(state.domains.items()) == before:
                status = "no_pruning"
                break
    except TimeoutError:
        status = "wall_timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    complete = dict(status=status, seconds=perf_counter()-started, contractions=completed,
        domains=dict(state.domains),
        unresolved_positions=sum(len(state.domains[c]) > 1 for c in data["reward_colors"]))
    write(OUT / "complete_support.json", complete)
    reference = json.loads((PREVIOUS / "rows/expert51_08.json").read_text())["assignment"]
    assert _valid_assignment(network, domains, reference)
    assert all(reference[c] in values for c, values in state.domains.items())
    assert all(reference[c] in values for c, values in baseline["domains"].items())
    assert not assignments and not state.decision_level and not state._added_core_keys
    assert state._cores.keys() == network.coresDict.keys()
    assert all(digest(p) == h for p, h in {**protocol["references"], **protocol["source_sha256"]}.items())
    if state.is_solved():
        assert _valid_assignment(network, domains, {c: next(iter(v)) for c, v in state.domains.items()})
    write(OUT / "audit.json", dict(branch_assignments=0, no_solver_invoked=True,
        no_added_cores=True, reference_retained=True, source_and_reference_hashes_unchanged=True,
        final_status=status, final_solved=state.is_solved(), protocol_sha256=digest(OUT / "protocol.json")))
    print("Finished:", status, complete["unresolved_positions"], complete["seconds"], flush=True)


if __name__ == "__main__":
    main()
