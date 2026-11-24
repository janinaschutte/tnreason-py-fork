"""Independently replay every useful new-rule contraction, without search."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_mac import prepare_candidate
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.reinforcement.experiment import instances, _valid_assignment
from experiments.cluster_selection.runtime import apply_candidate

OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def normalized(domains):
    return {c: tuple(values) for c, values in domains.items()}


def forbidden(*args, **kwargs):
    raise AssertionError("No constructor, assignment or solver may run during proof replay")


def main():
    plan = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in plan["source_sha256"].items())
    assert digest(plan["data"]) == plan["data_sha256"]
    data = read(plan["data"])
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = forbidden
    domains = instances(data)[0]
    paths = sorted((OUT / "rows").glob("*.json"))
    rows = [read(p) for p in paths]
    assert len(rows) == 80 and {(r["method"], r["position"]) for r in rows} == set(map(tuple, plan["order"]))
    checked, traces, primary_removed, contradiction_queues = [], [], 0, 0
    for row in rows:
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        if row["solved"]:
            assert row["valid"] and row["status"] == "solved"
            assert _valid_assignment(network, domains[row["position"]], row["assignment"])
            checked.append((row["method"], row["position"]))
        else:
            assert row["status"] in ("wall_timeout", "node_limit", "partial")
        counts = row["application_counts"]
        expected = sum(v["useful"] for k, v in counts.items() if k in ("overlap", "hall"))
        assert len(row["useful_rule_traces"]) == expected
        assert row["method"] == "cycles8_rules" or not row["useful_rule_traces"]
        for index, trace in enumerate(row["useful_rule_traces"]):
            before, after = normalized(trace["before"]), normalized(trace["after"])
            state = network.create_mac_state(before)
            state.assign = forbidden
            assert state.propagate() and state.snapshot_domains() == before
            assert state.on_closure is None and state.decision_level == 0
            candidate = SummaryClusterCandidate(f"replay_{row['position']}_{index}",
                                                tuple(trace["core_keys"]), tuple(trace["open_colors"]))
            assert len(candidate.core_keys) <= plan["limits"]["max_cores"]
            assert len(candidate.open_colors) <= plan["limits"]["max_open_colors"]
            prepared = prepare_candidate(state, candidate, compact_domains=True,
                max_work=plan["limits"]["max_work"], max_size=plan["limits"]["max_size"])
            assert prepared[0].work == trace["estimated_work"]
            assert prepared[0].max_size == trace["max_tensor_size"]
            result = apply_candidate(state, candidate, prepared)
            assert state.snapshot_domains() == after
            assert result.removed_values == trace["removed"]
            assert result.contradiction == trace["contradiction"]
            assert set(state._cores) == set(network.coresDict) and not state._added_core_keys
            assert state.on_closure is None and state.decision_level == 0
            assert set(state._queue) == state._queued and state._queued <= network.coresDict.keys()
            if state.contradiction:
                contradiction_queues += int(bool(state._queue))
            else:
                assert not state._queue
            for color, values in state.domains.items():
                expected_mask = np.zeros(state.dimensions[color], dtype=bool)
                expected_mask[list(values)] = True
                assert np.array_equal(state._domain_tensors[color], expected_mask)
            removed = sum(len(before[c])-len(after[c]) for c in data["reward_colors"])
            primary_removed += removed
            traces.append(dict(position=row["position"], index=index, original_level=trace["level"],
                removed=trace["removed"], removed_primary=removed, contradiction=trace["contradiction"],
                estimated_work=trace["estimated_work"], max_tensor_size=trace["max_tensor_size"]))
    assert all(digest(p) == h for p, h in plan["source_sha256"].items())
    output = dict(protocol_sha256=digest(OUT / "protocol.json"), audit_source_sha256=digest(__file__),
        rows=80, valid_solved_assignments=len(checked), checked_assignments=checked,
        original_cores=len(network.coresDict), replayed_useful_rule_traces=len(traces),
        original_levels=dict(Counter(t["original_level"] for t in traces)),
        primary_removed_in_replayed_traces=primary_removed, trace_results=traces,
        replay_uses_no_detector_or_neural_model=True, replay_uses_no_assign_or_dfs=True,
        original_branch_domains_recreated_directly=True, all_domains_and_costs_match=True,
        temporary_cores_and_queue_keys_absent=True, noncontradictory_queues_empty=True,
        contradictory_pending_original_queues=contradiction_queues, source_hashes_unchanged=True,
        row_sha256={p.name: digest(p) for p in paths})
    with (OUT / "trace_audit.json").open("x") as stream:
        json.dump(output, stream, separators=(",", ":"))
        stream.write("\n")
    print("Audited", len(checked), "valid solves and", len(traces), "exact rule traces; primary removed", primary_removed)


if __name__ == "__main__":
    main()
