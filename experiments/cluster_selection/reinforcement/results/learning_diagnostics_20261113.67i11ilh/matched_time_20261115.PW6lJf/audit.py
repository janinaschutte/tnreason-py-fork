"""Independently replay timed root proofs and verify every reported time prefix."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[5]))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def domains(values):
    return {color: tuple(allowed) for color, allowed in values.items()}


def forbid(*args, **kwargs):
    raise AssertionError("Independent proof audit must not use assignments, models or DFS")


def clean(state, network):
    assert not state.contradiction and not state.decision_level
    assert not state._queue and not state._queued and state.on_closure is None
    assert not state._added_core_keys and set(state._cores) == set(network.coresDict)
    for color, allowed in state.domains.items():
        mask = np.zeros(state.dimensions[color], dtype=bool)
        mask[list(allowed)] = True
        assert np.array_equal(mask, state._domain_tensors[color])


def valid(network, original, assignment):
    assert set(assignment) == set(network.dimension_dict)
    assert all(assignment[color] in values for color, values in original.items())
    for core in network.coresDict.values():
        assert core[{color: assignment[color] for color in core.colors}] != 0


def check_hashes(plan):
    for path, expected in {**plan["source_sha256"], **plan["reference_sha256"]}.items():
        assert digest(path) == expected, path
    assert digest(plan["data"]) == plan["data_sha256"]
    for item in plan["checkpoints"].values():
        assert digest(item["path"]) == item["sha256"]


def audit(allow_partial=False):
    from demonstrations.sudoku.inferenceClusters import RuleCluster, RuleContext, RuleLimits, contract_cluster
    from experiments.cluster_selection.candidate_evaluation_mac import prepare_candidate
    from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.cluster_selection.runtime import apply_candidate
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from tnreason.engine.workload_to_numpy import NumpyCore

    plan = read(OUT / "protocol.json")
    protocol_hash = digest(OUT / "protocol.json")
    check_hashes(plan)
    assert plan["workers"] == 1 and plan["contraction_cap"] is None
    assert plan["cumulative_work_cap"] is None
    assert plan["budgets_s"] == [1, 5, 15, 60] and plan["seconds"] == 60
    data = read(plan["data"])
    raw, initial = data["network"], instances(data)[0]
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]]) for core in raw["cores"]}, maintain_indices=True)
    network.solve_mac = forbid
    paths = sorted(p for p in (OUT / "rows").glob("*.json") if not p.name.endswith(".dfs.json"))
    rows = [read(p) for p in paths]
    keys = {(row["method"], row["position"]) for row in rows}
    assert len(keys) == len(rows) and keys <= set(map(tuple, plan["order"]))
    if not allow_partial:
        assert keys == set(map(tuple, plan["order"])) and len(rows) == 128
    total_proofs, snapshots_checked, assignments_checked = 0, 0, 0
    solved_by_budget = {method: Counter() for method in plan["methods"]}
    residual_paths, residual_checks = [], 0
    for path, row in zip(paths, rows):
        assert row["protocol_sha256"] == protocol_hash
        method, position = row["method"], row["position"]
        assert path.name == f"{method}_{position:02d}.json"
        assert row["decision_level"] == 0 and row["no_added_cores"] and not row["dfs_called"]
        assert row["reference_retained"] and row["affinity"] == [plan["cpu"]]
        if "no_pending_queue" in row:
            assert row["no_pending_queue"]
        reference = read(plan["reference_assignments"][str(position)])["assignment"]
        valid(network, initial[position], reference)
        baseline = network.create_mac_state(initial[position])
        baseline.assign = forbid
        baseline.propagate()
        clean(baseline, network)
        snapshots, proofs = row["snapshots"], row["proofs"]
        assert snapshots and snapshots[0]["reason"] == "ordinary_gac"
        assert domains(snapshots[0]["domains"]) == baseline.snapshot_domains()
        assert row["baseline_s"] == snapshots[0]["wall_s"]
        previous_domains, previous_wall, previous_cpu = None, 0, 0
        for index, snap in enumerate(snapshots):
            current = domains(snap["domains"])
            assert set(current) == set(network.dimension_dict)
            assert all(values and tuple(sorted(set(values))) == values for values in current.values())
            assert all(reference[color] in values for color, values in current.items())
            assert all(0 <= value < network.dimension_dict[color] for color, values in current.items() for value in values)
            assert all(math.isfinite(snap[key]) for key in ("wall_s", "cpu_s"))
            assert snap["wall_s"] >= previous_wall and snap["cpu_s"] >= previous_cpu
            assert snap["cpu_s"] <= snap["wall_s"] + .02
            assert snap["solved"] == all(len(values) == 1 for values in current.values())
            if previous_domains is not None:
                assert snap["reason"] == "useful_contraction"
                assert all(set(values) <= set(previous_domains[color]) for color, values in current.items())
                assert current != previous_domains
            if snap["solved"]:
                valid(network, initial[position], {color: values[0] for color, values in current.items()})
                assignments_checked += 1
            previous_domains, previous_wall, previous_cpu = current, snap["wall_s"], snap["cpu_s"]
            snapshots_checked += 1
        assert row["wall_s"] >= previous_wall and row["cpu_s"] >= previous_cpu
        assert row["cpu_s"] <= row["wall_s"] + .02
        assert len(proofs) == len(snapshots)-1 == row["totals"].get("useful", 0)
        assert row["totals"].get("contractions", 0) >= len(proofs)
        assert method != "plain" or not proofs
        solved = next((snap for snap in snapshots if snap["solved"]), None)
        assert row["solved_s"] == (solved["wall_s"] if solved else None)
        assert row["solved_cpu_s"] == (solved["cpu_s"] if solved else None)
        assert row["valid"] == bool(solved)
        if solved:
            valid(network, initial[position], row["assignment"])
            assert row["status"] == ("solved" if row["solved_s"] <= plan["seconds"] else "solved_after_deadline")
        else:
            assert not row["assignment"] and row["status"] in (
                "resource_guard", "ordinary_gac_stalled", "expert_bounded_detectors_exhausted", "technical_iterator_exhausted")
        for budget in plan["budgets_s"]:
            eligible = [index for index, snap in enumerate(snapshots) if snap["wall_s"] <= budget]
            expected = eligible[-1] if eligible else None
            actual = row["prefixes"][str(budget)]
            assert actual["snapshot"] == expected
            assert actual["solved"] == bool(expected is not None and snapshots[expected]["solved"])
            solved_by_budget[method][str(budget)] += int(actual["solved"])
        for index, proof in enumerate(proofs):
            assert proof["before"] == index and proof["after"] == index+1
            before, after = (domains(snapshots[i]["domains"]) for i in (index, index+1))
            state = network.create_mac_state(before)
            state.assign = forbid
            state.propagate()
            clean(state, network)
            assert state.snapshot_domains() == before
            candidate = SummaryClusterCandidate(proof.get("candidate_key", f"audit_{position}_{index}"),
                tuple(proof["core_keys"]), tuple(proof["open_colors"]))
            assert candidate.core_keys and len(candidate.core_keys) == len(set(candidate.core_keys))
            assert set(candidate.core_keys) <= set(network.coresDict)
            if method == "expert":
                # Registries use plural collection names; individual subset
                # and ALS-chain proofs carry the singular motif name.
                aliases = {"naked_subset": "naked_subsets", "hidden_subset": "hidden_subsets", "als_chain": "als_chains"}
                assert aliases.get(proof["family"], proof["family"]) in plan["expert_families"]
                scope = tuple(dict.fromkeys(color for key in candidate.core_keys for color in network.coresDict[key].colors))
                cluster = RuleCluster(proof["family"], candidate, scope,
                    tuple(color for color in scope if color not in candidate.open_colors), scope)
                context = RuleContext(network, before, num=3, limits=RuleLimits(**plan["expert_limits"]))
                result = contract_cluster(context, cluster)
                assert result.status == "executed" and not result.contradiction
                assert result.work == proof["estimated_work"] and result.max_size == proof["max_size"]
                for color, allowed in result.supported_domains.items():
                    state.restrict_domain(color, allowed)
                state.propagate()
            else:
                assert proof["family"] is None
                assert len(candidate.core_keys) <= plan["limits"]["max_cores"]
                assert len(candidate.open_colors) <= plan["limits"]["max_open_colors"]
                prepared = prepare_candidate(state, candidate, compact_domains=True,
                    max_work=plan["limits"]["max_work"], max_size=plan["limits"]["max_size"])
                assert prepared[0].work == proof["estimated_work"] and prepared[0].max_size == proof["max_size"]
                apply_candidate(state, candidate, prepared)
            clean(state, network)
            assert state.snapshot_domains() == after
            assert proof["removed"] == sum(len(before[color])-len(after[color]) for color in before) > 0
            total_proofs += 1
        assert row["totals"].get("removed_values", 0) == sum(proof["removed"] for proof in proofs)
        residual_path = path.with_name(path.stem + ".dfs.json")
        if residual_path.exists():
            residual = read(residual_path)
            assert residual["root_row_sha256"] == digest(path) and residual["protocol_sha256"] == protocol_hash
            assert residual["method"] == method and residual["position"] == position and residual["timed_separately"]
            for budget in plan["budgets_s"]:
                result = residual["budgets"][str(budget)]
                assert result["snapshot"] == row["prefixes"][str(budget)]["snapshot"]
                if result["solved"]:
                    assert result["valid"] and result["nodes"] >= 1
                    if row["prefixes"][str(budget)]["solved"]:
                        assert result["nodes"] == 1
                residual_checks += 1
            residual_paths.append(residual_path)
    check_hashes(plan)
    return dict(protocol_sha256=protocol_hash, audit_source_sha256=digest(__file__), partial=allow_partial,
        rows=len(rows), original_cores=len(network.coresDict), snapshots_checked=snapshots_checked,
        useful_proofs_replayed=total_proofs, solved_snapshot_assignments_checked=assignments_checked,
        solved_by_budget={method: dict(counts) for method, counts in solved_by_budget.items()},
        residual_prefix_records_checked=residual_checks,
        replay_uses_no_models_detectors_assignments_or_dfs=True, all_prefix_cutoffs_match=True,
        all_original_core_solutions_valid=True, all_domain_and_cost_replays_match=True,
        no_temporary_cores_or_pending_queues_in_replayed_states=True, sources_unchanged=True,
        row_sha256={p.name: digest(p) for p in paths+residual_paths})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.allow_partial)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, separators=(",", ":"))
            stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "row_sha256"}, indent=2))
