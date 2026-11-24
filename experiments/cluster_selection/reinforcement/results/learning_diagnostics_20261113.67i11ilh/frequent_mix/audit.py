"""Independently replay recorded proofs and check scheduling; never search CSPs."""
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
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_evaluation_mac import prepare_candidate
from experiments.cluster_selection.runtime import _prepare, apply_candidate
from experiments.cluster_selection.reinforcement.experiment import instances
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.mac import ConstraintNetworkSolver
from tnreason.engine.workload_to_numpy import NumpyCore


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def domain_hash(domains):
    return sha256(json.dumps(domains, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot(state):
    return {color: list(values) for color, values in state.domains.items()}


def forbidden(*args, **kwargs):
    raise AssertionError("DFS and assignments are forbidden in proof replay")


def valid_reference(network, domains, reference):
    assert all(reference[c] in values for c, values in domains.items())
    assert all(core[{c: reference[c] for c in core.colors}] != 0
               for core in network.coresDict.values())


def clean_state(state, network, reference):
    assert state.decision_level == 0 and state.on_closure is None
    assert not state.contradiction and not state._queue and not state._queued
    assert set(state._cores) == set(network.coresDict)
    assert not state._added_core_keys and not state._temporary_core_keys
    assert not state._level_markers
    assert state._core_keys_by_color == network.core_keys_by_color
    assert set(state._contractions) == set(network.coresDict)
    for color, values in state.domains.items():
        assert reference[color] in values
        assert tuple(np.flatnonzero(state._domain_tensors[color])) == values


def fresh_state(network, domains):
    state = network.create_mac_state(domains)
    state.assign = forbidden
    state.propagate()
    return state


def candidate_for(trace, key):
    keys, outputs = tuple(trace["core_keys"]), tuple(trace["open_colors"])
    assert 0 < len(keys) <= 64 and len(set(keys)) == len(keys)
    assert len(outputs) <= 6 and len(set(outputs)) == len(outputs)
    return SummaryClusterCandidate(key, keys, outputs)


def check_events(row, sources):
    generation, cursor, current = 0, 0, row["baseline_domain_hash"]
    exhausted, created = set(), set()
    counts = {s: Counter() for s in sources}
    per_generation = Counter()
    trace_index = 0
    for turn, event in enumerate(row["sequence"], 1):
        while sources[cursor] in exhausted:
            cursor = (cursor + 1) % len(sources)
        source = sources[cursor]
        cursor = (cursor + 1) % len(sources)
        assert event["turn"] == turn and event["source"] == source
        assert event["generation"] == generation and event["before"] == current
        created.add(source)
        kind = event["kind"]
        counts[source][kind] += 1
        per_generation[generation, source] += int(kind in ("cache_hit", "unaffordable", "noop", "useful"))
        if kind == "useful":
            trace = row["useful_traces"][trace_index]
            assert event["trace_index"] == trace_index
            for key in ("turn", "source", "generation", "before", "after", "candidate_signature"):
                assert event[key] == trace[key]
            trace_index += 1
            current = trace["after"]
            generation += 1
            exhausted, created = set(), set()
        elif kind == "bounded_iterator_exhausted":
            exhausted.add(source)
        elif kind == "resource_guard_interrupted_turn":
            assert turn == len(row["sequence"]) and row["resource_guard_reached"]
        else:
            assert kind in ("cache_hit", "unaffordable", "noop")
        assert event["after"] == current
    assert trace_index == len(row["useful_traces"]) == generation == row["generations"]
    assert current == row["domain_hash"]
    assert sorted(exhausted) == row["final_exhausted_sources"]
    if not row["resource_guard_reached"]:
        assert sources[cursor] == row["next_source"]
        assert set(row["final_created_sources"]) == created
    if row["status"] == "bounded_iterators_exhausted":
        assert exhausted == set(sources)
    for (_, source), proposals in per_generation.items():
        assert proposals <= (512 if source == "heuristic" else 32 if source == "mask" else 8192)
    for source, total in row["totals"].items():
        observed = counts[source]
        incomplete = observed["resource_guard_interrupted_turn"]
        normal_turns = sum(observed.values()) - incomplete
        assert abs(total.get("turns", 0) - sum(observed.values())) <= incomplete
        assert total.get("proposal_calls_completed", 0) >= normal_turns
        assert 0 <= total.get("proposal_calls_started", 0) - total.get("proposal_calls_completed", 0) <= incomplete
        assert 0 <= total.get("prepare_calls_started", 0) - total.get("prepare_calls_completed", 0) <= incomplete
        for key, event_kind in (("bounded_iterator_exhaustions", "bounded_iterator_exhausted"),
                                ("cache_skips", "cache_hit"), ("rejected", "unaffordable"),
                                ("noops", "noop"), ("useful", "useful")):
            assert total.get(key, 0) == observed[event_kind]
        applied = observed["noop"] + observed["useful"]
        proposals = applied + observed["cache_hit"] + observed["unaffordable"]
        assert proposals <= total.get("proposals", 0) <= proposals + incomplete
        prepared = applied + observed["unaffordable"]
        assert prepared <= total.get("prepare_calls_completed", 0) <= prepared + incomplete
        assert total.get("contractions_started", 0) == total.get("contractions_completed", 0) == applied
        relevant = [t for t in row["useful_traces"] if t["source"] == source]
        assert total.get("removed_all", 0) == sum(t["removed_all"] for t in relevant)
        assert total.get("removed_primary", 0) == sum(t["removed_primary"] for t in relevant)
        assert math.isclose(total.get("entropy_gain", 0), sum(t["entropy_gain"] for t in relevant), abs_tol=1e-10)
        assert total.get("estimated_work", 0) >= sum(t["estimated_work"] for t in relevant)
        assert 0 <= total.get("max_tensor_size", 0) <= 100000
        assert all(math.isfinite(value) and value >= 0 for value in total.values())
    return counts, per_generation


def audit(variant, omissions=False):
    plan = read(OUT / "protocol.json")
    protocol_hash = digest(OUT / "protocol.json")
    for path, expected in {**plan["source_sha256"], **plan["reference_sha256"]}.items():
        assert digest(path) == expected, path
    assert digest(plan["data"]) == plan["data_sha256"]
    for info in plan["checkpoints"].values():
        assert digest(info["path"]) == info["sha256"]
    paths = sorted((OUT / "cases" / variant).glob("*/row.json"))
    rows = [read(path) for path in paths]
    assert len(rows) == 16 and {r["position"] for r in rows} == set(range(16))
    data = read(plan["data"])
    initial, _ = instances(data)
    raw = data["network"]
    primary = data["reward_colors"]
    entropy = lambda domains: sum(math.log(len(domains[c])) for c in primary)
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    originals = {key: core.values.copy() for key, core in network.coresDict.items()}
    ConstraintNetworkSolver.solve = forbidden
    ConstraintNetworkSolver._search = forbidden
    network.solve_mac = forbidden
    sources = plan["variants"][variant]
    cases, checkpoint_hashes, omission_rows = [], {}, []
    for path, row in zip(paths, rows):
        i = row["position"]
        assert row["variant"] == variant and row["protocol_sha256"] == protocol_hash
        assert row["decisions"] == 0 and row["dfs_called"] is False
        reference_path = OUT.parent / "heldout" / "rows" / f"plain_{i:02d}.json"
        reference = read(reference_path)["assignment"]
        valid_reference(network, initial[i], reference)
        state = fresh_state(network, initial[i])
        baseline = snapshot(state)
        assert baseline == row["baseline_domains"] and domain_hash(baseline) == row["baseline_domain_hash"]
        assert math.isclose(entropy(baseline), row["baseline_entropy"], abs_tol=1e-10)
        clean_state(state, network, reference)
        for n, trace in enumerate(row["useful_traces"]):
            before = snapshot(state)
            assert domain_hash(before) == trace["before"]
            candidate = candidate_for(trace, f"audit_{variant}_{i}_{n}")
            signature = tuple(sorted(candidate.core_keys)), tuple(sorted(candidate.open_colors))
            assert sha256(repr(signature).encode()).hexdigest() == trace["candidate_signature"]
            prepared = prepare_candidate(state, candidate, compact_domains=True, max_work=1000000, max_size=100000)
            assert prepared[0].work == trace["estimated_work"] and prepared[0].max_size == trace["max_tensor_size"]
            result = apply_candidate(state, candidate, prepared)
            assert result.removed_values == trace["removed_all"] and result.contradiction == trace["contradiction"]
            assert result.removed_values or result.contradiction
            assert snapshot(state) == trace["domains"] and domain_hash(snapshot(state)) == trace["after"]
            assert trace["removed_primary"] == sum(len(before[c])-len(state.domains[c]) for c in primary)
            assert math.isclose(trace["entropy_gain"], entropy(before)-entropy(state.domains), abs_tol=1e-10)
            clean_state(state, network, reference)
        assert snapshot(state) == row["domains"] and domain_hash(snapshot(state)) == row["domain_hash"]
        assert state.is_solved() == row["solved"]
        assert math.isclose(entropy(state.domains), row["remaining_entropy"], abs_tol=1e-10)
        if row["solved"]:
            assert row["valid"] and row["assignment"] == reference and row["status"] == "solved"
        else:
            assert not row["assignment"] and row["status"] in ("resource_guard", "bounded_iterators_exhausted")
        assert snapshot(fresh_state(network, row["domains"])) == row["domains"]
        check_events(row, sources)
        for progress in row["structural_progress"]:
            assert progress["generation"] <= row["generations"]
            assert [[stage["quota"], stage["steps"]] for stage in progress["stages_started"]] == plan["structural_stages"][:len(progress["stages_started"])]
        checkpoints = sorted(path.parent.glob("checkpoint_*.json"))
        assert [p.name for p in checkpoints] == [f"checkpoint_{n:05d}.json" for n in range(len(checkpoints))]
        assert len(checkpoints) == len(row["useful_traces"]) + 2
        for checkpoint in checkpoints:
            saved = read(checkpoint)
            assert saved["protocol_sha256"] == protocol_hash and saved["position"] == i and saved["variant"] == variant
            n = len(saved["useful_traces"])
            assert saved["useful_traces"] == row["useful_traces"][:n]
            assert saved["sequence"] == row["sequence"][:len(saved["sequence"])]
            assert saved["domains"] == (row["useful_traces"][n-1]["domains"] if n else baseline)
            assert saved["domain_hash"] == domain_hash(saved["domains"]) and saved["generation"] == n
            assert all(reference[c] in values for c, values in saved["domains"].items())
            checkpoint_hashes[str(checkpoint.relative_to(OUT))] = digest(checkpoint)
        assert read(checkpoints[-1])["domains"] == row["domains"]
        cases.append(dict(position=i, status=row["status"], solved=row["solved"],
            reference_sha256=digest(reference_path), traces_replayed=len(row["useful_traces"]),
            checkpoints=len(checkpoints), turns=len(row["sequence"]), exact_domains=True,
            round_robin_and_counters=True, reference_preserved=True))
        if omissions:
            for omitted in sources:
                alternative = fresh_state(network, initial[i])
                applied, rejected = 0, []
                for n, trace in enumerate(row["useful_traces"]):
                    if trace["source"] == omitted or alternative.is_solved():
                        continue
                    candidate = candidate_for(trace, f"omit_{omitted}_{i}_{n}")
                    prepared = _prepare(alternative, candidate, compact_domains=True, max_work=1000000, max_size=100000)
                    if prepared is None:
                        rejected.append(n)
                        continue
                    apply_candidate(alternative, candidate, prepared)
                    applied += 1
                    clean_state(alternative, network, reference)
                if alternative.is_solved():
                    assert row["solved"] and alternative.assignment() == reference
                omission_rows.append(dict(omitted=omitted, position=i, solved=alternative.is_solved(),
                    applied=applied, unaffordable_trace_indices=rejected,
                    remaining_values=sum(len(v) for v in alternative.domains.values()),
                    final_domain_hash=domain_hash(snapshot(alternative))))
        for key, core in network.coresDict.items():
            np.testing.assert_array_equal(core.values, originals[key])
        print(variant, i, "independent checks passed", flush=True)
    result = dict(variant=variant, protocol_sha256=protocol_hash, audit_source_sha256=digest(__file__),
        source_and_checkpoint_hashes_match=True, neural_inference=False, dfs=False, assignments=False,
        original_core_count=len(network.coresDict), cases=cases,
        row_sha256={str(p.relative_to(OUT)): digest(p) for p in paths}, checkpoint_sha256=checkpoint_hashes,
        limitations=["Reused exploratory16 puzzles; uniqueness proofs inherited from frozen heldout preparation.",
                     "Bounded iterator exhaustion does not establish absence of all useful clusters.",
                     "Interrupted proposal/preparation tail counters may differ by one; applications/proofs are protected."])
    with (OUT / f"audit_{variant}.json").open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    if omissions:
        original_roots = {r["position"] for r in rows if r["solved"]}
        omission_result = dict(variant=variant, protocol_sha256=protocol_hash,
            audit_source_sha256=digest(__file__), cases=omission_rows,
            summary={source: dict(solved=sum(r["solved"] for r in omission_rows if r["omitted"] == source),
                root_positions=[r["position"] for r in omission_rows if r["omitted"] == source and r["solved"]],
                lost_root_positions=sorted(original_roots-{r["position"] for r in omission_rows if r["omitted"] == source and r["solved"]}),
                gained_root_positions=sorted({r["position"] for r in omission_rows if r["omitted"] == source and r["solved"]}-original_roots),
                applied=sum(r["applied"] for r in omission_rows if r["omitted"] == source),
                unaffordable_clusters=sum(len(r["unaffordable_trace_indices"]) for r in omission_rows if r["omitted"] == source),
                unaffordable_positions=[r["position"] for r in omission_rows if r["omitted"] == source and r["unaffordable_trace_indices"]]) for source in sources},
            interpretation="Recontract only the remaining recorded useful sequence after omitting a source. Success establishes redundancy within this recorded proof; failure does not establish necessity because alternative proposals are not searched. These are not scheduled leave-one-source-out reruns.")
        with (OUT / f"proof_omissions_{variant}.json").open("x") as stream:
            json.dump(omission_result, stream, separators=(",", ":"))
            stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--omit-proofs", action="store_true")
    args = parser.parse_args()
    audit(args.variant, args.omit_proofs)
