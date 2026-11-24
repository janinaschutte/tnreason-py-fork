"""Reuse independent proof replay, then check the shared portfolio time ledger."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_matched_time_audit", OUT.parent / "audit.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.OUT = OUT


def audit():
    # The original auditor's only fixed-size assumption is its 128-row check.
    # Verify this extension's complete 16-row coverage explicitly instead.
    result = base.audit(allow_partial=True)
    plan = base.read(OUT / "protocol.json")
    paths = sorted((OUT / "rows").glob("combined_[0-9][0-9].json"))
    rows = [base.read(path) for path in paths]
    assert plan["methods"] == ["combined"] and len(rows) == 16
    assert {(row["method"], row["position"]) for row in rows} == set(map(tuple, plan["order"]))
    assert result["rows"] == 16 and result["residual_prefix_records_checked"] == 64
    expected = {name for path in paths for name in (path.name, path.stem + ".dfs.json")}
    assert set(result["row_sha256"]) == expected and len(expected) == 32
    sources = tuple(plan["sources"])
    assert sources == ("heuristic", "teacher", "reinforce", "replay", "q")
    events_checked = 0
    for row in rows:
        spent, turns = dict.fromkeys(sources, 0.0), Counter()
        exhausted, generation, previous_end, last_candidate = set(), -1, 0.0, {}
        for event in row["source_events"]:
            assert event["generation"] in (generation, generation + 1)
            if event["generation"] != generation:
                generation, exhausted = event["generation"], set()
            available = [source for source in sources if source not in exhausted]
            assert event["source"] == min(available, key=spent.get)
            start, end, charge = (event[key] for key in ("start_s", "end_s", "charged_s"))
            assert all(math.isfinite(value) for value in (start, end, charge))
            assert previous_end <= start <= end <= row["wall_s"] + 1e-8
            assert math.isclose(charge, end-start, abs_tol=1e-10)
            spent[event["source"]] += charge
            turns[event["source"]] += 1
            if event.get("exhausted"):
                assert "candidate_key" not in event
                exhausted.add(event["source"])
            elif "candidate_key" in event:
                last_candidate[generation] = event
            else:
                assert row["status"] == "resource_guard" and event is row["source_events"][-1]
            previous_end = end
            events_checked += 1
        useful = Counter()
        for proof in row["proofs"]:
            event = last_candidate[proof["before"]]
            assert proof["candidate_key"] == event["candidate_key"]
            assert proof["source"] == event["source"]
            assert event["start_s"] <= row["snapshots"][proof["after"]]["wall_s"] <= event["end_s"]
            useful[proof["source"]] += 1
        emitted = sum("candidate_key" in event for event in row["source_events"])
        difference = emitted-row["totals"].get("proposals", 0)
        assert difference in (0, 1)
        if difference:
            # The deadline can land between an emitted yield and the caller's counter.
            assert row["status"] == "resource_guard" and "candidate_key" in row["source_events"][-1]
        for source in sources:
            actual = row["source_totals"][source]
            assert actual["turns"] == turns[source] and actual["useful"] == useful[source]
            assert math.isclose(actual["charged_s"], spent[source], abs_tol=1e-9)
        assert row["unattributed_overhead_s"] >= 0
        assert math.isclose(sum(spent.values()) + row["unattributed_overhead_s"], row["wall_s"], abs_tol=1e-8)
    result.update(partial=False, portfolio_audit_source_sha256=base.digest(__file__),
        source_events_checked=events_checked, source_ledger_and_schedule_verified=True,
        shared_clock_no_unaccounted_time=True, useful_proposal_sources_verified=True)
    return result


if __name__ == "__main__":
    result = audit()
    with (OUT / "audit.json").open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "row_sha256"}, indent=2))
