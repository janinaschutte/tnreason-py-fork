"""Longer reruns of timed-out method/puzzle pairs; reuse completed old results.

These are fresh trajectories from the original domains, not exact resumptions.
Old and new 1,800-second measurements stay separate and are compared explicitly.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import mkdtemp


BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_extended_budget", BASE / "extended_budget.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
engine, cluster, ROOT = frozen.engine, frozen.frozen.cluster, frozen.ROOT
METHODS = frozen.METHODS


def here(path, plan):
    path, previous = Path(path), Path(plan["data"]).parents[5]
    return ROOT / path.relative_to(previous) if path.is_relative_to(previous) else path


def relocated(plan):
    result = deepcopy(plan)
    for field in ("source_sha256", "reference_sha256"):
        result[field] = {str(here(path, plan)): value for path, value in plan[field].items()}
    result["data"] = str(here(plan["data"], plan))
    result["reference_assignments"] = {key: str(here(path, plan)) for key, path in plan["reference_assignments"].items()}
    for checkpoint in result["checkpoints"].values():
        checkpoint["path"] = str(here(checkpoint["path"], plan))
    if "parent_run" in plan:
        result["parent_run"] = str(here(plan["parent_run"], plan))
    return result


def completed(parent):
    """Yield only hash-checked complete parent attempts, including residuals."""
    plan = engine.read(parent / "protocol.json")
    cluster.verify(relocated(plan))
    protocol_hash = engine.digest(parent / "protocol.json")
    assert plan["positions"] == list(range(100)) and plan["methods"] == list(METHODS)
    for position in plan["positions"]:
        marker = engine.read(parent / "done" / f"{position:02d}.json")
        assert marker["position"] == position and marker["protocol_sha256"] == protocol_hash
        attempt = here(marker["attempt"], plan)
        assert engine.digest(attempt / "protocol.json") == protocol_hash
        assert engine.digest(attempt / "worker.json") == marker["worker_sha256"]
        assert set(marker["row_sha256"]) == {f"{method}_{position:02d}{suffix}" for method in METHODS for suffix in (".json", ".dfs.json")}
        for method in METHODS:
            path = attempt / "rows" / f"{method}_{position:02d}.json"
            residual_path = path.with_suffix(".dfs.json")
            assert engine.digest(path) == marker["row_sha256"][path.name]
            assert engine.digest(residual_path) == marker["row_sha256"][residual_path.name]
            row, residual = engine.read(path), engine.read(residual_path)
            assert row["protocol_sha256"] == residual["protocol_sha256"] == protocol_hash
            assert row["position"] == residual["position"] == position and row["method"] == residual["method"] == method
            assert residual["root_row_sha256"] == engine.digest(path)
            assert row["reference_retained"] and row["no_added_cores"] and not row["dfs_called"]
            yield position, method, path, row, residual


def prepare(parent, seconds, max_jobs):
    parent = parent.resolve()
    original = engine.read(parent / "protocol.json")
    assert seconds > original["seconds"] and max_jobs > 0
    selected, counts, stopped = defaultdict(list), Counter(), defaultdict(Counter)
    for position, method, _, row, _ in completed(parent):
        solved = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= original["seconds"])
        if not solved and row["status"] == "resource_guard":
            selected[position].append(method)
            counts[method] += 1
        else:
            assert solved or row["status"] in {"ordinary_gac_stalled", "technical_iterator_exhausted", "expert_bounded_detectors_exhausted"}
            stopped[method]["solved" if solved else row["status"]] += 1
    assert selected, "No time-limited cases to extend"
    plan = relocated(original)
    plan.update(kind="selective_longer_reruns", seconds=seconds,
        budgets_s=sorted({original["seconds"], *[value for value in (3600, 7200, seconds) if original["seconds"] < value <= seconds]}),
        parent_run=str(parent), parent_protocol_sha256=engine.digest(parent / "protocol.json"),
        parent_markers={str(position): engine.digest(parent / "done" / f"{position:02d}.json") for position in original["positions"]},
        rerun_methods={str(position): methods for position, methods in sorted(selected.items())},
        rerun_counts=dict(counts), reused_counts={method: dict(values) for method, values in stopped.items()},
        workers=max_jobs,
        interpretation="Only previously time-limited pairs rerun, from ORIGINAL domains, with a larger total deadline. "
            "Completed and naturally stopped parent results are reused with their original provenance. "
            "This is not exact resumption or a new fully paired replication. Original curves through the old deadline "
            "remain old measurements; larger-budget totals combine reused cases and fresh reruns. "
            "New old-deadline prefixes are exported separately, including any disagreements. "
            "Mixed scheduling is wall-time-dependent and can follow another path. Sudoku expert remains a separate reference.")
    plan["source_sha256"][str(Path(__file__).resolve())] = engine.digest(Path(__file__))
    plan["source_sha256"][str(parent / "protocol.json")] = engine.digest(parent / "protocol.json")
    out = Path(mkdtemp(prefix=f"rerun{seconds}_", dir=BASE))
    for name in ("attempts", "done", "logs"):
        (out / name).mkdir()
    positions = "\n".join(str(position) for position in selected)
    submission = f'''universe = vanilla
initialdir = {ROOT}
executable = {sys.executable}
transfer_executable = False
should_transfer_files = NO
arguments = "-u '{Path(__file__).resolve()}' task '{out}' $(position)"
request_cpus = 1
request_gpus = 0
requirements = (TARGET.TotalGPUs == 0)
request_memory = 4GB
request_disk = 1GB
getenv = PATH, LD_LIBRARY_PATH
environment = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cluster-mpl-$(ClusterID)-$(Process) XDG_CACHE_HOME=/tmp/cluster-cache-$(ClusterID)-$(Process)"
output = {out}/logs/$(ClusterID).$(Process).out
error = {out}/logs/$(ClusterID).$(Process).err
log = {out}/logs/$(ClusterID).$(Process).log
stream_output = True
stream_error = True
on_exit_hold = (ExitBySignal =?= True) || (ExitCode =!= 0)
max_materialize = {max_jobs}
queue position from (
{positions}
)
'''
    with (out / "cpu.sub").open("x") as stream:
        stream.write(submission)
    plan["source_sha256"][str(out / "cpu.sub")] = engine.digest(out / "cpu.sub")
    cluster.verify(plan)
    engine.write(out / "protocol.json", plan)
    print(f"{sum(counts.values())} method/puzzle pairs in {len(selected)} jobs; at most {sum(counts.values()) * seconds / 3600:g} CPU-hours plus setup/residuals", file=sys.stderr)
    print(out)


def finished_attempt(out, position, plan, raw):
    marker = engine.read(out / "done" / f"{position:02d}.json")
    methods = plan["rerun_methods"][str(position)]
    expected = {f"{method}_{position:02d}{suffix}" for method in methods for suffix in (".json", ".dfs.json")}
    assert marker["position"] == position and marker["methods"] == methods
    assert marker["protocol_sha256"] == engine.digest(out / "protocol.json")
    assert set(marker["row_sha256"]) == expected
    attempt = here(marker["attempt"], raw)
    assert engine.digest(attempt / "protocol.json") == engine.digest(out / "protocol.json")
    assert engine.digest(attempt / "worker.json") == marker["worker_sha256"]
    assert all(engine.digest(attempt / "rows" / name) == value for name, value in marker["row_sha256"].items())
    return attempt


def task(out, position):
    raw = engine.read(out / "protocol.json")
    plan = relocated(raw)
    cluster.verify(plan)
    methods = plan["rerun_methods"][str(position)]
    finished = out / "done" / f"{position:02d}.json"
    protocol_hash = engine.digest(out / "protocol.json")
    if finished.exists():
        finished_attempt(out, position, plan, raw)
        print(f"Puzzle {position} already complete")
        return
    attempt = Path(mkdtemp(prefix=f"{position:02d}_", dir=out / "attempts"))
    (attempt / "rows").mkdir()
    with (attempt / "protocol.json").open("x") as stream:
        stream.write((out / "protocol.json").read_text())
    cluster.initialize(attempt, plan)
    portfolio = frozen.frozen.module("frozen_shared_portfolio", frozen.frozen.PORTFOLIO)
    portfolio.helper, portfolio.write = engine, engine.write
    combined = portfolio.Combined(engine.CONTEXT["models"])
    engine.CONTEXT["models"]["combined"] = combined
    ordered = [method for method, index in plan["order"] if index == position and method in methods]
    assert set(ordered) == set(methods) and len(ordered) == len(methods)
    original_clock, original_write = engine.perf_counter, engine.write
    for method in ordered:
        try:
            if method == "combined":
                combined.reset()
                engine.perf_counter, engine.write = combined.clock, combined.publish
            engine.evaluate(method, position)
        finally:
            engine.perf_counter, engine.write = original_clock, original_write
    for method in ordered:
        engine.residual(method, position)
    paths = sorted((attempt / "rows").glob("*.json"))
    assert len(paths) == 2 * len(methods)
    marker = dict(position=position, protocol_sha256=protocol_hash, attempt=str(attempt),
        methods=methods, row_sha256={path.name: engine.digest(path) for path in paths},
        worker_sha256=engine.digest(attempt / "worker.json"))
    try:
        engine.write(finished, marker)
    except FileExistsError:
        assert engine.read(finished)["protocol_sha256"] == protocol_hash
    print(f"COMPLETE puzzle {position}: {len(methods)} selected methods and residual DFS", flush=True)


def collect(out):
    raw = engine.read(out / "protocol.json")
    plan = relocated(raw)
    cluster.verify(plan)
    parent = Path(plan["parent_run"])
    assert engine.digest(parent / "protocol.json") == plan["parent_protocol_sha256"]
    for position, expected in plan["parent_markers"].items():
        assert engine.digest(parent / "done" / f"{int(position):02d}.json") == expected
    outputs = [out / name for name in ("totals.csv", "per_puzzle.csv", "old_deadline_comparison.csv", "summary.json")]
    assert not any(path.exists() for path in outputs), "Preserve previous exports"
    old_seconds = engine.read(parent / "protocol.json")["seconds"]
    details, comparisons, attempts = [], [], {}
    protocol_hash = engine.digest(out / "protocol.json")
    for position, method, parent_path, old, old_residual in completed(parent):
        rerun = method in plan["rerun_methods"].get(str(position), [])
        row, residual, path = old, old_residual, parent_path
        if rerun:
            if position not in attempts:
                attempts[position] = finished_attempt(out, position, plan, raw)
            attempt = attempts[position]
            path = attempt / "rows" / f"{method}_{position:02d}.json"
            residual_path = path.with_suffix(".dfs.json")
            row, residual = engine.read(path), engine.read(residual_path)
            assert row["protocol_sha256"] == residual["protocol_sha256"] == protocol_hash
            assert residual["root_row_sha256"] == engine.digest(path)
            old_index = old["prefixes"][str(old_seconds)]["snapshot"]
            new_index = row["prefixes"][str(old_seconds)]["snapshot"]
            old_domains = old["snapshots"][old_index]["domains"] if old_index is not None else None
            new_domains = row["snapshots"][new_index]["domains"] if new_index is not None else None
            comparisons.append(dict(method=method, position=position, old_solved=old["prefixes"][str(old_seconds)]["solved"],
                rerun_solved=row["prefixes"][str(old_seconds)]["solved"], same_domains=old_domains == new_domains,
                old_row_sha256=engine.digest(parent_path), rerun_row_sha256=engine.digest(path)))
        assert row["method"] == residual["method"] == method and row["position"] == residual["position"] == position
        assert row["reference_retained"] and row["no_added_cores"] and row["no_pending_queue"] and not row["dfs_called"]
        assert row["decision_level"] == 0 and residual["timed_separately"]
        for budget in plan["budgets_s"]:
            if budget <= old_seconds:
                continue
            key = str(budget if rerun else old_seconds)
            prefix, finish = row["prefixes"][key], residual["budgets"][key]
            eligible = [index for index, snap in enumerate(row["snapshots"]) if snap["wall_s"] <= (budget if rerun else old_seconds)]
            assert prefix["snapshot"] == finish["snapshot"] == (eligible[-1] if eligible else None)
            solved = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= (budget if rerun else old_seconds))
            assert solved == prefix["solved"]
            assert not solved or finish["valid"] and finish["solved"] and finish["nodes"] == 1
            details.append(dict(method=method, position=position, budget_s=budget, origin="rerun" if rerun else "reused",
                solved_without_branching=int(solved), remaining_dfs_nodes=finish["nodes"] if finish["valid"] and finish["solved"] else None,
                residual_status=finish["status"], solved_s=row["solved_s"], status=row["status"],
                row_sha256=engine.digest(path), used_root_wall_s=min(row["wall_s"], budget)))
    totals = []
    for method in METHODS:
        for budget in plan["budgets_s"]:
            if budget <= old_seconds:
                continue
            group = [row for row in details if row["method"] == method and row["budget_s"] == budget]
            assert len(group) == 100 and {row["position"] for row in group} == set(range(100))
            censored = sum(row["remaining_dfs_nodes"] is None for row in group)
            totals.append(dict(method=method, budget_s=budget, instances=100,
                solved_without_branching=sum(row["solved_without_branching"] for row in group),
                remaining_dfs_nodes=None if censored else sum(row["remaining_dfs_nodes"] for row in group),
                residual_censored=censored, rerun_instances=sum(row["origin"] == "rerun" for row in group),
                mean_used_root_wall_s=sum(row["used_root_wall_s"] for row in group) / 100))
    for path, rows in zip(outputs, (totals, details, comparisons)):
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    engine.write(out / "summary.json", dict(protocol_sha256=protocol_hash, interpretation=plan["interpretation"],
        old_deadline_domains_differ=sum(not row["same_domains"] for row in comparisons),
        old_deadline_new_solves=sum(row["rerun_solved"] for row in comparisons), totals=totals))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("parent", type=Path)
    preparation.add_argument("--seconds", type=int, default=7200)
    preparation.add_argument("--max-jobs", type=int, default=10)
    for name in ("task", "collect"):
        command = commands.add_parser(name)
        command.add_argument("out", type=Path)
        if name == "task":
            command.add_argument("position", type=int)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "task": task, "collect": collect}[command](**args)
