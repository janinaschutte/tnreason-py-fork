"""Paired 300-second CPU replication, including one shared-budget generic mix.

Each job reruns all nine methods on one of the original 100 puzzles. Existing
results and checkpoints remain inputs only; every preparation creates a new run.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
from random import Random
import sys
from tempfile import mkdtemp


BASE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


cluster = module("frozen_paired_cluster", BASE / "cluster.py")
engine, ROOT, OLD = cluster.engine, cluster.ROOT, cluster.OLD
PORTFOLIO = OLD / "portfolio_6ddpp97e/run.py"
METHODS = (*engine.METHODS, "combined")
SOURCES = ("heuristic", "teacher", "reinforce", "replay", "q")
BUDGETS = [1, 5, 15, 60, 120, 300]
SEED = 2026112301


def prepare(max_jobs):
    assert max_jobs > 0
    original = engine.read(BASE / "protocol.json")
    old_root = Path(original["data"]).parents[5]

    def relocated(path):
        return str(ROOT / Path(path).relative_to(old_root))

    plan = deepcopy(original)
    for field in ("source_sha256", "reference_sha256"):
        plan[field] = {relocated(path): digest for path, digest in original[field].items()}
    plan["data"] = relocated(original["data"])
    plan["reference_assignments"] = {key: relocated(path) for key, path in original["reference_assignments"].items()}
    for checkpoint in plan["checkpoints"].values():
        checkpoint["path"] = relocated(checkpoint["path"])
    assert plan["positions"] == list(range(100))
    order = [(method, position) for position in plan["positions"] for method in METHODS]
    Random(SEED).shuffle(order)
    plan.update(kind="paired_100_long_budget_generic_portfolio", methods=list(METHODS),
        sources=list(SOURCES), budgets_s=BUDGETS, seconds=300, order=order, seed=SEED,
        cpu=None, workers=max_jobs, parent_protocol=str(BASE / "protocol.json"),
        parent_protocol_sha256=engine.digest(BASE / "protocol.json"),
        scheduling="least cumulative charged source wall time; one candidate turn",
        environment={"recorded_per_attempt": True, "torch_threads": 1, "blas_threads": 1})
    plan.pop("cpu_topology", None)
    plan.pop("original_affinity", None)
    plan["timing"] = original["timing"].replace("One serial pinned CPU process;", "One single-thread CPU process per puzzle;")
    plan["interpretation"] = (
        "Exploratory replication on all 100 previously evaluated puzzles, without performance-based "
        "selection. All nine methods rerun from the original domains, on the same worker per puzzle, "
        "with fixed checkpoints, decoders and per-cluster limits. One 300s trajectory supplies prefixes "
        "1/5/15/60/120/300s; no saved trajectory is resumed or spliced into this replication. The generic "
        "mix shares one aggregate budget across heuristic/teacher/REINFORCE/replay/Q, restarting their "
        "iterators after useful pruning on shared updated domains. It targets equal cumulative wall "
        "shares with nonpreemptive one-candidate turns, not hard per-source quotas. Switching, proposal, "
        "planning, contraction and GAC are charged; no Sudoku expert or untrained control enters the mix. "
        "The expert remains a separate reference with its original larger limits. Model/network/state "
        "loading is excluded. Hardware is recorded per attempt; do not merge these timings with old runs.")
    plan["execution"] = (
        "One one-thread CPU job per puzzle; all nine methods sequentially on the same worker, in seeded "
        "random order. Respect scheduler affinity. A restarted job reruns the whole puzzle in a separate "
        "attempt; only a complete attempt publishes a completion marker. Residual plain MAC is separate.")
    for path in (Path(__file__).resolve(), BASE / "cluster.py", BASE / "protocol.json", PORTFOLIO, OLD / "plot.py"):
        plan["source_sha256"][str(path)] = engine.digest(path)
    cluster.verify(plan)
    out = Path(mkdtemp(prefix="long300_", dir=BASE))
    for name in ("attempts", "done", "logs"):
        (out / name).mkdir()
    submission = f'''universe = vanilla
initialdir = {ROOT}
executable = {sys.executable}
transfer_executable = False
should_transfer_files = NO
arguments = "-u '{Path(__file__).resolve()}' task '{out}' $(Process)"
request_cpus = 1
request_gpus = 0
requirements = (TARGET.TotalGPUs == 0)
request_memory = 4GB
request_disk = 1GB
getenv = PATH, LD_LIBRARY_PATH
environment = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cluster-mpl-$(ClusterID)-$(Process)"
output = {out}/logs/$(ClusterID).$(Process).out
error = {out}/logs/$(ClusterID).$(Process).err
log = {out}/logs/$(ClusterID).$(Process).log
stream_output = True
stream_error = True
on_exit_hold = (ExitBySignal =?= True) || (ExitCode =!= 0)
max_materialize = {max_jobs}
queue 100
'''
    with (out / "cpu.sub").open("x") as handle:
        handle.write(submission)
    plan["source_sha256"][str(out / "cpu.sub")] = engine.digest(out / "cpu.sub")
    engine.write(out / "protocol.json", plan)
    print(out)


def task(out, position):
    plan = engine.read(out / "protocol.json")
    assert position in plan["positions"] and plan["methods"] == list(METHODS)
    finished = out / "done" / f"{position:02d}.json"
    if finished.exists():
        cluster.verify(plan)
        marker = engine.read(finished)
        assert marker["protocol_sha256"] == engine.digest(out / "protocol.json")
        assert len(marker["row_sha256"]) == 18
        assert all(engine.digest(Path(marker["attempt"]) / "rows" / name) == digest
                   for name, digest in marker["row_sha256"].items())
        print(f"Puzzle {position} already complete")
        return
    attempt = Path(mkdtemp(prefix=f"{position:02d}_", dir=out / "attempts"))
    (attempt / "rows").mkdir()
    with (attempt / "protocol.json").open("x") as handle:
        handle.write((out / "protocol.json").read_text())
    cluster.initialize(attempt, plan)
    original_clock, original_write = engine.perf_counter, engine.write
    portfolio = module("frozen_shared_portfolio", PORTFOLIO)
    portfolio.helper, portfolio.write = engine, original_write
    assert portfolio.SOURCES == SOURCES
    combined = portfolio.Combined(engine.CONTEXT["models"])
    engine.CONTEXT["models"]["combined"] = combined
    methods = [method for method, index in plan["order"] if index == position]
    assert set(methods) == set(METHODS) and len(methods) == 9
    for method in methods:
        try:
            if method == "combined":
                combined.reset()
                engine.perf_counter, engine.write = combined.clock, combined.publish
            engine.evaluate(method, position)
        finally:
            engine.perf_counter, engine.write = original_clock, original_write
    for method in methods:
        engine.residual(method, position)
    paths = sorted((attempt / "rows").glob("*.json"))
    assert len(paths) == 18
    record = dict(position=position, protocol_sha256=engine.digest(out / "protocol.json"), attempt=str(attempt),
        row_sha256={path.name: engine.digest(path) for path in paths},
        worker_sha256=engine.digest(attempt / "worker.json"))
    try:
        engine.write(finished, record)
    except FileExistsError:
        assert engine.read(finished)["protocol_sha256"] == record["protocol_sha256"]
    print(f"COMPLETE puzzle {position}: nine methods and residual DFS", flush=True)


def collect(out):
    plan = engine.read(out / "protocol.json")
    cluster.verify(plan)
    assert plan["methods"] == list(METHODS) and plan["positions"] == list(range(100))
    assert plan["budgets_s"] == BUDGETS and plan["seconds"] == 300
    stems = ("matched_time_roots_no_caption", "matched_time_roots_and_dfs_no_caption")
    outputs = [f"{stem}.{extension}" for stem in stems for extension in ("png", "pdf", "svg")]
    outputs += ["matched_time_totals.csv", "matched_time_per_puzzle.csv", "summary.json", "RESULTS.txt"]
    assert not any((out / name).exists() for name in outputs), "Refusing to overwrite exports"
    groups, details = defaultdict(list), []
    protocol_hash = engine.digest(out / "protocol.json")
    for position in plan["positions"]:
        marker = engine.read(out / "done" / f"{position:02d}.json")
        assert marker["position"] == position and marker["protocol_sha256"] == protocol_hash
        attempt = Path(marker["attempt"])
        assert engine.digest(attempt / "protocol.json") == protocol_hash
        assert engine.digest(attempt / "worker.json") == marker["worker_sha256"]
        assert set(marker["row_sha256"]) == {f"{m}_{position:02d}{s}" for m in METHODS for s in (".json", ".dfs.json")}
        worker = engine.read(attempt / "worker.json")
        for name, expected in marker["row_sha256"].items():
            assert engine.digest(attempt / "rows" / name) == expected
        for method in METHODS:
            path = attempt / "rows" / f"{method}_{position:02d}.json"
            row, residual = engine.read(path), engine.read(path.with_suffix(".dfs.json"))
            assert row["protocol_sha256"] == residual["protocol_sha256"] == protocol_hash
            assert row["position"] == residual["position"] == position and row["method"] == residual["method"] == method
            assert row["no_added_cores"] and not row["dfs_called"] and row["reference_retained"]
            assert row["decision_level"] == 0 and row["no_pending_queue"] and residual["timed_separately"]
            assert row["affinity"] == worker["affinity"] and residual["root_row_sha256"] == engine.digest(path)
            if method == "combined":
                assert set(row["source_totals"]) == set(SOURCES)
                assert all(proof["source"] in SOURCES for proof in row["proofs"])
                charged = sum(info["charged_s"] for info in row["source_totals"].values())
                assert abs(charged + row["unattributed_overhead_s"] - row["wall_s"]) < 1e-7
                assert row["unattributed_overhead_s"] >= -1e-8
                previous_end = 0.
                for event in row["source_events"]:
                    assert event["source"] in SOURCES
                    assert previous_end - 1e-7 <= event["start_s"] <= event["end_s"] <= row["wall_s"] + 1e-7
                    assert abs(event["charged_s"] - (event["end_s"] - event["start_s"])) < 1e-7
                    previous_end = event["end_s"]
                for source in SOURCES:
                    events = [event for event in row["source_events"] if event["source"] == source]
                    info = row["source_totals"][source]
                    assert info["turns"] == len(events)
                    assert info["useful"] == sum(proof["source"] == source for proof in row["proofs"])
                    assert abs(info["charged_s"] - sum(event["charged_s"] for event in events)) < 1e-7
            groups[method].append((row, residual))
            for budget in BUDGETS:
                prefix, finish = row["prefixes"][str(budget)], residual["budgets"][str(budget)]
                eligible = [i for i, snapshot in enumerate(row["snapshots"]) if snapshot["wall_s"] <= budget]
                assert prefix["snapshot"] == finish["snapshot"] == (eligible[-1] if eligible else None)
                solved = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= budget)
                assert prefix["solved"] == solved
                assert not solved or finish["valid"] and finish["solved"] and finish["nodes"] == 1
                details.append(dict(method=method, position=position, budget_s=budget, host=worker["host"],
                    cpu_model=worker["cpu_model"], solved_without_branching=int(solved),
                    remaining_dfs_nodes=finish["nodes"] if finish["valid"] and finish["solved"] else None,
                    residual_status=finish["status"], solved_s=row["solved_s"], trajectory_wall_s=row["wall_s"],
                    trajectory_cpu_s=row["cpu_s"]))
    totals = []
    for method in METHODS:
        for budget in BUDGETS:
            values = [row for row in details if row["method"] == method and row["budget_s"] == budget]
            assert len(values) == 100
            censored = sum(row["remaining_dfs_nodes"] is None for row in values)
            totals.append(dict(method=method, budget_s=budget, instances=100,
                solved_without_branching=sum(row["solved_without_branching"] for row in values),
                remaining_dfs_nodes=None if censored else sum(row["remaining_dfs_nodes"] for row in values),
                residual_censored=censored))
    for name, rows in (("totals", totals), ("per_puzzle", details)):
        with (out / f"matched_time_{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    engine.write(out / "summary.json", dict(protocol_sha256=protocol_hash, totals=totals))
    drawing = module("frozen_plot_styles", OLD / "plot.py")
    styles = dict(drawing.STYLES, combined=("Generic mix (shared time)", "#000000", "-", "*"))
    plt = drawing.plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 12,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    for paired, stem in zip((False, True), stems):
        fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(16, 7.3) if paired else (11.4, 7.1), squeeze=False)
        for index, ax in enumerate(axes.flat):
            ax.set_xscale("symlog", linthresh=1, linscale=.65)
            ax.set_xlim(0, 300)
            ax.set_xticks([0, *BUDGETS], [str(value) for value in (0, *BUDGETS)])
            ax.minorticks_off()
            ax.set_xlabel("Root propagation time allowed per puzzle (seconds)")
            ax.grid(axis="y", color="0.87", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            for method, (label, color, style, marker) in styles.items():
                if index == 0:
                    times = sorted(row["solved_s"] for row, _ in groups[method]
                        if row["valid"] and row["solved_s"] is not None and row["solved_s"] <= 300)
                    ax.step([0, *times, 300], [0, *range(1, len(times)+1), len(times)], where="post",
                            label=label, color=color, linestyle=style, linewidth=2.1)
                    ax.scatter(BUDGETS, [sum(t <= budget for t in times) for budget in BUDGETS],
                               color=color, marker=marker, s=34, zorder=4, clip_on=False)
                else:
                    nodes = [row["remaining_dfs_nodes"] for row in totals if row["method"] == method]
                    ax.plot(BUDGETS, [float("nan") if n is None else n for n in nodes],
                            label=label, color=color, marker=marker, linestyle="None", markersize=7)
            ax.set_ylabel("Puzzles solved without branching" if index == 0 else "Remaining DFS nodes, summed over 100 puzzles")
            ax.set_title("Solving 100 held-out Sudokus without branching" if index == 0 else "Search remaining after propagation", pad=18)
            if index == 0:
                ax.set_ylim(-1, 103)
                ax.set_yticks(range(0, 101, 10))
            else:
                ax.axhline(100, color=".6", linewidth=1, linestyle=":")
                ax.set_ylim(bottom=98)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, loc="lower center", bbox_to_anchor=(.5, .025), frameon=False)
        fig.subplots_adjust(left=.065 if paired else .09, right=.975, top=.88, bottom=.27, wspace=.23)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{extension}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
    text = (plan["timing"] + "\n\n" + plan["interpretation"] + "\n\n"
            "Residual DFS totals include each root and use identical, separately timed plain MAC. "
            "Censored totals are left blank and omitted from DFS plots. The Sudoku expert has larger "
            "per-cluster limits. Figures omit footer text for poster use; this protocol records exclusions.\n")
    with (out / "RESULTS.txt").open("x") as handle:
        handle.write(text + "\n" + json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--max-jobs", type=int, default=10)
    worker = commands.add_parser("task")
    worker.add_argument("out", type=Path)
    worker.add_argument("position", type=int)
    commands.add_parser("collect").add_argument("out", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "task": task, "collect": collect}[command](**args)
