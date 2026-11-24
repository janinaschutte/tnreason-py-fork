"""Extend the frozen paired CPU experiment without changing its solver methods.

One trajectory per method/puzzle records the first valid solve time. Saved
domains supply separately timed residual DFS at the declared time checkpoints.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
from random import Random
import sys
from tempfile import mkdtemp
import importlib.util


BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_long_budget", BASE / "long_budget.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
engine, ROOT = frozen.engine, frozen.ROOT
METHODS, SOURCES = frozen.METHODS, frozen.SOURCES
CHECKPOINTS = (1, 5, 15, 60, 120, 300, 600, 1200)


def prepare(seconds, max_jobs):
    assert seconds > 0 and max_jobs > 0
    plan = deepcopy(engine.read(BASE / "protocol.json"))
    previous_root = Path(plan["data"]).parents[5]

    def relocated(path):
        return str(ROOT / Path(path).relative_to(previous_root))

    for field in ("source_sha256", "reference_sha256"):
        plan[field] = {relocated(path): value for path, value in plan[field].items()}
    plan["data"] = relocated(plan["data"])
    plan["reference_assignments"] = {key: relocated(path) for key, path in plan["reference_assignments"].items()}
    for checkpoint in plan["checkpoints"].values():
        checkpoint["path"] = relocated(checkpoint["path"])
    budgets = [b for b in CHECKPOINTS if b < seconds] + [seconds]
    order = [(method, position) for position in plan["positions"] for method in METHODS]
    Random(frozen.SEED).shuffle(order)
    plan.update(kind="paired_100_extended_time_generic_portfolio", methods=list(METHODS),
        sources=list(SOURCES), budgets_s=budgets, seconds=seconds, order=order, seed=frozen.SEED,
        cpu=None, workers=max_jobs, parent_protocol=str(BASE / "protocol.json"),
        parent_protocol_sha256=engine.digest(BASE / "protocol.json"),
        scheduling="least cumulative charged source wall time; one candidate turn",
        environment={"recorded_per_attempt": True, "torch_threads": 1, "blas_threads": 1})
    assert plan["positions"] == list(range(100))
    plan.pop("cpu_topology", None)
    plan.pop("original_affinity", None)
    plan["timing"] = plan["timing"].replace("One serial pinned CPU process;", "One single-thread CPU process per puzzle;")
    plan["interpretation"] = (
        f"Exploratory rerun on the same 100 previously evaluated puzzles, with fixed checkpoints and "
        f"per-cluster limits. Each method/puzzle runs once for up to {seconds}s, stopping if solved. "
        f"First valid solve times determine the continuous root-solve curve at any time threshold; "
        f"saved-domain prefixes {budgets} supply separately timed plain-MAC DFS. No trajectory is "
        "resumed or spliced from older runs. The generic mix shares one aggregate wall budget "
        "across heuristic/teacher/REINFORCE/replay/Q, with equal cumulative wall shares targeted "
        "by nonpreemptive one-candidate turns. All sources restart on updated domains after pruning. "
        "Switching, proposals, planning, contraction and GAC count. The Sudoku expert stays a separate "
        "reference with larger per-cluster limits; model/network/state loading is excluded.")
    plan["execution"] = (
        "One single-thread CPU job per puzzle; all nine methods run sequentially on the same worker "
        "in seeded random order. A restarted job reruns the entire puzzle in a fresh attempt. Only "
        "a complete attempt publishes a completion marker. Hardware is recorded per attempt.")
    for path in (Path(__file__).resolve(), BASE / "long_budget.py", BASE / "cluster.py",
                 BASE / "protocol.json", frozen.PORTFOLIO, frozen.OLD / "plot.py"):
        plan["source_sha256"][str(path)] = engine.digest(path)
    frozen.cluster.verify(plan)
    out = Path(mkdtemp(prefix=f"long{seconds}_", dir=BASE))
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
environment = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cluster-mpl-$(ClusterID)-$(Process) XDG_CACHE_HOME=/tmp/cluster-cache-$(ClusterID)-$(Process)"
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


def collect(out):
    plan = engine.read(out / "protocol.json")
    previous_root = Path(plan["data"]).parents[5]

    def here(path):
        path = Path(path)
        return ROOT / path.relative_to(previous_root) if path.is_relative_to(previous_root) else path

    def read(path):
        return engine.read(here(path))

    def digest(path):
        return engine.digest(here(path))

    hashes = {**plan["source_sha256"], **plan["reference_sha256"], plan["data"]: plan["data_sha256"]}
    hashes.update({c["path"]: c["sha256"] for c in plan["checkpoints"].values()})
    assert all(digest(p) == value for p, value in hashes.items()), "Frozen input changed"
    assert plan["methods"] == list(METHODS) and plan["positions"] == list(range(100))
    budgets, seconds = plan["budgets_s"], plan["seconds"]
    assert budgets == sorted(set(budgets)) and 0 < budgets[0] and budgets[-1] == seconds
    stems = ("matched_time_roots_no_caption", "matched_time_roots_and_dfs_no_caption")
    outputs = [f"{stem}.{ext}" for stem in stems for ext in ("png", "pdf", "svg")]
    outputs += ["matched_time_totals.csv", "matched_time_per_puzzle.csv", "summary.json", "RESULTS.txt"]
    assert not any((out / name).exists() for name in outputs), "Preserve existing exports"
    groups, details = defaultdict(list), []
    protocol_hash = digest(out / "protocol.json")
    for position in plan["positions"]:
        marker = read(out / "done" / f"{position:02d}.json")
        assert marker["position"] == position and marker["protocol_sha256"] == protocol_hash
        attempt = here(marker["attempt"])
        assert digest(attempt / "protocol.json") == protocol_hash
        worker = read(attempt / "worker.json")
        assert digest(attempt / "worker.json") == marker["worker_sha256"]
        assert set(marker["row_sha256"]) == {f"{m}_{position:02d}{s}" for m in METHODS for s in (".json", ".dfs.json")}
        assert all(digest(attempt / "rows" / name) == value for name, value in marker["row_sha256"].items())
        for method in METHODS:
            path = attempt / "rows" / f"{method}_{position:02d}.json"
            row, residual = read(path), read(path.with_suffix(".dfs.json"))
            assert row["protocol_sha256"] == residual["protocol_sha256"] == protocol_hash
            assert row["position"] == residual["position"] == position and row["method"] == residual["method"] == method
            assert row["no_added_cores"] and not row["dfs_called"] and row["reference_retained"]
            assert row["decision_level"] == 0 and row["no_pending_queue"] and residual["timed_separately"]
            assert row["affinity"] == worker["affinity"] and residual["root_row_sha256"] == digest(path)
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
                    events = [e for e in row["source_events"] if e["source"] == source]
                    info = row["source_totals"][source]
                    assert info["turns"] == len(events)
                    assert info["useful"] == sum(p["source"] == source for p in row["proofs"])
                    assert abs(info["charged_s"] - sum(e["charged_s"] for e in events)) < 1e-7
            groups[method].append(row)
            for budget in budgets:
                prefix, finish = row["prefixes"][str(budget)], residual["budgets"][str(budget)]
                eligible = [i for i, s in enumerate(row["snapshots"]) if s["wall_s"] <= budget]
                assert prefix["snapshot"] == finish["snapshot"] == (eligible[-1] if eligible else None)
                solved = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= budget)
                assert prefix["solved"] == solved
                assert not solved or finish["valid"] and finish["solved"] and finish["nodes"] == 1
                details.append(dict(method=method, position=position, budget_s=budget, host=worker["host"],
                    cpu_model=worker["cpu_model"], solved_without_branching=int(solved),
                    remaining_dfs_nodes=finish["nodes"] if finish["valid"] and finish["solved"] else None,
                    residual_status=finish["status"], solved_s=row["solved_s"],
                    trajectory_wall_s=row["wall_s"], trajectory_cpu_s=row["cpu_s"]))
    totals = []
    for method in METHODS:
        for budget in budgets:
            rows = [r for r in details if r["method"] == method and r["budget_s"] == budget]
            assert len(rows) == 100
            censored = sum(r["remaining_dfs_nodes"] is None for r in rows)
            totals.append(dict(method=method, budget_s=budget, instances=100,
                solved_without_branching=sum(r["solved_without_branching"] for r in rows),
                remaining_dfs_nodes=None if censored else sum(r["remaining_dfs_nodes"] for r in rows),
                residual_censored=censored,
                mean_used_root_wall_s=sum(min(r["trajectory_wall_s"], budget) for r in rows) / 100))
    for name, rows in (("totals", totals), ("per_puzzle", details)):
        with (out / f"matched_time_{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    engine.write(out / "summary.json", dict(protocol_sha256=protocol_hash, totals=totals))
    drawing = frozen.module("frozen_plot_styles", frozen.OLD / "plot.py")
    styles = dict(drawing.STYLES, combined=("Generic mix (shared time)", "#000000", "-", "*"))
    plt = drawing.plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 12,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    for paired, stem in zip((False, True), stems):
        fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(17.5, 7.3) if paired else (12.6, 7.1), squeeze=False)
        for index, ax in enumerate(axes.flat):
            ax.set_xscale("symlog", linthresh=1, linscale=.65)
            ax.set_xlim(0, seconds)
            ax.set_xticks([0, *budgets], [str(b) for b in (0, *budgets)])
            ax.tick_params(axis="x", labelsize=10)
            ax.minorticks_off()
            ax.set_xlabel("Root propagation time allowed per puzzle (seconds)")
            ax.grid(axis="y", color=".87", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            for method, (label, color, style, marker) in styles.items():
                if index == 0:
                    times = sorted(r["solved_s"] for r in groups[method]
                        if r["valid"] and r["solved_s"] is not None and r["solved_s"] <= seconds)
                    ax.step([0, *times, seconds], [0, *range(1, len(times) + 1), len(times)], where="post",
                            label=label, color=color, linestyle=style, linewidth=2.1)
                    ax.scatter(budgets, [sum(t <= b for t in times) for b in budgets], color=color,
                               marker=marker, s=34, zorder=4, clip_on=False)
                else:
                    nodes = [r["remaining_dfs_nodes"] for r in totals if r["method"] == method]
                    ax.plot(budgets, [float("nan") if n is None else n for n in nodes],
                            label=label, color=color, marker=marker, linestyle="None", markersize=7, clip_on=False)
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
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.12)
        svg = out / f"{stem}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)
    with (out / "RESULTS.txt").open("x") as handle:
        handle.write(plan["timing"] + "\n\n" + plan["interpretation"] + "\n\n"
            "Residual DFS totals include the root; censored totals stay blank. Figures omit footer text. "
            "Mean used root wall time is capped at each checkpoint; loading and residual DFS are excluded.\n\n"
            + json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--seconds", type=int, default=1800)
    preparation.add_argument("--max-jobs", type=int, default=10)
    worker = commands.add_parser("task")
    worker.add_argument("out", type=Path)
    worker.add_argument("position", type=int)
    commands.add_parser("collect").add_argument("out", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "task": frozen.task, "collect": collect}[command](**args)
