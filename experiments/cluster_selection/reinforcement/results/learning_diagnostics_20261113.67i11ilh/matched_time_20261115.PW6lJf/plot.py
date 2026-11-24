"""Export matched-time figures from complete, hash-checked measured rows only."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


OUT = Path(__file__).resolve().parent
STYLES = {
    "plain": ("Plain MAC", "#222222", "--", "+"),
    "heuristic": ("Structural heuristic", "#D55E00", "-", "s"),
    "initial": ("Untrained constructor", "#777777", ":", "v"),
    "teacher": ("Teacher imitation", "#0072B2", "-", "o"),
    "reinforce": ("REINFORCE", "#AA4499", "-.", "^"),
    "replay": ("REINFORCE + replay", "#009E73", "-", "D"),
    "q": ("Q-learning", "#CC79A7", ":", "P"),
    "expert": ("Sudoku expert*", "#882255", "--", "X"),
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load():
    plan = read(OUT / "protocol.json")
    assert set(plan["methods"]) == set(STYLES), "Unexpected comparison arms"
    assert plan["positions"] == list(range(16)) and plan["budgets_s"] == [1, 5, 15, 60]
    assert plan["workers"] == 1 and plan["seconds"] == 60
    protocol_hash = digest(OUT / "protocol.json")
    rows = {}
    for method in STYLES:
        group = []
        for position in plan["positions"]:
            path = OUT / "rows" / f"{method}_{position:02d}.json"
            residual_path = path.with_name(f"{method}_{position:02d}.dfs.json")
            assert path.exists() and residual_path.exists(), f"Incomplete measured data: {path.name}"
            row, residual = read(path), read(residual_path)
            assert row["method"] == residual["method"] == method
            assert row["position"] == residual["position"] == position
            assert row["protocol_sha256"] == residual["protocol_sha256"] == protocol_hash
            assert residual["root_row_sha256"] == digest(path)
            assert row["affinity"] == [plan["cpu"]]
            assert row["no_added_cores"] and not row["dfs_called"] and row["reference_retained"]
            assert row["decision_level"] == 0 and residual["timed_separately"]
            for budget in plan["budgets_s"]:
                key = str(budget)
                eligible = [i for i, snap in enumerate(row["snapshots"]) if snap["wall_s"] <= budget]
                index = eligible[-1] if eligible else None
                prefix, finish = row["prefixes"][key], residual["budgets"][key]
                assert prefix["snapshot"] == finish["snapshot"] == index
                solved = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= budget)
                assert prefix["solved"] == solved
                assert finish["solved"] and finish["valid"] and finish["nodes"] >= 1, (
                    f"Censored residual DFS cannot be summed: {method}, puzzle {position}, {budget} s")
                assert not solved or finish["nodes"] == 1
            if row["solved_s"] is not None:
                assert row["valid"] and row["solved_s"] >= 0
                assert row["solved_s"] == next(s["wall_s"] for s in row["snapshots"] if s["solved"])
            group.append((row, residual))
        rows[method] = group
    return plan, rows


def configure(ax, ylabel):
    ax.set_xscale("symlog", linthresh=1, linscale=.65)
    ax.set_xlim(0, 60)
    ax.xaxis.set_major_locator(FixedLocator([0, 1, 5, 15, 60]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.minorticks_off()
    ax.set_xlabel("Root propagation time allowed per puzzle (seconds)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.87", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)


def root_curves(ax, plan, rows):
    configure(ax, "Puzzles solved without branching")
    ax.set_ylim(-.3, 16.5)
    ax.set_yticks(range(0, 17, 2))
    for method, (label, color, style, marker) in STYLES.items():
        times = sorted(row["solved_s"] for row, _ in rows[method]
                       if row["valid"] and row["solved_s"] is not None and row["solved_s"] <= 60)
        x = [0, *times, 60]
        y = [0, *range(1, len(times)+1), len(times)]
        ax.step(x, y, where="post", label=label, color=color, linestyle=style,
                linewidth=2.1, marker=marker, markevery=[], markersize=6)
        budgets = np.asarray(plan["budgets_s"])
        counts = [sum(t <= budget for t in times) for budget in budgets]
        ax.scatter(budgets, counts, color=color, marker=marker, s=34, zorder=4, clip_on=False)


def dfs_points(ax, plan, rows):
    configure(ax, "Remaining DFS nodes, summed over 16 puzzles")
    for method, (label, color, _, marker) in STYLES.items():
        nodes = [sum(residual["budgets"][str(budget)]["nodes"] for _, residual in rows[method])
                 for budget in plan["budgets_s"]]
        ax.plot(plan["budgets_s"], nodes, label=label, color=color, marker=marker,
                linestyle="None", markersize=7, clip_on=False)
    ax.axhline(16, color=".6", linewidth=1, linestyle=":")
    ax.set_ylim(bottom=14)


def caption(fig, paired=False):
    note = ("16 reused held-out Sudoku puzzles; selected checkpoints; exploratory results.\n"
            "One pinned CPU thread. Initial GAC included; model/network loading excluded.\n"
            "Time axis: linear to 1 s, logarithmic thereafter. "
            "*Sudoku-specific reference with larger per-cluster limits.")
    if paired:
        note += "\nResidual DFS is plain MAC, timed separately; each root counts as one node."
    fig.text(.5, .025, note, ha="center", va="bottom", fontsize=12, linespacing=1.5)


def export(fig, stem):
    for extension in ("png", "pdf", "svg"):
        metadata = {"Title": "Matched-time Sudoku cluster propagation"}
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, facecolor="white", metadata=metadata)
    plt.close(fig)


def write_csv(path, fields, records):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    outputs = [f"{stem}.{extension}" for stem in ("matched_time_roots", "matched_time_roots_and_dfs")
               for extension in ("png", "pdf", "svg")]
    outputs += ["matched_time_totals.csv", "matched_time_per_puzzle.csv", "matched_time_solve_events.csv", "RESULTS.txt"]
    assert not any((OUT / name).exists() for name in outputs), "Refusing to overwrite exported results"
    plan, rows = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
                         "axes.titlesize": 15, "legend.fontsize": 12, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.8, 6.9))
    root_curves(ax, plan, rows)
    ax.set_title("Solving Sudoku without branching at matched propagation time", pad=18)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .15), frameon=False,
               columnspacing=1.4, handlelength=2.7)
    fig.subplots_adjust(left=.09, right=.975, top=.88, bottom=.32)
    caption(fig)
    export(fig, "matched_time_roots")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    root_curves(axes[0], plan, rows)
    dfs_points(axes[1], plan, rows)
    axes[0].set_title("Solved by propagation alone", pad=16)
    axes[1].set_title("Search remaining after propagation", pad=16)
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .16), frameon=False,
               columnspacing=2, handlelength=2.7)
    fig.subplots_adjust(left=.065, right=.98, top=.89, bottom=.33, wspace=.22)
    caption(fig, paired=True)
    export(fig, "matched_time_roots_and_dfs")

    totals, per_puzzle, events = [], [], []
    for method, (label, *_style) in STYLES.items():
        for budget in plan["budgets_s"]:
            values = []
            for row, residual in rows[method]:
                prefix, finish = row["prefixes"][str(budget)], residual["budgets"][str(budget)]
                record = dict(method=method, position=row["position"], budget_s=budget,
                    solved_without_branching=int(prefix["solved"]), remaining_dfs_nodes=finish["nodes"],
                    solved_s=row["solved_s"], snapshot=prefix["snapshot"], residual_s=finish["solve_s"],
                    trajectory_wall_s=row["wall_s"], trajectory_cpu_s=row["cpu_s"], status=row["status"])
                per_puzzle.append(record)
                values.append(record)
            totals.append(dict(method=method, label=label, budget_s=budget, instances=len(values),
                solved_without_branching=sum(r["solved_without_branching"] for r in values),
                remaining_dfs_nodes=sum(r["remaining_dfs_nodes"] for r in values)))
        events.extend(dict(method=method, position=row["position"], solved_s=row["solved_s"],
            solved_cpu_s=row["solved_cpu_s"], counted_by_60_s=int(bool(row["valid"] and
                row["solved_s"] is not None and row["solved_s"] <= 60))) for row, _ in rows[method])
    for name, records in (("totals", totals), ("per_puzzle", per_puzzle), ("solve_events", events)):
        write_csv(OUT / f"matched_time_{name}.csv", list(records[0]), records)

    lines = ["MATCHED-TIME ROOT PROPAGATION", "", plan["timing"], "", plan["interpretation"], "",
             "Budget (seconds):          1      5     15     60", "", "Solved without branching (of 16):"]
    for method, (label, *_style) in STYLES.items():
        values = [r["solved_without_branching"] for r in totals if r["method"] == method]
        lines.append(f"{label:24}" + "".join(f"{value:7d}" for value in values))
    lines += ["", "Remaining DFS nodes (root included; identical plain MAC):"]
    for method, (label, *_style) in STYLES.items():
        values = [r["remaining_dfs_nodes"] for r in totals if r["method"] == method]
        lines.append(f"{label:24}" + "".join(f"{value:7d}" for value in values))
    lines += ["", "All residual solves completed and returned valid assignments; no censored node totals.",
              "The root curve changes only at recorded completed GAC closures; it is not interpolated.",
              "DFS points are measured only at the four declared budgets, with no interpolating lines.",
              "No training, checkpoint selection or hyperparameter change used these outcomes.",
              "Sudoku expert is a domain-specific reference with different, larger per-cluster limits.",
              "This is an exploratory comparison on a previously reused held-out set, not a fresh final test."]
    with (OUT / "RESULTS.txt").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
