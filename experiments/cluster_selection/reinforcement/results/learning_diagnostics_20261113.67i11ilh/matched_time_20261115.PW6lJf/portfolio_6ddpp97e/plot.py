"""Append a measured shared-budget portfolio to the frozen matched-time comparison."""
import csv
import importlib.util
from pathlib import Path

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_matched_time_plot", OUT.parent / "plot.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MIX_STYLE = {"combined": ("Generic mix (shared time)", "#000000", "-", "*")}


def load():
    parent, rows = base.load()
    original_styles = base.STYLES
    base.OUT, base.STYLES = OUT, MIX_STYLE
    plan, mixed = base.load()
    assert plan["parent_protocol"] == str(OUT.parent / "protocol.json")
    assert plan["parent_protocol_sha256"] == base.digest(OUT.parent / "protocol.json")
    assert plan["kind"] == "matched_time_shared_generic_portfolio"
    assert plan["sources"] == ["heuristic", "teacher", "reinforce", "replay", "q"]
    for field in ("data", "data_sha256", "positions", "budgets_s", "seconds", "cpu", "workers",
                  "checkpoints", "reference_assignments", "limits", "compact_domains", "environment",
                  "contraction_cap", "cumulative_work_cap", "residual_dfs"):
        assert plan[field] == parent[field], f"Changed comparison condition: {field}"
    hashes = {**plan["source_sha256"], **plan["reference_sha256"],
              plan["data"]: plan["data_sha256"]}
    hashes.update({info["path"]: info["sha256"] for info in plan["checkpoints"].values()})
    assert all(base.digest(Path(path)) == expected for path, expected in hashes.items())
    for row, _ in mixed["combined"]:
        assert set(row["source_totals"]) == set(plan["sources"])
        charged = sum(values["charged_s"] for values in row["source_totals"].values())
        assert abs(charged + row["unattributed_overhead_s"] - row["wall_s"]) < 1e-7
        assert row["unattributed_overhead_s"] >= -1e-8
        for source in plan["sources"]:
            events = [event for event in row["source_events"] if event["source"] == source]
            assert row["source_totals"][source]["turns"] == len(events)
            assert row["source_totals"][source]["useful"] == sum(
                proof["source"] == source for proof in row["proofs"])
            assert all(0 <= event["start_s"] <= event["end_s"] <= row["wall_s"] + 1e-7
                       for event in events)
            assert abs(sum(event["charged_s"] for event in events)
                       - row["source_totals"][source]["charged_s"]) < 1e-7
    base.STYLES = {**original_styles, **MIX_STYLE}
    return plan, {**rows, **mixed}


def write_csv(path, records):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def caption(fig, paired=False):
    note = ("16 reused held-out Sudoku puzzles; selected checkpoints; exploratory results.\n"
            "One pinned CPU thread. Initial GAC included; model/network loading excluded.\n"
            "Generic mix shares one budget across heuristic, imitation, REINFORCE, replay and Q-learning.\n"
            "Time axis: linear to 1 s, logarithmic thereafter. "
            "*Sudoku-specific reference with larger per-cluster limits.")
    if paired:
        note += "\nResidual DFS is plain MAC, timed separately; each root counts as one node."
    fig.text(.5, .025, note, ha="center", va="bottom", fontsize=12, linespacing=1.4)


def main():
    stems = ("matched_time_roots_with_mix", "matched_time_roots_and_dfs_with_mix")
    names = [f"{stem}.{extension}" for stem in stems for extension in ("png", "pdf", "svg")]
    names += ["matched_time_totals.csv", "matched_time_per_puzzle.csv", "matched_time_solve_events.csv", "RESULTS.txt"]
    assert not any((OUT / name).exists() for name in names), "Refusing to overwrite exported results"
    plan, rows = load()
    base.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
        "axes.titlesize": 15, "legend.fontsize": 12, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = base.plt.subplots(figsize=(11.4, 8.0))
    base.root_curves(ax, plan, rows)
    ax.set_title("Solving Sudoku without branching at matched propagation time", pad=18)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, .18), frameon=False,
               columnspacing=1.5, handlelength=2.7)
    fig.subplots_adjust(left=.09, right=.975, top=.89, bottom=.40)
    caption(fig)
    base.export(fig, stems[0])

    fig, axes = base.plt.subplots(1, 2, figsize=(16, 8.2))
    base.root_curves(axes[0], plan, rows)
    base.dfs_points(axes[1], plan, rows)
    axes[0].set_title("Solved by propagation alone", pad=16)
    axes[1].set_title("Search remaining after propagation", pad=16)
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, .20), frameon=False,
               columnspacing=2, handlelength=2.7)
    fig.subplots_adjust(left=.065, right=.98, top=.89, bottom=.41, wspace=.22)
    caption(fig, paired=True)
    base.export(fig, stems[1])

    totals, per_puzzle, events = [], [], []
    for method, (label, *_style) in base.STYLES.items():
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
        write_csv(OUT / f"matched_time_{name}.csv", records)
    assert len(totals) == 36 and len(per_puzzle) == 576 and len(events) == 144

    lines = ["MATCHED-TIME PROPAGATION INCLUDING A SHARED-BUDGET GENERIC MIX", "", plan["timing"], "",
        plan["interpretation"], "", "Budget (seconds):                 1      5     15     60", "",
        "Solved without branching (of 16):"]
    for field, heading in (("solved_without_branching", None),
                           ("remaining_dfs_nodes", "Remaining DFS nodes (root included; identical plain MAC):")):
        if heading:
            lines += ["", heading]
        for method, (label, *_style) in base.STYLES.items():
            values = [record[field] for record in totals if record["method"] == method]
            lines.append(f"{label:31}" + "".join(f"{value:7d}" for value in values))
    lines += ["", "All root rows and residual solves pass the original matched-time plotting assertions.",
        "Parent protocol, shared source/data/checkpoint/reference hashes, limits, hardware and budget match.",
        "All residual solves completed with valid assignments; no censored DFS totals.",
        "Generic mix is an actual shared-domain trajectory under one total time budget, not a union of independent runs.",
        "Source turns target equal cumulative charged wall time; nonpreemptive turns can overshoot their ideal share.",
        "Root curves are completed closure events; DFS points are measured only at the four declared budgets.",
        "Prior eight curves are reused unchanged; this follow-up remains exploratory on the reused held-out set."]
    with (OUT / "RESULTS.txt").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
