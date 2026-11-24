"""Validate and plot the completed paired sweep; never run or train a solver."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np


folder = Path(__file__).resolve().parent
read = lambda path: json.loads(path.read_text())
digest = lambda path: sha256(path.read_bytes()).hexdigest()
protocol = read(folder / "protocol.json")
indices, allowances = protocol["indices"], protocol["allowances"]
ties = ("name", "oldest_pruned")
labels = {"name": "Lexicographical", "oldest_pruned": "Oldest-pruned"}
records, summaries, hashes = {}, {}, {"protocol.json": digest(folder / "protocol.json"),
                                     "plot.py": digest(Path(__file__))}
for tie in ties:
    for allowance in allowances:
        path = folder / f"{tie}_{allowance}.json"
        summary = summaries[tie, allowance] = read(path)
        hashes[path.name] = digest(path)
        for puzzle in indices:
            path = folder / "rows" / f"{tie}_{allowance}_{puzzle}.json"
            row = records[tie, puzzle, allowance] = read(path)
            hashes[str(path.relative_to(folder))] = digest(path)
            assert (row["tie"], row["puzzle"], row["allowance"]) == (tie, puzzle, allowance)
            assert row["protocol_sha256"] == hashes["protocol.json"]
            assert row["status"] in ("solved", "wall_timeout", "node_limit")
            assert row["solved"] == (row["status"] == "solved")
            if row["solved"]:
                assert row["valid"] and isinstance(row["nodes"], int) and row["nodes"] >= 1
            assert summary["nodes_by_puzzle"][str(puzzle)] == row["nodes"]
        rows = [records[tie, i, allowance] for i in indices]
        assert summary["completed"] == sum(row["solved"] for row in rows)
        assert summary["nodes"] == (sum(row["nodes"] for row in rows)
                                    if all(row["solved"] for row in rows) else None)
        assert summary["branch_free"] == sum(row["solved"] and row["nodes"] == 1 for row in rows)

order = sorted(indices, key=lambda i: (-max(
    (records[t, i, a]["nodes"] for t in ties for a in allowances
     if records[t, i, a]["solved"]), default=0), i))
maximum = max(row["nodes"] for row in records.values() if row["solved"])
norm = LogNorm(vmin=1, vmax=max(2, maximum))
comparison = {"protocol_sha256": hashes["protocol.json"], "row_order": order,
              "source_sha256": hashes, "allowances": {}}
for a in allowances:
    paired = [i for i in indices if all(records[t, i, a]["solved"] for t in ties)]
    delta = {i: records["oldest_pruned", i, a]["nodes"] - records["name", i, a]["nodes"]
             for i in paired}
    comparison["allowances"][a] = dict(
        totals={t: summaries[t, a]["nodes"] for t in ties},
        completed={t: summaries[t, a]["completed"] for t in ties},
        branch_free={t: summaries[t, a]["branch_free"] for t in ties},
        solved_pairs=len(paired), wins=sum(d < 0 for d in delta.values()),
        ties=sum(d == 0 for d in delta.values()), losses=sum(d > 0 for d in delta.values()),
        paired_node_delta=sum(delta.values()), paired_deltas=delta,
        censored={t: [{"puzzle": i, "status": records[t, i, a]["status"]}
                      for i in indices if not records[t, i, a]["solved"]] for t in ties},
        sensitivity_puzzle_113={t: dict(
            puzzle={k: records[t, 113, a][k] for k in ("nodes", "status", "solve_s")},
            other_22_nodes=(sum(records[t, i, a]["nodes"] for i in indices if i != 113)
                if all(records[t, i, a]["solved"] for i in indices if i != 113) else None)) for t in ties},
        historical_name_mismatches=summaries["name", a]["historical_node_mismatches"])

outputs = ["nodes.csv", "comparison.json", "REPORT.txt"] + [f"{stem}.{suffix}"
    for stem in ("dfs_nodes_oldest_pruned", "dfs_nodes_branching_comparison") for suffix in ("png", "pdf")]
for name in outputs:
    if (folder / name).exists():
        raise FileExistsError(folder / name)
with (folder / "nodes.csv").open("x", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["puzzle_index", "allowance", "branching", "status", "dfs_nodes",
                     "partial_dfs_nodes", "solve_seconds"])
    for i in indices:
        for a in allowances:
            for t in ties:
                r = records[t, i, a]
                writer.writerow([i, a, t, r["status"], r["nodes"] if r["solved"] else None,
                                 None if r["solved"] else r["nodes"], r["solve_s"]])
with (folder / "comparison.json").open("x") as stream:
    json.dump(comparison, stream, indent=2, allow_nan=False)
    stream.write("\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#ececec")
for selected, stem in [(("oldest_pruned",), "dfs_nodes_oldest_pruned"),
                       (ties, "dfs_nodes_branching_comparison")]:
    paired = len(selected) == 2
    fig, axes = plt.subplots(1, len(selected), figsize=(15 if paired else 8.5, 10.5), squeeze=False)
    fig.subplots_adjust(left=.07 if paired else .13, right=.9 if paired else .82,
                        bottom=.12, top=.81, wspace=.15)
    for ax, t in zip(axes[0], selected):
        values = np.array([[records[t, i, a]["nodes"] if records[t, i, a]["solved"] else np.nan
                            for a in allowances] for i in order])
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(allowances)), labels=[f"{a:,}" for a in allowances])
        ax.set_yticks(range(len(order)), labels=order)
        ax.tick_params(axis="both", which="both", length=0, pad=7)
        ax.xaxis.tick_top()
        ax.set_title(labels[t], pad=27, fontsize=13, fontweight="bold")
        ax.set_ylabel("Sudoku index")
        ax.set_xticks(np.arange(-.5, len(allowances), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r, i in enumerate(order):
            for c, a in enumerate(allowances):
                row = records[t, i, a]
                if not row["solved"]:
                    ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, facecolor="#e6e6e6",
                                          edgecolor="#8c8c8c", hatch="///", linewidth=0))
                text = str(row["nodes"]) if row["solved"] else (
                    "Timeout" if row["status"] == "wall_timeout" else "Node cap")
                ax.text(c, r, text, ha="center", va="center", fontsize=10,
                        color="white" if row["solved"] and norm(row["nodes"]) > .55 else "#17212b")
    ticks = [n for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if n <= maximum]
    cax = fig.add_axes([.93 if paired else .85, .38, .014 if paired else .022, .37])
    cb = fig.colorbar(image, cax=cax, ticks=ticks)
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([str(n) for n in ticks])
    cb.set_label("DFS nodes (shared log colour scale)", labelpad=10)
    left = .07 if paired else .13
    fig.suptitle("DFS nodes for each Sudoku", x=left, y=.965, ha="left", fontsize=18)
    fig.text(left, .925, "Structural heuristic · 23 validation puzzles · fresh paired runs", fontsize=11)
    fig.text(left, .89, "Columns: cluster allowance per propagation episode", fontsize=11)
    fig.text(left, .065, "1 node = solved without branching. Identical row order and colour scale in both figures.", fontsize=10)
    fig.text(left, .039, f"Censored cells have no final node count; whole-solve timeout = {protocol['guards']['seconds']} s.", fontsize=10)
    for suffix in ("png", "pdf"):
        with (folder / f"{stem}.{suffix}").open("xb") as stream:
            fig.savefig(stream, format=suffix, dpi=180, facecolor="white")
    plt.close(fig)

report = ["Paired structural allowance sweep with oldest-pruned branching", "",
    "23 original validation Sudoku indices; no retraining and no final-test queries.",
    "Both rules: smallest domain, highest incident-core degree. Then name, or oldest retained pruning then name.",
    "Never-pruned variables are oldest. Backtracking restores history; temporary summaries retain only pruning.",
    "Same nested structural proposal generator and contraction limits in both arms; no total-work limit.",
    f"Whole-solve wall guard: {protocol['guards']['seconds']} s; independent DFS node guard: {protocol['guards']['nodes']}.",
    "Wall time includes cluster generation, contraction, GAC and DFS. Censored final node counts are unavailable.",
    "A complete solve counts its root: 1 node means no branching. Totals are absent if any solve is censored.",
    "Rows sorted by largest completed node count across both rules and all allowances, then puzzle index.",
    "Puzzle 113 was identified as an outlier in the original sweep; sensitivity totals retain all other 22 puzzles.",
    "Wins/ties/losses and delta compare only solved pairs; delta is oldest-pruned minus name.", "",
    "Allowance | Name nodes | Oldest nodes | Branch-free name/oldest | Pairs | Wins/ties/losses | Paired delta"]
for a, c in comparison["allowances"].items():
    totals = [c["totals"][t] if c["totals"][t] is not None else "incomplete" for t in ties]
    report.append(f"{a} | {totals[0]} | {totals[1]} | "
                  f"{c['branch_free']['name']}/{c['branch_free']['oldest_pruned']} | {c['solved_pairs']} | "
                  f"{c['wins']}/{c['ties']}/{c['losses']} | {c['paired_node_delta']:+d}")
    report.append(f"  Censored: {c['censored']}; name differs from historical node/status results: {c['historical_name_mismatches']}")
    for t, sensitivity in c["sensitivity_puzzle_113"].items():
        r = sensitivity["puzzle"]
        report.append(f"  {t}: puzzle 113 {r['status']}, {r['nodes']} nodes, {r['solve_s']:.3f}s; "
                      f"other 22 nodes: {sensitivity['other_22_nodes']}")
report += ["", f"Validated {len(records)} solve rows and {len(summaries)} summaries.",
           "Every aggregate improvement at allowances 32, 64, 128 and 512 depends on puzzle 113.",
           "Across the other 22 puzzles, oldest-pruned uses more total nodes at every allowance.",
           "At allowance 256, oldest-pruned is worse overall: 60 versus 52 nodes.",
           "The best new total is 51 at allowance 1024; the best name total is 52 at allowance 256.",
           "Those best totals use different allowances; this is exploratory validation, not a final-test result.",
           "The solver's default branching rule remains name-based; this sweep makes no production changes.",
           "Full source hashes, per-puzzle paired deltas and censoring lists: comparison.json.",
           "CSV leaves censored dfs_nodes blank; partial_dfs_nodes is observed work, not a completed solve count."]
with (folder / "REPORT.txt").open("x") as stream:
    stream.write("\n".join(report) + "\n")
print("\n".join(report))
