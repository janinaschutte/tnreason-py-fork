"""Plot the saved structural allowance sweep without rerunning any solver."""
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
source = folder.parent / "allowance_plateau_20261030.bih66zeb"
protocol = json.loads((source / "protocol.json").read_text())
extension = json.loads((source / "extension.json").read_text())
summary = json.loads((source / "summary.json").read_text())
allowances = protocol["allowances"] + extension["allowances"]
indices = protocol["indices"]
records, hashes = {}, {}
for allowance in allowances:
    path = source / f"heuristic_{allowance}.jsonl"
    hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    solves = [row for row in rows if row.get("type") == "solve"]
    assert [row["index"] for row in solves] == indices
    for row in solves:
        assert row["allowance"] == allowance
        assert row["status"] in ("solved", "wall_timeout")
        assert (row["nodes"] is None) == (row["status"] == "wall_timeout")
        if row["status"] == "solved":
            assert row["valid"] and row["nodes"] >= 1
        assert summary["results"][str(allowance)]["nodes_by_puzzle"][str(row["index"])] == row["nodes"]
        records[row["index"], allowance] = row
    completed = [row for row in solves if row["status"] == "solved"]
    total = sum(row["nodes"] for row in completed) if len(completed) == len(indices) else None
    assert total == summary["results"][str(allowance)]["total_nodes"]

order = sorted(indices, key=lambda i: (
    -max(records[i, a]["nodes"] or 0 for a in allowances), i))
values = np.array([[records[i, a]["nodes"] or np.nan for a in allowances] for i in order])
timeouts = [row for row in records.values() if row["status"] == "wall_timeout"]
assert len(timeouts) == 1 and (timeouts[0]["index"], timeouts[0]["allowance"]) == (113, 1024)

with (folder / "nodes.csv").open("x", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["puzzle_index", "allowance", "status", "dfs_nodes", "solve_seconds"])
    for i in indices:
        for a in allowances:
            row = records[i, a]
            writer.writerow([i, a, row["status"], row["nodes"], row["solve_s"]])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False,
                     "axes.spines.bottom": False})
fig, ax = plt.subplots(figsize=(8.5, 10.5))
fig.subplots_adjust(left=0.13, right=0.82, bottom=0.12, top=0.86)
norm = LogNorm(vmin=1, vmax=float(np.nanmax(values)))
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#ececec")
image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(allowances)), labels=[f"{a:,}" for a in allowances])
ax.set_yticks(range(len(order)), labels=order)
ax.tick_params(axis="both", which="both", length=0, pad=8)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("Cluster allowance per propagation episode", labelpad=12)
ax.set_ylabel("Sudoku index", labelpad=12)
ax.set_xticks(np.arange(-0.5, len(allowances), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.4)
for r, i in enumerate(order):
    for c, a in enumerate(allowances):
        row = records[i, a]
        if row["nodes"] is None:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  facecolor="#e6e6e6", edgecolor="#8c8c8c",
                                  hatch="///", linewidth=0))
            ax.text(c, r, "Timeout", ha="center", va="center", fontsize=10,
                    bbox={"facecolor": "#e6e6e6", "edgecolor": "none", "pad": 1})
        else:
            ax.text(c, r, str(row["nodes"]), ha="center", va="center",
                    color="white" if norm(row["nodes"]) > 0.55 else "#17212b")
for label in ax.get_yticklabels():
    if label.get_text() == "113":
        label.set_fontweight("bold")
cax = fig.add_axes([0.85, 0.38, 0.022, 0.37])
colorbar = fig.colorbar(image, cax=cax, ticks=[1, 2, 5, 10, 20, 50, 100, 136])
colorbar.ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "136"])
colorbar.ax.minorticks_off()
colorbar.set_label("DFS nodes (log colour scale)", labelpad=10)
fig.suptitle("DFS nodes for each Sudoku", x=0.13, y=0.965, ha="left", fontsize=18)
fig.text(0.13, 0.93, "Structural heuristic · 23 validation puzzles · saved 30 October sweep", fontsize=11)
fig.text(0.13, 0.065, "1 node = solved without branching. Rows ordered by largest completed node count.", fontsize=10)
fig.text(0.13, 0.039, "Timeout = 600 s for the whole solve; its DFS count is unavailable.", fontsize=10)
for suffix in ("png", "pdf"):
    with (folder / f"dfs_nodes_by_allowance.{suffix}").open("xb") as stream:
        fig.savefig(stream, format=suffix, dpi=180, facecolor="white")
plt.close(fig)

report = [
    "Per-puzzle DFS nodes across structural cluster allowances",
    "",
    f"Source: {source.name}",
    "Uses only the existing 23-puzzle validation sweep; no solver runs or training.",
    "These are the previous sweep's puzzle indices, not the newer 55-puzzle evaluation.",
    "The sweep uses nested structural proposal pools and the original branching rule.",
    "It does not use the newly added oldest-pruned tie-break.",
    "1 DFS node includes the root and means solved without branching.",
    "Rows are sorted by maximum available completed node count, descending, then index.",
    "Timeouts have no numeric node value and are excluded from the colour scale.",
    "",
    f"Guards: {protocol['guards']['seconds']} seconds per complete solve; "
    f"{protocol['guards']['nodes']} DFS nodes, independently.",
    "The wall timer includes cluster generation, contraction, GAC and DFS.",
    "It is set once around attachment and solve; network/state setup is measured separately.",
    "The timeout handler does not save the partial DFS count.",
    "",
    "Allowance | All 23 total | Puzzle 113 nodes | Puzzle 113 seconds | Other 22 total",
]
for a in allowances:
    row = records[113, a]
    total = summary["results"][str(a)]["total_nodes"]
    rest = sum(records[i, a]["nodes"] for i in indices if i != 113)
    report.append(f"{a} | {total if total is not None else 'incomplete'} | "
                  f"{row['nodes'] if row['nodes'] is not None else 'timeout'} | "
                  f"{row['solve_s']:.3f} | {rest}")
report += ["", "Source JSONL SHA-256:"] + [f"{name}: {digest}" for name, digest in hashes.items()]
with (folder / "README.txt").open("x") as stream:
    stream.write("\n".join(report) + "\n")
print("\n".join(report[:22]))
print(f"Validated {len(records)} source solve rows; wrote figures and nodes.csv to {folder}")
