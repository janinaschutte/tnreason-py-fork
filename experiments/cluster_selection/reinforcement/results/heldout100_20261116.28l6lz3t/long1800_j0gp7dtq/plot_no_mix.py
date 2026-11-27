"""Export the footer-free, no-mix variant from the audited completed-run CSV."""
import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
STYLE = OUT.parents[1] / "learning_diagnostics_20261113.67i11ilh/matched_time_20261115.PW6lJf/plot.py"
spec = importlib.util.spec_from_file_location("matched_time_styles", STYLE)
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)


def main():
    plan = json.loads((OUT / "protocol.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["protocol_sha256"] == sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    seconds, budgets = plan["seconds"], plan["budgets_s"]
    with (OUT / "matched_time_per_puzzle.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["budget_s"]) == seconds]
    assert len(rows) == len(plan["methods"]) * len(plan["positions"])
    stem = OUT / "matched_time_roots_without_mix_no_caption"
    targets = [stem.with_suffix("." + ext) for ext in ("png", "pdf", "svg")]
    receipt = OUT / "no_mix_plot.json"
    assert not any(path.exists() for path in [*targets, receipt]), "Preserve existing exports"
    plt = drawing.plt
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 12,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.6, 7.1))
    for method, (label, color, style, marker) in drawing.STYLES.items():
        group = [row for row in rows if row["method"] == method]
        assert {int(row["position"]) for row in group} == set(plan["positions"])
        assert len(group) == len(plan["positions"])
        times = sorted(float(row["solved_s"]) for row in group if int(row["solved_without_branching"]))
        expected = next(row for row in summary["totals"] if row["method"] == method and row["budget_s"] == seconds)
        assert len(times) == expected["solved_without_branching"] and all(0 <= time <= seconds for time in times)
        ax.step([0, *times, seconds], [0, *range(1, len(times) + 1), len(times)], where="post",
                label=label, color=color, linestyle=style, linewidth=2.1)
        ax.scatter(budgets, [sum(time <= budget for time in times) for budget in budgets],
                   color=color, marker=marker, s=34, zorder=4, clip_on=False)
    ax.set_xscale("symlog", linthresh=1, linscale=.65)
    ax.set_xlim(0, seconds)
    ax.set_xticks([0, *budgets], [str(budget) for budget in (0, *budgets)])
    ax.tick_params(axis="x", labelsize=10)
    ax.minorticks_off()
    ax.set_xlabel("Root propagation time allowed per puzzle (seconds)")
    ax.set_ylabel("Puzzles solved without branching")
    ax.set_title("Solving 100 held-out Sudokus without branching", pad=18)
    ax.set_ylim(-1, 103)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", color=".87", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), ncol=3, loc="lower center",
               bbox_to_anchor=(.5, .025), frameon=False)
    fig.subplots_adjust(left=.09, right=.975, top=.88, bottom=.27)
    for path in targets:
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.12)
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    inputs = [OUT / name for name in ("protocol.json", "summary.json", "matched_time_per_puzzle.csv")]
    receipt.write_text(json.dumps(dict(methods=list(drawing.STYLES), excluded=["combined"], footer=False,
        inputs={path.name: sha256(path.read_bytes()).hexdigest() for path in inputs},
        style_sha256=sha256(STYLE.read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={path.name: sha256(path.read_bytes()).hexdigest() for path in targets}), indent=2) + "\n")
    print(stem)


if __name__ == "__main__":
    main()
