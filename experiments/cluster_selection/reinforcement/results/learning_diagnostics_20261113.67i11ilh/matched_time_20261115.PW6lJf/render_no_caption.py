"""Export the original eight measured curves without poster footer text."""
import importlib.util
from pathlib import Path

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("measured_figure", OUT / "plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def main():
    stems = ("matched_time_roots_no_caption", "matched_time_roots_and_dfs_no_caption")
    assert not any((OUT / f"{stem}.{ext}").exists()
                   for stem in stems for ext in ("png", "pdf", "svg"))
    plan, rows = plot.load()
    plot.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 13, "axes.titlesize": 15, "legend.fontsize": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    for paired, stem in enumerate(stems):
        fig, axes = plot.plt.subplots(1, 2 if paired else 1,
            figsize=(16, 7.2) if paired else (10.8, 6.9), squeeze=False)
        left = axes[0, 0]
        plot.root_curves(left, plan, rows)
        if paired:
            plot.dfs_points(axes[0, 1], plan, rows)
            left.set_title("Solved by propagation alone", pad=16)
            axes[0, 1].set_title("Search remaining after propagation", pad=16)
        else:
            left.set_title("Solving Sudoku without branching at matched propagation time", pad=18)
        handles, labels = left.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="lower center",
            bbox_to_anchor=(.5, .16 if paired else .15), frameon=False,
            columnspacing=2 if paired else 1.4, handlelength=2.7)
        fig.subplots_adjust(left=.065 if paired else .09, right=.98 if paired else .975,
            top=.89 if paired else .88, bottom=.33 if paired else .32, wspace=.22)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"{stem}.{extension}", dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=.12,
                metadata={"Title": "Matched-time Sudoku cluster propagation"})
        plot.plt.close(fig)
        print(stem, flush=True)


if __name__ == "__main__":
    main()
