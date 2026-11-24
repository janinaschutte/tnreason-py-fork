"""Redraw measured figures with a shorter caption, or none for the poster."""
import argparse
import importlib.util
from pathlib import Path

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("portfolio_figure", OUT / "plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def caption(fig, paired):
    plot.caption(fig, paired)
    text = fig.texts[-1]
    lines = text.get_text().splitlines()
    assert lines[1].startswith("One pinned CPU thread. ")
    text.set_text("\n".join([lines[1].removeprefix("One pinned CPU thread. "), *lines[2:]]))


def main(no_caption=False):
    layouts = (
        (False, (11.4, 8.0), .40, .18, "matched_time_roots_with_mix_short_caption"),
        (True, (16, 8.2), .41, .20, "matched_time_roots_and_dfs_with_mix_short_caption"),
    )
    if no_caption:
        layouts = tuple((*layout[:-1], layout[-1].replace("short_caption", "no_caption"))
                        for layout in layouts)
    assert not any((OUT / f"{stem}.{ext}").exists()
                   for *_, stem in layouts for ext in ("png", "pdf", "svg"))
    plan, rows = plot.load()
    base = plot.base
    base.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 13, "axes.titlesize": 15, "legend.fontsize": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    for paired, size, bottom, legend_y, stem in layouts:
        fig, axes = base.plt.subplots(1, 2 if paired else 1, figsize=size, squeeze=False)
        left = axes[0, 0]
        base.root_curves(left, plan, rows)
        if paired:
            base.dfs_points(axes[0, 1], plan, rows)
            left.set_title("Solved by propagation alone", pad=16)
            axes[0, 1].set_title("Search remaining after propagation", pad=16)
        else:
            left.set_title("Solving Sudoku without branching at matched propagation time", pad=18)
        handles, labels = left.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, legend_y),
                   frameon=False, columnspacing=2 if paired else 1.5, handlelength=2.7)
        fig.subplots_adjust(left=.065 if paired else .09, right=.98 if paired else .975,
                            top=.89, bottom=bottom, wspace=.22)
        if no_caption:
            for extension in ("png", "pdf", "svg"):
                fig.savefig(OUT / f"{stem}.{extension}", dpi=300, facecolor="white",
                            bbox_inches="tight", pad_inches=.12,
                            metadata={"Title": "Matched-time Sudoku cluster propagation"})
            base.plt.close(fig)
        else:
            caption(fig, paired)
            base.export(fig, stem)
        print(stem, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-caption", action="store_true")
    main(**vars(parser.parse_args()))
