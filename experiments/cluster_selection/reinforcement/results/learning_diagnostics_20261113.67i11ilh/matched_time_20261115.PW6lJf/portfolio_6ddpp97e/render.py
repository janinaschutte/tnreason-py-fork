"""Presentation-only caption clarification; frozen measurements and plotting checks are unchanged."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shared_time_plot", OUT / "plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
original_caption = plot.caption


def caption(fig, paired=False):
    original_caption(fig, paired)
    text = fig.texts[-1]
    text.set_text(text.get_text().replace(
        "16 reused held-out Sudoku puzzles; selected checkpoints; exploratory results.",
        "16 Sudoku puzzles excluded from training; selected checkpoints; exploratory evaluation."))


if __name__ == "__main__":
    plot.caption = caption
    plot.main()
    names = ("render.py", "plot.py", "protocol.json", "../plot.py", "../protocol.json")
    manifest = {"kind": "presentation_only_caption_clarification",
        "input_sha256": {name: plot.base.digest(OUT / name) for name in names},
        "change": "Caption explicitly states training exclusion; evaluation set was reused for later experiments."}
    with (OUT / "render_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
