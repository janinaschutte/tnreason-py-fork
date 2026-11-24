"""Run the frozen collector through the local view of Bernoulli's shared files.

Only filesystem reads/hash checks translate the repository prefix. Protocols,
recorded paths, result bytes, source files and collector logic stay unchanged.
"""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
plan = json.loads((OUT / "protocol.json").read_text())
REMOTE = Path(plan["data"]).parents[5]
spec = importlib.util.spec_from_file_location("frozen_remote_collector", OUT.parent / "cluster.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def here(path):
    path = Path(path)
    return ROOT / path.relative_to(REMOTE) if path.is_relative_to(REMOTE) else path


if __name__ == "__main__":
    names = ["matched_time_totals.csv", "matched_time_per_puzzle.csv", "summary.json"]
    names += [f"matched_time_roots_no_caption.{extension}" for extension in ("png", "pdf", "svg")]
    assert not any((OUT / name).exists() for name in names), "Preserve existing exports"
    read, digest = collector.engine.read, collector.engine.digest
    collector.engine.read = lambda path: read(here(path))
    collector.engine.digest = lambda path: digest(here(path))
    collector.collect(OUT)
