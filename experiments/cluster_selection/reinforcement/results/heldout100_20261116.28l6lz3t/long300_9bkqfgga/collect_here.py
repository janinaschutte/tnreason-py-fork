"""Collect frozen Bernoulli results through their local shared-filesystem paths.

Only filesystem reads and hashes translate the repository prefix. Recorded
protocols, result bytes, source files and collector calculations stay unchanged.
"""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
plan = json.loads((OUT / "protocol.json").read_text())
REMOTE = Path(plan["data"]).parents[5]
spec = importlib.util.spec_from_file_location("frozen_long_budget_collector", OUT.parent / "long_budget.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def here(path):
    path = Path(path)
    return ROOT / path.relative_to(REMOTE) if path.is_relative_to(REMOTE) else path


if __name__ == "__main__":
    read, digest = collector.engine.read, collector.engine.digest
    collector.engine.read = lambda path: read(here(path))
    collector.engine.digest = lambda path: digest(here(path))
    collector.collect(OUT)
    for stem in ("matched_time_roots_no_caption", "matched_time_roots_and_dfs_no_caption"):
        path = OUT / f"{stem}.svg"
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
