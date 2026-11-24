"""Retry an existing censored solve once with an1800-second wall guard.

Example: extend_guards.py nearbest_pool/dfs_1000/puzzle_113.json
Original checkpoints, rows, summaries and protocols remain untouched.
"""
import argparse
from functools import partial
from pathlib import Path
import runpy

OUT = Path(__file__).resolve().parent
base = runpy.run_path(str(OUT / "run.py"))
read, digest, write = (base[key] for key in ("read", "digest", "write"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Original row relative to this experiment directory")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze and validate inputs without solving")
    args = parser.parse_args()
    original = (OUT / args.case).resolve()
    parts = original.relative_to(OUT).parts
    plan = read(OUT / "protocol.json")
    if len(parts) == 3 and parts[0] in base["ARMS"] and parts[1] in ("dfs_1000", "dfs_4000"):
        arm, update = parts[0], int(parts[1].removeprefix("dfs_"))
        checkpoint, mode = OUT / arm / f"policy_{update:04d}.pt", "trained"
    elif len(parts) == 2 and parts[0] in ("baseline_plain", "baseline_heuristic32", "baseline_parent"):
        arm, update = parts[0], None
        checkpoint = Path(plan["parent"])
        mode = {"baseline_plain": "plain", "baseline_heuristic32": "heuristic", "baseline_parent": "trained"}[arm]
    else:
        raise ValueError("Expected an existing arm milestone or baseline solve row")
    previous = read(original)
    puzzle = previous["puzzle"]
    assert parts[-1] == f"puzzle_{puzzle:03d}.json" and puzzle in plan["validation_indices"]
    assert previous["status"] == "wall_timeout" and not previous["solved"] and not previous["valid"]
    assert plan["seconds"] == 600 and plan["nodes"] == 10000 and plan["cluster_attempts"] == 32
    assert plan["decoder"] == "discrepancy" and plan["total_work"] is None
    sources = dict(plan["source_sha256"])
    sources[str(Path(__file__).resolve())] = digest(__file__)
    assert all(digest(path) == expected for path, expected in sources.items())
    protocol = dict(original_protocol=str(OUT / "protocol.json"),
        original_protocol_sha256=digest(OUT / "protocol.json"),
        original_row=str(original), original_row_sha256=digest(original),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        arm=arm, update=update, puzzle=puzzle, mode=mode, source_sha256=sources,
        original_seconds=600, seconds=1800, nodes=10000, total_work=None,
        cluster_attempts=32, baseline_candidates=32, baseline_steps=8192,
        stop_on_noop=False, decoder="discrepancy", beta=0., branching="name",
        note="Prespecified wall-time extension of an existing censor; no checkpoint selection or tuning. "
             "Only the wall guard changes. An1800-second censor remains unknown; no automatic extension.")
    tag = "_".join((arm, str(update), str(puzzle))) if update is not None else f"{arm}_{puzzle}"
    directory = OUT / "retries_1800" / tag
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        assert read(protocol_path) == protocol, "Frozen retry inputs changed"
    else:
        write(protocol_path, protocol)
    output = directory / "row.json"
    if output.exists():
        assert read(output)["retry_protocol_sha256"] == digest(protocol_path)
        print("Already recorded", output, flush=True)
        return
    base["configure"]()
    _, data, _, _, state = base["context"]()
    model = base["load"](checkpoint).cpu().eval()
    model.beta = 0.
    model.config["decoder"] = "discrepancy"
    model.config.pop("max_total_work", None)
    if update is not None:
        saved = base["torch"].load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["training"]["protocol_sha256"] == protocol["original_protocol_sha256"]
        assert saved["training"]["arm"] == arm and saved["training"]["updates"] == update
    assert model.config["max_cores"] == 64 and model.config["max_open_colors"] == 6
    assert model.config["max_work"] == 1000000 and model.config["max_size"] == 100000
    assert model.config["compact_domains"] is True
    initial, _ = base["instances"](data)
    print("RETRY", tag, "checkpoint", protocol["checkpoint_sha256"], "wall guard1800s", flush=True)
    if args.prepare_only:
        print("Preflight passed; no solve performed", flush=True)
        return
    network = state.constraint_network
    original_solve = network.solve_mac
    network.solve_mac = partial(original_solve, variable_tie_break="name")
    try:
        with base["torch"].no_grad():
            result = base["solve"](network, initial[puzzle], mode, model,
                nodes=protocol["nodes"], seconds=protocol["seconds"], total_work=None,
                cluster_attempts=32, baseline_candidates=32, baseline_steps=8192, stop_on_noop=False)
    finally:
        network.solve_mac = original_solve
    assert result["status"] in ("solved", "wall_timeout", "node_limit")
    assert not result["solved"] or result["valid"]
    assert digest(checkpoint) == protocol["checkpoint_sha256"]
    assert digest(original) == protocol["original_row_sha256"]
    assert all(digest(path) == expected for path, expected in sources.items())
    write(output, dict(retry_protocol_sha256=digest(protocol_path), puzzle=puzzle, **result))
    print("RECORDED", tag, result["status"], result["nodes"], result["solve_s"], flush=True)


if __name__ == "__main__":
    main()
