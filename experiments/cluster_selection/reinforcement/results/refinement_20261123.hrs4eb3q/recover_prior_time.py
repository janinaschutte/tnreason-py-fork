"""Recover completed time16 rows after the preserved reference-index audit error."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
OUT = BASE / "prior_gpu_replay_arli8b26"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    protocol = read(OUT / "protocol.json")
    data = read(Path(protocol["phases"]["time16"]["data"]))
    references_path = BASE / "baselines32_wgk4zatg/plain.json"
    references = {row["index"]: row["residual"]["assignment"] for row in read(references_path)["rows"]}
    rows = [read(OUT / f"time16_row_{i:02d}.json") for i in range(16)]
    assert [row["index"] for row in rows] == data["validation_indices"]
    digest = lambda path: sha256(path.read_bytes()).hexdigest()
    assert all(digest(Path(path)) == value for path, value in protocol["hashes"].items())
    assert all(digest(ROOT / path) == value for path, value in protocol["source_sha256"].items())
    colors = data["network"]["colors"]
    entries = givens = 0
    for row in rows:
        assert row["residual"]["status"] == "solved"
        assignment = row["residual"]["assignment"]
        assert set(assignment) == set(colors)
        assert all(value in row["root"]["domains"][color] for color, value in references[row["index"]].items())
        assert all(assignment[color] in values for color, values in row["root"]["domains"].items())
        for core in data["network"]["cores"]:
            offset = 0
            for color, size in zip(core["colors"], core["shape"]):
                assert assignment[colors[color]] in range(size)
                offset = offset * size + assignment[colors[color]]
            assert core["values"][offset]
            entries += 1
        for index, digit in enumerate(data["puzzles"][row["index"]]):
            if digit not in "0.":
                r, c = divmod(index, 9)
                assert assignment[f"pos_{r//3}_{r%3}_{c//3}_{c%3}"] == int(digit) - 1
                givens += 1
        if row["root"]["solved"]:
            assert row["residual"]["nodes"] == 1 and all(len(v) == 1 for v in row["root"]["domains"].values())
        row["sampled"] = dict(trials=0, useful=0, reward=0., solved=0, seconds=0.)
    totals = dict(instances=16, sampled_trials=0, sampled_useful=0, sampled_reward=0., sampled_solved=0,
        sampled_seconds=0., sampled_useful_fraction=0., sampled_mean_reward=0.,
        root_solved=sum(row["root"]["solved"] for row in rows), baseline_solved=sum(row["root"]["baseline_solved"] for row in rows),
        mean_progress=sum(row["root"]["progress"] or 0. for row in rows)/16,
        root_seconds=sum(row["root"]["solve_s"] for row in rows), dfs_complete=16,
        dfs_nodes=sum(row["residual"]["nodes"] for row in rows), residual_seconds=sum(row["residual"]["solve_s"] for row in rows))
    settings = dict(mode="trained", **protocol["phases"]["time16"]["settings"], time_only=True,
        timing="Root and residual wall limits are separate; model copy, setup and one-step trials are excluded")
    write(OUT / "time16_recovered.json", dict(rows=rows, totals=totals, settings=settings))
    write(OUT / "time16_summary.json", dict(totals=totals, settings=settings, root_statuses=dict(Counter(row["root"]["status"] for row in rows)),
        recovered_from_complete_durable_rows=True, failure="Post-evaluation reference lookup used original55 indices instead of pilot-local indices; no solving work lost.",
        recovery_driver_sha256=digest(Path(__file__)), original_tensor_entries_checked=entries, original_givens_checked=givens,
        original_references_retained=16, assignments_valid=16, sources_inputs_checkpoint_unchanged=True,
        baseline_reference_sha256=digest(references_path), durable_row_sha256={f"time16_row_{i:02d}.json": digest(OUT/f"time16_row_{i:02d}.json") for i in range(16)}))
    print(totals)


if __name__ == "__main__":
    main()
