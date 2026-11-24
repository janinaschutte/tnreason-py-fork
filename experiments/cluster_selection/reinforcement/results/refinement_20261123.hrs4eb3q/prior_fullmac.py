"""Actual full-MAC validation of the strongest archived GPU replay policy."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp

from experiments.cluster_selection.reinforcement import refine, refine_eval

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE.parent / "learning_diagnostics_20261113.67i11ilh/continuations/gpu_self_imitation_8051.2/policy.best.pt"


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    output = Path(mkdtemp(prefix="prior_gpu_fullmac_", dir=BASE))
    (output / "rows").mkdir()
    data_path = BASE / "pilot_data.json"
    data = json.loads(data_path.read_text())
    indices = data["validation_indices"]
    assert len(indices) == 16
    arguments = [str(data_path), "--checkpoint", str(CHECKPOINT), "--output", str(output / "result.json"),
                  "--full-mac", "--attempts", "32", "--seconds", "300", "--nodes", "10000", "--seed", "100000"]
    hashes = {str(path): sha256(path.read_bytes()).hexdigest() for path in (data_path, CHECKPOINT, Path(__file__))}
    assert hashes[str(CHECKPOINT)] == "7d4e9e57244cda6041de7a1538181432a5ca437186b2bb2ee0713f2ed2c64049"
    sources = refine.source_hashes()
    write(output / "receipt.json", dict(hashes=hashes, source_sha256=sources, arguments=arguments,
        scope="16 reused development puzzles; clusters throughout DFS, one300s whole-solve limit per puzzle",
        timing="CPU0 on a shared host; model loading excluded from per-solve time"))
    original, rows = refine_eval.solve, []

    def observed(*args, **kwargs):
        row = original(*args, **kwargs)
        position = len(rows)
        rows.append(dict(index=indices[position], **row))
        write(output / "rows" / f"{position:02d}.json", rows[-1])
        return row

    refine_eval.solve = observed
    try:
        report = refine_eval.main(arguments)
    finally:
        refine_eval.solve = original
    assert len(rows) == 16 and sources == refine.source_hashes()
    assert all(sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    checked_entries = checked_givens = 0
    colors = data["network"]["colors"]
    for row in rows:
        assert row["status"] == "solved" and set(row["assignment"]) == set(colors)
        assignment = row["assignment"]
        for core in data["network"]["cores"]:
            offset = 0
            for color, size in zip(core["colors"], core["shape"]):
                assert assignment[colors[color]] in range(size)
                offset = offset * size + assignment[colors[color]]
            assert core["values"][offset]
            checked_entries += 1
        for index, digit in enumerate(data["puzzles"][row["index"]]):
            if digit not in "0.":
                r, c = divmod(index, 9)
                assert assignment[f"pos_{r//3}_{r%3}_{c//3}_{c%3}"] == int(digit) - 1
                checked_givens += 1
    write(output / "summary.json", dict(totals=report["totals"], settings=report["protocol"]["settings"],
        complete=len(rows), assignments_valid=len(rows), original_tensor_entries_checked=checked_entries,
        original_givens_checked=checked_givens, sources_inputs_checkpoint_unchanged=True))
    print(output, flush=True)


if __name__ == "__main__":
    main()
