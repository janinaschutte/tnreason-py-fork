"""Durable full-MAC check of the new global-context RMS continuation."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[4]))
from experiments.cluster_selection.reinforcement import refine, refine_eval


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked(data, index, assignment, domains):
    colors = data["network"]["colors"]
    assert set(assignment) == set(colors)
    assert all(assignment[color] in domain for color, domain in domains.items())
    for core in data["network"]["cores"]:
        offset = 0
        for color, size in zip(core["colors"], core["shape"]):
            assert assignment[colors[color]] in range(size)
            offset = offset * size + assignment[colors[color]]
        assert core["values"][offset]
    givens = 0
    for position, digit in enumerate(data["puzzles"][index]):
        if digit not in "0.":
            row, col = divmod(position, 9)
            assert assignment[f"pos_{row//3}_{row%3}_{col//3}_{col%3}"] == int(digit) - 1
            givens += 1
    return len(data["network"]["cores"]), givens


def main():
    data_path, checkpoint = BASE / "pilot_data.json", BASE / "global_rms_seed0/final.pt"
    reference_path, expert_path = [BASE / "baselines32_wgk4zatg" / name for name in ("plain.json", "expert.json")]
    data = json.loads(data_path.read_text())
    initial, _ = refine_eval.instances(data)
    references = {row["index"]: row["residual"]["assignment"] for row in json.loads(reference_path.read_text())["rows"]}
    output = Path(mkdtemp(prefix="global_rms_fullmac_", dir=BASE))
    (output / "rows").mkdir()
    arguments = [str(data_path), "--checkpoint", str(checkpoint), "--output", str(output / "result.json"),
        "--full-mac", "--attempts", "32", "--seconds", "300", "--nodes", "10000", "--seed", "100000"]
    sources = refine.source_hashes()
    hashes = {str(path): digest(path) for path in (data_path, checkpoint, reference_path, expert_path, Path(__file__))}
    assert len(data["validation_indices"]) == 16
    for index in data["validation_indices"]:
        checked(data, index, references[index], initial[index])
    write(output / "protocol.json", dict(hashes=hashes, source_sha256=sources, arguments=arguments,
        affinity=sorted(os.sched_getaffinity(0)), scope="16 reused development puzzles; actual full MAC",
        settings=dict(attempts=32, seconds=300, nodes=10000, seed=100000, full_mac=True,
            branching="MRV, maximum incident-core degree, variable name"),
        timing="CPU1, one thread; 300s root-and-DFS wall limit per puzzle; loading excluded"))
    original, rows = refine_eval.solve, []

    def observed(*args, **kwargs):
        row = dict(index=data["validation_indices"][len(rows)], **original(*args, **kwargs))
        if row["status"] == "solved":
            entries, givens = checked(data, row["index"], row["assignment"], initial[row["index"]])
            row["assignment_audit"] = dict(tensor_entries=entries, givens=givens,
                equals_plain_reference=row["assignment"] == references[row["index"]])
        rows.append(row)
        write(output / "rows" / f"{row['index']:02d}.json", row)
        return {key: value for key, value in row.items() if key not in {"index", "assignment_audit"}}

    refine_eval.solve = observed
    try:
        report = refine_eval.main(arguments)
    finally:
        refine_eval.solve = original
    assert len(rows) == 16 and sources == refine.source_hashes()
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    write(output / "summary.json", dict(totals=report["totals"], completed_assignment_audits=sum(
        "assignment_audit" in row for row in rows), sources_inputs_checkpoint_references_unchanged=True))
    write(output / "expert_reference.json", dict(path=str(expert_path), sha256=hashes[str(expert_path)],
        totals=json.loads(expert_path.read_text())["totals"],
        scope="51 Sudoku-specific families; larger limits; root propagation then separate plain residual DFS, not full MAC"))
    print(output, flush=True)


if __name__ == "__main__":
    main()
