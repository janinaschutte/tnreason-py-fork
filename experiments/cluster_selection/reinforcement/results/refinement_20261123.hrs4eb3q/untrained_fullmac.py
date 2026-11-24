"""Durable untrained-constructor control under the actual full-MAC protocol."""
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import mkdtemp

from full_mac_global_rms import checked, digest, write
from experiments.cluster_selection.reinforcement import refine, refine_eval

BASE = Path(__file__).resolve().parent


def main():
    data_path = BASE / "pilot_data.json"
    checkpoint = BASE / "common_v2_2wlie6na/untrained.pt"
    reference_path = BASE / "baselines32_wgk4zatg/plain.json"
    data = json.loads(data_path.read_text())
    initial, _ = refine_eval.instances(data)
    references = {row["index"]: row["residual"]["assignment"]
                  for row in json.loads(reference_path.read_text())["rows"]}
    assert len(data["validation_indices"]) == 16
    assert digest(checkpoint) == "469d75c6c65f25a8f5b56820bb2a844b71dfe89a64d254c8810a19a7e048041b"
    for index in data["validation_indices"]:
        checked(data, index, references[index], initial[index])
    output = Path(mkdtemp(prefix="untrained_fullmac_", dir=BASE))
    (output / "rows").mkdir()
    arguments = [str(data_path), "--checkpoint", str(checkpoint), "--output", str(output / "result.json"),
        "--full-mac", "--attempts", "32", "--seconds", "300", "--nodes", "10000", "--seed", "100000"]
    sources = refine.source_hashes()
    hashes = {str(path): digest(path) for path in (data_path, checkpoint, reference_path,
        Path(__file__), BASE / "full_mac_global_rms.py")}
    write(output / "protocol.json", dict(hashes=hashes, source_sha256=sources, arguments=arguments,
        affinity=sorted(os.sched_getaffinity(0)), scope="16 reused development puzzles; untrained constructor, actual full MAC",
        settings=dict(attempts=32, seconds=300, nodes=10000, seed=100000, full_mac=True,
            branching="MRV, maximum incident-core degree, variable name"),
        timing="CPU2, one thread; 300s root-and-DFS wall limit per puzzle; loading excluded"))
    print(f"Output: {output}", flush=True)
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
    write(output / "summary.json", dict(totals=report["totals"],
        completed_assignment_audits=sum("assignment_audit" in row for row in rows),
        assignments_equal_reference=sum(row.get("assignment_audit", {}).get("equals_plain_reference", False) for row in rows),
        original_tensor_entries_checked=sum(row.get("assignment_audit", {}).get("tensor_entries", 0) for row in rows),
        original_givens_checked=sum(row.get("assignment_audit", {}).get("givens", 0) for row in rows),
        reference_assignments_valid=len(references),
        censored=sum(row["status"] not in {"solved", "unsat"} for row in rows),
        sources_inputs_checkpoint_references_unchanged=True))
    print(output, flush=True)


if __name__ == "__main__":
    main()
