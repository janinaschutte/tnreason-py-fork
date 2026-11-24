"""Validate a one-policy full55 artifact directly against original tensor tables."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol, result = read(args.output / "protocol.json"), read(args.output / "result.json")
    data = read(Path(protocol["data"]))
    colors, cores = data["network"]["colors"], data["network"]["cores"]
    digest = lambda path: sha256(path.read_bytes()).hexdigest()
    assert digest(Path(protocol["data"])) == protocol["data_sha256"]
    assert all(digest(ROOT / path) == value for path, value in protocol["source_sha256"].items())
    assert all(digest(Path(path)) == protocol["checkpoint_sha256"][name] for name, path in protocol["checkpoints"].items())
    assert [row["index"] for row in result["rows"]] == protocol["indices"]
    checked, entries, givens = 0, 0, 0
    parent = read(BASE / "full55_q13qt7s1/parent.json")["rows"]
    pairs = []
    for row, original in zip(result["rows"], parent):
        assignment = row["residual"]["assignment"]
        assert row["residual"]["status"] == "solved" and set(assignment) == set(colors)
        assert all(assignment[color] in row["root"]["domains"][color] for color in colors)
        for core in cores:
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
            assert row["residual"]["nodes"] == 1 and all(len(values) == 1 for values in row["root"]["domains"].values())
        assert row["index"] == original["index"]
        pairs.append(dict(index=row["index"], root_delta=int(row["root"]["solved"])-int(original["root"]["solved"]),
            node_delta=row["residual"]["nodes"]-original["residual"]["nodes"]))
        checked += 1
    audit = dict(assignments_checked=checked, original_tensor_entries_checked=entries, original_givens_checked=givens,
        all_assignments_valid=True, all_root_domains_retained_valid_solution=True, source_data_checkpoint_hashes_unchanged=True,
        root_wins=sum(row["root_delta"]>0 for row in pairs), root_losses=sum(row["root_delta"]<0 for row in pairs),
        dfs_wins=sum(row["node_delta"]<0 for row in pairs), dfs_losses=sum(row["node_delta"]>0 for row in pairs),
        root_delta=sum(row["root_delta"] for row in pairs), dfs_delta=sum(row["node_delta"] for row in pairs),
        changed_puzzles=[row for row in pairs if row["node_delta"] or row["root_delta"]])
    with (args.output / "audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
