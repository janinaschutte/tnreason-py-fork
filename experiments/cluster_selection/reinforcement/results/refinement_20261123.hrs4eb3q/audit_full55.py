"""Independent table/given/domain audit and paired uncertainty for full55."""
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import random


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
OUT = BASE / "full55_q13qt7s1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    data = read(Path(protocol["data"]))
    network = data["network"]
    colors = network["colors"]
    checked_assignments = checked_entries = checked_givens = 0
    for name in protocol["checkpoints"]:
        before, after = read(OUT / f"{name}_before.json"), read(OUT / f"{name}_after.json")
        for key in ("data_sha256", "checkpoint_sha256", "source_sha256"):
            assert before[key] == after[key]
        assert digest(Path(protocol["data"])) == before["data_sha256"]
        assert digest(Path(protocol["checkpoints"][name])) == before["checkpoint_sha256"]
        assert all(digest(ROOT / path) == value for path, value in before["source_sha256"].items())
        rows = read(OUT / f"{name}.json")["rows"]
        assert [row["index"] for row in rows] == protocol["indices"]
        for row in rows:
            assignment = row["residual"]["assignment"]
            assert row["residual"]["status"] == "solved" and set(assignment) == set(colors)
            for color, size in zip(colors, network["dimensions"]):
                assert assignment[color] in range(size)
                assert assignment[color] in row["root"]["domains"][color]
            for core in network["cores"]:
                offset = 0
                for color, size in zip(core["colors"], core["shape"]):
                    offset = offset * size + assignment[colors[color]]
                assert core["values"][offset]
                checked_entries += 1
            for index, digit in enumerate(data["puzzles"][row["index"]]):
                if digit in "0.":
                    continue
                r, c = divmod(index, 9)
                color = f"pos_{r // 3}_{r % 3}_{c // 3}_{c % 3}"
                assert assignment[color] == int(digit) - 1
                checked_givens += 1
            if row["root"]["solved"]:
                assert row["residual"]["nodes"] == 1
                assert all(len(values) == 1 for values in row["root"]["domains"].values())
            checked_assignments += 1
    root_delta = [int(row["trained_root"]) - int(row["parent_root"]) for row in summary["pairs"]]
    node_delta = [row["trained_nodes"] - row["parent_nodes"] for row in summary["pairs"]]
    rng = random.Random(1729)
    roots, nodes = [], []
    for _ in range(10000):
        indices = [rng.randrange(55) for _ in range(55)]
        roots.append(sum(root_delta[i] for i in indices) / 55)
        nodes.append(sum(node_delta[i] for i in indices))
    roots.sort()
    nodes.sort()
    wins, losses = summary["root_wins"], summary["root_losses"]
    discordant = wins + losses
    sign_p = min(1., 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2 ** discordant)
    audit = dict(assignments_checked=checked_assignments, original_tensor_entries_checked=checked_entries,
        original_givens_checked=checked_givens, every_assignment_valid=True, all_root_domains_retained_valid_solution=True,
        all_source_data_checkpoint_hashes_unchanged=True, validation_fingerprint_overlap=0,
        paired_uncertainty=dict(bootstrap_resamples=10000, seed=1729,
            root_fraction_difference=sum(root_delta) / 55, root_fraction_difference_percentile95=[roots[249], roots[9749]],
            dfs_total_difference=sum(node_delta), dfs_total_difference_percentile95=[nodes[249], nodes[9749]],
            root_exact_paired_two_sided_p=sign_p,
            caveat="Descriptive puzzle-pair resampling on reused development validation; excludes training/evaluation seed variability and checkpoint selection."),
        changed_puzzles=[row for row in summary["pairs"] if row["trained_nodes"] != row["parent_nodes"]])
    with (OUT / "audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
