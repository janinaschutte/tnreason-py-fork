"""Locate the first unsupported action on an unforced constructive greedy path."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection.reinforcement.model import load


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def audit(update):
    destination = OUT / f"prefix_audit_{update:04d}.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan = read(OUT / "protocol.json")
    assert all(digest(path) == expected for path, expected in plan["source_sha256"].items())
    records = [read(Path(plan["old"]) / "pools" / f"state_{idx:02d}.json") for idx in range(24)]
    all_rows, inputs = [], {str(OUT / "protocol.json"): digest(OUT / "protocol.json")}
    for arm in plan["arms"]:
        checkpoint = OUT / arm / f"policy_{update:04d}.pt"
        utility = OUT / arm / f"utility_{update:04d}.json"
        inputs.update({str(checkpoint): digest(checkpoint), str(utility): digest(utility)})
        model = load(checkpoint).cpu().eval()
        model.beta = 0.
        model.config["decoder"] = "discrepancy"
        measured = {row["state"]: row for row in read(utility)["train"]["rows"]}
        for record in records:
            choice = plan["selected"][str(record["id"])]
            indices = choice["eligible"] if arm == "nearbest_pool" else [choice["shortest"]]
            teachers = [record["pools"]["nested"]["teachers"][i] for i in indices]
            targets = [(set(model.key_idx[k] for k in t["cores"]),
                        set(model.color_idx[c] for c in t["outputs"]),
                        set(model.color_idx[c] for c in t["metrics"]["pruned_outputs"])) for t in teachers]
            original, first, calls = model.distribution, [], [0]

            def traced(encoded, sizes, keys, opened, phase, *args):
                actions, distribution = original(encoded, sizes, keys, opened, phase, *args)
                actual = int(model._greedy_choice(actions, distribution))
                choosing_target = model.config.get("target_first") and not keys and not opened and not phase
                valid, compatible = set(), 0
                for cores, outputs, goals in targets:
                    if choosing_target:
                        following = [("open", color) for color in goals]
                    elif phase:
                        if cores != set(keys) or not set(opened) <= outputs or opened[0] not in goals:
                            continue
                        following = [("open", color) for color in outputs - set(opened)]
                    else:
                        if not set(keys) <= cores or not set(opened) <= outputs or opened[0] not in goals:
                            continue
                        following = [("core", key) for key in cores - set(keys)]
                    compatible += 1
                    valid.update(following or [("finish", -1)])
                if not first and actions[actual] not in valid:
                    kind, index = actions[actual]
                    bucket = ("skip" if kind == "skip" else "anchor" if choosing_target else
                              "early_finish" if kind == "finish" and not phase else
                              "output" if phase else "core")
                    legal = [i for i, action in enumerate(actions) if action in valid]
                    first.append(dict(step=calls[0], bucket=bucket, action=actions[actual],
                        name=model.keys[index] if kind == "core" else model.network["colors"][index] if kind == "open" else None,
                        compatible_teachers=compatible, known_supported_legal_actions=len(legal),
                        selected_probability=float(distribution.probs[actual]),
                        best_supported_probability=max((float(distribution.probs[i]) for i in legal), default=None),
                        cores_before=len(keys), outputs_before=len(opened)))
                calls[0] += 1
                return actions, distribution

            model.distribution = traced
            try:
                with torch.no_grad():
                    candidate, _, _, trace = model.rollout(dict(zip(model.network["colors"], record["domains"])), greedy=True, beta=0.)
            finally:
                model.distribution = original
            known = measured[record["id"]]
            assert len(trace) == known["steps"] and calls[0] == len(trace)
            assert ([] if candidate is None else list(candidate.core_keys)) == known["cores"]
            assert ([] if candidate is None else list(candidate.open_colors)) == known["outputs"]
            all_rows.append(dict(arm=arm, update=update, state=record["id"], puzzle=record["puzzle"],
                root=record["root"], useful=known["metrics"]["useful"], contradiction=known["metrics"]["contradictions"],
                teacher_count=len(teachers), steps=len(trace), first_unsupported=first[0] if first else None))
    groups = {}
    for arm in plan["arms"]:
        groups[arm] = {}
        for split in ("all", "root", "branch"):
            selected = [row for row in all_rows if row["arm"] == arm and (split == "all" or row["root"] == (split == "root"))]
            counts = Counter(row["first_unsupported"]["bucket"] if row["first_unsupported"] else "supported_completion" for row in selected)
            groups[arm][split] = dict(states=len(selected), useful=sum(row["useful"] for row in selected),
                buckets=dict(counts), mean_first_unsupported_step=sum(row["first_unsupported"]["step"] for row in selected
                    if row["first_unsupported"]) / max(1, sum(bool(row["first_unsupported"]) for row in selected)))
    result = dict(kind="first_unsupported_free_greedy_action", update=update, groups=groups, rows=all_rows,
        input_sha256=inputs, audit_source_sha256=digest(Path(__file__)),
        additional_contractions=0, additional_dfs_solves=0, additional_training_updates=0,
        definition="First zero-based action outside every compatible verified teacher continuation for that arm. Unsupported does not mean wrong or useless; useful novel constructions can leave the finite teacher pool.",
        verification="Every unforced construction signature and trace length reproduces its saved utility probe; no teacher-forced action or prefix is used.")
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", type=int, choices=(240, 1000, 4000), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    audit(args.update)
