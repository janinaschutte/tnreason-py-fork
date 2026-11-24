"""Controlled fit check: a near-best teacher pool versus one shorter teacher.

This jointly changes teacher multiplicity and length, not only label ambiguity.
Production supervised/mask implementations and old experiments are untouched.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection.reinforcement import learn
from experiments.cluster_selection.reinforcement.diagnose import teachers_for
from experiments.cluster_selection.reinforcement.experiment import fingerprints, instance_hashes, instances, solve
from experiments.cluster_selection.reinforcement.model import load
from experiments.cluster_selection.reinforcement.train import _closed, _save, _state

OLD = OUT.parents[1] / "teacher_quality_20261111.zAq3x3X8"
ARMS = ("nearbest_pool", "shortest_teacher")
MILESTONES = (0, 240, 1000, 4000)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def prepare():
    old = read(OLD / "protocol.json")
    data_path = ROOT / "experiments/cluster_selection/reinforcement/results/teacher_forks_scaled_20261028.q9m8bpg9/development.json"
    parent = ROOT / old["parent"]
    states = ROOT / "experiments/cluster_selection/reinforcement/results/constructive_improvement_20261106.903639e9/states.json"
    assert digest(data_path) == old["data_sha256"]
    assert digest(parent) == old["parent_sha256"] and digest(states) == old["states_sha256"]
    records = [read(OLD / "pools" / f"state_{idx:02d}.json") for idx in range(24)]
    assert sum(record["root"] for record in records) == 13
    selected = {}
    for record in records:
        teachers = record["pools"]["nested"]["teachers"]
        best = max(teacher["metrics"]["reward"] for teacher in teachers)
        eligible = [idx for idx, teacher in enumerate(teachers) if teacher["metrics"]["reward"] >= .9 * best]
        chosen = min(eligible, key=lambda idx: (teachers[idx]["sequence_length"],
            -teachers[idx]["metrics"]["reward"], teachers[idx]["metrics"]["work"],
            tuple(teachers[idx]["cores"]), tuple(teachers[idx]["outputs"])))
        selected[str(record["id"])] = dict(eligible=eligible, shortest=chosen)
    data = read(data_path)
    initial, _ = instances(data)
    validation = old["controls"]["validation_puzzles"]
    saved = torch.load(parent, map_location="cpu", weights_only=True)
    exposure = saved["training"]["train_instance_hashes"]
    assert not set(exposure) & instance_hashes(data, [initial[idx] for idx in validation])
    assert not fingerprints(data, [initial[r["puzzle"]] for r in records]) & fingerprints(data, [initial[idx] for idx in validation])
    rng, order = Random(29), []
    while len(order) < 4000:
        visit = list(range(24))
        rng.shuffle(visit)
        order.extend(visit)
    sources = [Path(__file__), OLD / "protocol.json", OLD / "training_plan.json", parent, data_path, states]
    sources += [OLD / "pools" / f"state_{idx:02d}.json" for idx in range(24)]
    sources += [ROOT / path for path in (
        "experiments/cluster_selection/reinforcement/learn.py", "experiments/cluster_selection/reinforcement/model.py",
        "experiments/cluster_selection/reinforcement/train.py", "experiments/cluster_selection/reinforcement/diagnose.py",
        "experiments/cluster_selection/reinforcement/experiment.py", "experiments/cluster_selection/runtime.py",
        "experiments/cluster_selection/encoder.py", "experiments/cluster_selection/data.py",
        "experiments/cluster_selection/candidate_generation.py", "experiments/cluster_selection/candidate_evaluation_mac.py",
        "experiments/cluster_selection/contraction_cost.py", "experiments/constraint_networks/mac.py",
        "experiments/constraint_networks/generalized_arc_consistency.py", "experiments/constraint_networks/constraint_networks.py")]
    protocol = dict(kind="controlled_constructive_teacher_fit", data=str(data_path), parent=str(parent),
        old=str(OLD), states=str(states), validation_indices=validation, selected=selected,
        update_order=order[:4000], milestones=MILESTONES, arms=ARMS, seed=29, lr=.0003, workers=2, threads=1,
        teacher_fraction=.9, greedy_imitation=True, from_identical_parent=True, fresh_adam=True,
        no_validation_selection=True, no_test_queries=True, cluster_attempts=32, decoder="discrepancy",
        nodes=10000, seconds=600, total_work=None, torch_version=str(torch.__version__),
        selection_rule="Within each state's >=90%-of-best measured-reward pool choose shortest candidate_trace; ties higher reward, lower work, canonical cores/outputs. All legal core orders remain available.",
        note="Joint teacher-multiplicity and teacher-length ablation; free contraction/GAC utility is primary fitting evidence, exact mask agreement only diagnostic. Training states comprise 13 roots and 11 branches. Validation remains exploratory.",
        source_sha256={str(path): digest(path) for path in sources})
    write(OUT / "protocol.json", protocol)
    print("prepared", len(records), "states", flush=True)


def context():
    plan = read(OUT / "protocol.json")
    for path, expected in plan["source_sha256"].items():
        assert digest(path) == expected, f"Frozen source changed: {path}"
    data = read(plan["data"])
    records = [read(OLD / "pools" / f"state_{idx:02d}.json") for idx in range(24)]
    validation = [record for record in read(plan["states"]) if record["split"] == "validation"]
    assert len(validation) == 15
    state = _state({**data, "samples": [dict(domains=[list(range(dim)) for dim in data["network"]["dimensions"]])]})
    return plan, data, records, validation, state


def utility(model, state, records):
    model.eval()
    rows = []
    with torch.no_grad():
        for record in records:
            with _closed(state, record, model.network["colors"]):
                before = state.snapshot_domains()
                candidate, _, _, trace = model.rollout(state.domains, greedy=True, beta=0.)
                metrics = learn.measure(model, state, candidate, objective="pruning", known_satisfiable=record["known_satisfiable"])
                assert before == state.snapshot_domains(), "Trial changed frozen domains"
                teachers = record["pools"]["nested"]["teachers"] if "pools" in record else record["teachers"]
                teacher_match = candidate is not None and learn._signature(candidate) in {
                    (tuple(sorted(teacher["cores"])), tuple(sorted(teacher["outputs"]))) for teacher in teachers}
                rows.append(dict(state=record["id"], puzzle=record["puzzle"], root=record["root"],
                    teacher_match=teacher_match,
                    steps=len(trace), cores=[] if candidate is None else candidate.core_keys,
                    outputs=[] if candidate is None else candidate.open_colors, metrics=metrics))
    groups = {}
    for split in ("all", "root", "branch"):
        selected = [row for row in rows if split == "all" or row["root"] == (split == "root")]
        groups[split] = dict(states=len(selected), useful=sum(row["metrics"]["useful"] for row in selected),
            solved=sum(row["metrics"]["solved"] for row in selected),
            contradictions=sum(row["metrics"]["contradictions"] for row in selected),
            mean_reward=sum(row["metrics"]["reward"] for row in selected) / max(1, len(selected)),
            gain=sum(row["metrics"]["gain"] for row in selected),
            teacher_matches=sum(row["teacher_match"] for row in selected),
            rejected=sum(row["metrics"]["rejected"] for row in selected))
    return dict(groups=groups, rows=rows)


def evaluate(model, state, data, plan, directory, mode="trained"):
    initial, _ = instances(data)
    directory.mkdir(exist_ok=True)
    rows = []
    for puzzle in plan["validation_indices"]:
        path = directory / f"puzzle_{puzzle:03d}.json"
        if path.exists():
            row = read(path)
        else:
            with torch.no_grad():
                result = solve(state.constraint_network, initial[puzzle], mode, model,
                    nodes=plan["nodes"], seconds=plan["seconds"], cluster_attempts=32,
                    baseline_candidates=32, baseline_steps=8192, stop_on_noop=False)
            assert result["status"] not in ("invalid", "unsat"), "Invalid validation solution"
            row = dict(puzzle=puzzle, **result)
            write(path, row)
            print(directory.name, puzzle, result["status"], result["nodes"], flush=True)
        rows.append(row)
    completed = [row for row in rows if row["solved"] and row["valid"]]
    return dict(instances=len(rows), complete=len(completed),
        nodes=sum(row["nodes"] for row in completed) if len(completed) == len(rows) else None,
        completed_nodes=sum(row["nodes"] for row in completed),
        branch_free=sum(row["nodes"] == 1 for row in completed),
        censored=[row["puzzle"] for row in rows if not row["solved"]])


def run_arm(arm):
    configure()
    plan, data, records, validation, state = context()
    directory = OUT / arm
    directory.mkdir(exist_ok=True)
    model = load(plan["parent"]).cpu()
    model.beta = 0.
    model.config["decoder"] = "discrepancy"
    model.config.pop("max_total_work", None)
    torch.manual_seed(29)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["lr"])
    rng, history, start_update = Random(29), [], 0
    checkpoints = [value for value in MILESTONES if (directory / f"policy_{value:04d}.pt").exists()]
    if checkpoints:
        start_update = max(checkpoints)
        saved = torch.load(directory / f"policy_{start_update:04d}.pt", map_location="cpu", weights_only=True)
        assert saved["training"]["protocol_sha256"] == digest(OUT / "protocol.json")
        assert saved["training"]["arm"] == arm and saved["training"]["updates"] == start_update
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved["history"]
        print(arm, "resume", start_update, flush=True)
    teacher_sets = []
    for record in records:
        selected = plan["selected"][str(record["id"])]
        choices = selected["eligible"] if arm == "nearbest_pool" else [selected["shortest"]]
        teachers = record["pools"]["nested"]["teachers"]
        teacher_sets.append(teachers_for(dict(teachers=[teachers[idx] for idx in choices])))
    metadata = dict(method="controlled_constructive_teacher_fit", parent=plan["parent"],
        parent_sha256=digest(plan["parent"]), protocol_sha256=digest(OUT / "protocol.json"),
        train_puzzles=sorted({record["puzzle"] for record in records}),
        train_instance_hashes=torch.load(plan["parent"], map_location="cpu", weights_only=True)["training"]["train_instance_hashes"],
        arm=arm, seed=29, lr=plan["lr"], greedy_imitation=True, joint_coherence_length_ablation=True)
    began, last_progress = perf_counter(), perf_counter()
    for update in range(start_update, 4001):
        if update in MILESTONES:
            checkpoint = directory / f"policy_{update:04d}.pt"
            if not checkpoint.exists():
                _save(checkpoint, model, history, 0., 1., training={**metadata, "updates": update},
                      optimizer_state=optimizer.state_dict(), cost_metric="pruning_fraction")
            probe_path = directory / f"utility_{update:04d}.json"
            if not probe_path.exists():
                result = dict(update=update, train=utility(model, state, records),
                              validation=utility(model, state, validation),
                              checkpoint_sha256=digest(checkpoint))
                write(probe_path, result)
                print(arm, update, "train useful", result["train"]["groups"]["all"]["useful"], "/24",
                      "validation", result["validation"]["groups"]["all"]["useful"], "/15", flush=True)
            if update in (1000, 4000) and not (directory / f"dfs_{update:04d}.json").exists():
                result = evaluate(model.eval(), state, data, plan, directory / f"dfs_{update:04d}")
                write(directory / f"dfs_{update:04d}.json", result)
                print(arm, update, "DFS", result, flush=True)
        if update == 4000:
            break
        record = records[plan["update_order"][update]]
        model.train()
        with _closed(state, record, model.network["colors"]):
            optimizer.zero_grad(set_to_none=True)
            loss = learn.teacher_loss(model, state.domains, teacher_sets[record["id"]], rng, greedy=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(update=update + 1, state=record["id"], loss=float(loss.detach()), gradient_norm=float(norm)))
        if (update + 1) % 240 == 0 or perf_counter() - last_progress >= 50:
            print(arm, update + 1, "/4000 loss", history[-1]["loss"], flush=True)
            last_progress = perf_counter()
    return dict(arm=arm, elapsed_s=perf_counter() - began)


def baselines():
    configure()
    plan, data, _, _, state = context()
    model = load(plan["parent"]).cpu().eval()
    model.beta = 0.
    model.config["decoder"] = "discrepancy"
    model.config.pop("max_total_work", None)
    for name, mode in (("plain", "plain"), ("heuristic32", "heuristic"), ("parent", "trained")):
        path = OUT / f"baseline_{name}.json"
        if not path.exists():
            write(path, evaluate(model, state, data, plan, OUT / f"baseline_{name}", mode))


def report():
    plan = read(OUT / "protocol.json")
    summary = dict(kind=plan["kind"], protocol_sha256=digest(OUT / "protocol.json"),
        note=plan["note"], no_validation_selection=True, no_test_queries=True,
        baselines={name: read(OUT / f"baseline_{name}.json") for name in ("plain", "heuristic32", "parent")}, arms={})
    for arm in ARMS:
        summary["arms"][arm] = {}
        for update in MILESTONES:
            result = read(OUT / arm / f"utility_{update:04d}.json")
            row = dict(train=result["train"]["groups"], validation=result["validation"]["groups"])
            if update:
                saved = torch.load(OUT / arm / f"policy_{update:04d}.pt", map_location="cpu", weights_only=True)
                row["last_24_online_mean_loss"] = sum(item["loss"] for item in saved["history"][-24:]) / 24
            if update in (1000, 4000):
                row["dfs"] = read(OUT / arm / f"dfs_{update:04d}.json")
            summary["arms"][arm][str(update)] = row
    write(OUT / "summary.json", summary)
    lines = ["Controlled constructive-teacher fitting diagnostic", plan["note"],
             "Same parent,24 fixed states, greedy imitation, fresh Adam3e-4, identical state order; no validation selection.",
             "Train includes13 roots/11 branch states; validation15 unresolved roots and23 full puzzles. Allowance32, discrepancy, nodes10000, seconds600.",
             "Arm/update | train useful/24 | root useful/13 | branch useful/11 | validation useful/15 | full validation DFS"]
    for arm in ARMS:
        for update, row in summary["arms"][arm].items():
            lines.append(f"{arm}/{update} | {row['train']['all']['useful']} | {row['train']['root']['useful']} | {row['train']['branch']['useful']} | {row['validation']['all']['useful']} | {row.get('dfs', {}).get('nodes', 'not evaluated')}")
    lines.append("Fresh baselines: " + json.dumps(summary["baselines"], separators=(",", ":")))
    with (OUT / "REPORT.txt").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run", "report"), required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
            for future in as_completed([executor.submit(run_arm, arm) for arm in ARMS]):
                print("complete", future.result(), flush=True)
            executor.submit(baselines).result()
        report()
    else:
        report()


if __name__ == "__main__":
    main()
