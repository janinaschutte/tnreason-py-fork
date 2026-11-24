"""Serial common-protocol endpoint validation; all inputs are development only."""
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import mkdtemp
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
TRAINED = ("rms_seed0", "group_seed0", "global_seed0", "group_seed1")
OPTIONAL = ("objective_rms_seed0", "objective_group_seed0")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def checkpoint_info(name, path):
    import torch
    if not path.exists():
        return None
    saved = torch.load(path, weights_only=True, map_location="cpu")
    training_protocol = path.parent / "protocol.json"
    metadata = dict(path=str(path), sha256=digest(path), fresh_trials=saved.get("fresh_trials"))
    if name in (*TRAINED, *OPTIONAL):
        if not training_protocol.exists():
            return None
        planned = read(training_protocol)["trials"]
        if metadata["fresh_trials"] != planned:
            return None
        metadata.update(training_protocol=str(training_protocol), training_protocol_sha256=digest(training_protocol),
                        planned_trials=planned, training_source="v1 archived" if name in TRAINED else "v2")
    return metadata


def prepare(cpu):
    import torch
    from experiments.cluster_selection.reinforcement.actor import Policy, load
    from experiments.cluster_selection.reinforcement.refine import source_hashes
    out = Path(mkdtemp(prefix="common_v2_", dir=HERE))
    parent = HERE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    actor = load(parent)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        untrained = Policy(actor.network, **actor.config)
    untrained.reward_colors, untrained.beta = actor.reward_colors, actor.beta
    random_path = out / "untrained.pt"
    torch.save(dict(version=1, kind=untrained.kind, network=untrained.network, config=untrained.config,
        reward_colors=untrained.reward_colors, beta=untrained.beta, weights=untrained.state_dict(),
        actor_config=untrained.actor_config, training=dict(train_instance_hashes=[])), random_path)
    paths = dict(parent=parent, untrained=random_path,
                 **{name: HERE / name / "final.pt" for name in (*TRAINED, *OPTIONAL)})
    reused = {"plain": HERE / "baselines32_wgk4zatg/plain.json",
        "structural_batch32": HERE / "baselines32_wgk4zatg/heuristic.json",
        "structural_progressive30": HERE / "progressive30_78ufubt4/heuristic_progressive_time30.json",
        "sudoku_expert51": HERE / "baselines32_wgk4zatg/expert.json"}
    plan = dict(version=1, cpu=cpu, data=str(HERE / "pilot_data.json"),
        data_sha256=digest(HERE / "pilot_data.json"), source_sha256=source_hashes(),
        driver_sha256=digest(Path(__file__)), checkpoint_paths={name: str(path) for name, path in paths.items()},
        prepared_checkpoints={name: checkpoint_info(name, path) for name, path in paths.items()},
        reused_baselines={name: dict(path=str(path), sha256=digest(path)) for name, path in reused.items()},
        settings=dict(samples=8, seed=100000, attempts=32, seconds=30., nodes=10000),
        note="16 fixed development validation puzzles; no test100 inputs/results. V1 endpoints are finals, not selected checkpoints. "
             "Parent/untrained and completed endpoints are re-evaluated serially on CPU3 with identical MC seed and limits. "
             "Optional objective endpoints are checked after the six controls; incomplete endpoints are skipped. "
             "Every loaded checkpoint receives an immutable SHA receipt before inference. Residual DFS is separately timed plain MAC. "
             "Reused structural/plain/expert timings are earlier CPU2 observations, not fresh serial CPU3 timings; expert has larger limits.")
    write(out / "protocol.json", plan)
    print(out, flush=True)


def run(out, wait):
    import torch
    from experiments.cluster_selection.reinforcement.actor import load
    from experiments.cluster_selection.reinforcement.refine import source_hashes
    from experiments.cluster_selection.reinforcement.refine_eval import evaluate
    plan = read(out / "protocol.json")
    while not (all(Path(plan["checkpoint_paths"][name]).exists() for name in TRAINED)
               and all(checkpoint_info(name, Path(plan["checkpoint_paths"][name])) is not None for name in TRAINED)):
        if not wait:
            raise SystemExit("Wait for all four v1 final.pt files, or use --wait")
        print("Waiting for the four v1 endpoint checkpoints before serial CPU3 evaluation", flush=True)
        time.sleep(15)
    if plan["cpu"] not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is outside the scheduler affinity")
    os.sched_setaffinity(0, {plan["cpu"]})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    def verify():
        assert source_hashes() == plan["source_sha256"] and digest(Path(__file__)) == plan["driver_sha256"]
        assert digest(Path(plan["data"])) == plan["data_sha256"]
    verify()
    data = read(Path(plan["data"]))
    reports, receipts, skipped = {}, {}, {}
    worker = dict(affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
                  torch=str(torch.__version__), hostname=platform.node())
    if (out / "worker.json").exists():
        assert read(out / "worker.json") == worker, "Do not combine serial timings from different workers"
    else:
        write(out / "worker.json", worker)
    for name, filename in plan["checkpoint_paths"].items():
        verify()
        info = checkpoint_info(name, Path(filename))
        if info is None:
            skipped[name] = "No complete endpoint at its evaluation turn"
            continue
        prepared = plan["prepared_checkpoints"][name]
        if prepared is not None:
            assert info == prepared, "A checkpoint changed after protocol preparation"
        receipt = out / f"{name}.input.json"
        if receipt.exists():
            assert read(receipt) == info
        else:
            write(receipt, info)
        result_path = out / f"{name}.json"
        if result_path.exists():
            report = read(result_path)
            assert report["protocol_sha256"] == digest(out / "protocol.json")
        else:
            print(f"START {name}", flush=True)
            report = evaluate(load(filename), data, **plan["settings"])
            report.update(protocol_sha256=digest(out / "protocol.json"), checkpoint=info,
                timing_origin="fresh serial CPU3", residual_strategy="plain MAC on retained root domains; separate wall limit")
            write(result_path, report)
        receipts[name], reports[name] = info, report
        print(name, json.dumps(report["totals"], sort_keys=True), flush=True)
    for name, item in plan["reused_baselines"].items():
        path = Path(item["path"])
        assert digest(path) == item["sha256"]
        reports[name] = {**read(path), "timing_origin": "reused earlier CPU2 measurement"}
    verify()
    summary = {name: dict(**report["totals"], timing_origin=report["timing_origin"]) for name, report in reports.items()}
    if not (out / "summary.json").exists():
        write(out / "summary.json", dict(results=summary, skipped=skipped, checkpoint_receipts=receipts,
            interpretation="Exploratory development-set comparison; endpoints and common validation protocol fixed before outcomes."))
        with (out / "per_puzzle.csv").open("x", newline="") as stream:
            fields = ["method", "index", "root_solved", "dfs_nodes", "root_seconds", "residual_seconds", "timing_origin"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for name, report in reports.items():
                for row in report["rows"]:
                    writer.writerow(dict(method=name, index=row["index"], root_solved=row["root"]["solved"],
                        dfs_nodes=row["residual"]["nodes"], root_seconds=row["root"]["solve_s"],
                        residual_seconds=row["residual"]["solve_s"], timing_origin=report["timing_origin"]))
        lines = ["Common fixed development validation:16 puzzles;32 attempts,30 seconds,8 MC samples,seed100000.",
            "Residual DFS is separate plain MAC. Expert uses51 Sudoku-specific families and larger limits.",
            "Method                         Roots   DFS    Root(s) Residual(s)  Timing origin"]
        for name, values in summary.items():
            lines.append(f"{name:30} {values['root_solved']:2}/16 {str(values['dfs_nodes']):>5} "
                f"{values['root_seconds']:10.3f} {values['residual_seconds']:10.3f}  {values['timing_origin']}")
        with (out / "TABLE.txt").open("x") as stream:
            stream.write("\n".join(lines) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("out", nargs="?", type=Path)
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.cpu)
    elif args.out is None:
        parser.error("run requires the prepared output folder")
    else:
        run(args.out.resolve(), args.wait)
