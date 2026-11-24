"""Additional exploratory comparison of fixed final continuations on reused tests."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import sha256
from io import BytesIO
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[6]
HELDOUT = OUT.parents[1] / "heldout"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("heldout_final_helper", HELDOUT / "run.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = OUT
METHODS = ("reinforce", "lower_lr", "self_imitation")


def prepare():
    import torch
    from experiments.cluster_selection.reinforcement.experiment import instance_hashes, instances
    original = helper.read(HELDOUT / "protocol.json")
    assert all(helper.digest(p) == h for p, h in original["source_sha256"].items())
    assert helper.digest(original["data"]) == original["data_sha256"]
    data = helper.read(original["data"])
    domains, _ = instances(data)
    fingerprints = instance_hashes(data, domains)
    checkpoints = {}
    expected_config = original["checkpoints"]["reinforce"]["config"]
    for name in METHODS:
        source = OUT.parent / f"local_{name}" / "policy.pt"  # Final only; no selection.
        contents = source.read_bytes()
        checksum = sha256(contents).hexdigest()
        saved = torch.load(BytesIO(contents), map_location="cpu", weights_only=True)
        assert saved["kind"] == "construction" and saved["config"] == expected_config
        assert saved["network"] == data["network"] and saved["beta"] == 0
        training = saved["training"]
        assert training["epoch"] == 6 and len(saved["history"]) == 6
        assert training["parent"]["sha256"] == original["checkpoints"]["reinforce"]["sha256"]
        assert not fingerprints & set(training["train_instance_hashes"])
        destination = OUT / f"{name}.pt"
        if destination.exists():
            assert helper.digest(destination) == checksum
        else:
            with destination.open("xb") as stream:
                stream.write(contents)
        assert helper.digest(source) == checksum
        checkpoints[name] = dict(path=str(destination), source=str(source), sha256=checksum,
            config=saved["config"], final_epoch=6, updates=training["updates"],
            lr=training["lr"], self_imitation_weight=training.get("self_imitation_weight", 0),
            parent=training["parent"], recorded_training_exposure=len(training["train_instance_hashes"]),
            test_training_overlap=0)
    sources = [Path(__file__), HELDOUT / "run.py"]
    for directory in ("experiments/cluster_selection", "experiments/cluster_selection/reinforcement",
                      "experiments/constraint_networks", "tnreason/engine"):
        sources.extend((ROOT / directory).glob("*.py"))
    order = [(name, position) for position in range(16) for name in METHODS]
    Random(20261115).shuffle(order)
    earlier = [HELDOUT / "protocol.json", HELDOUT / "summary.json", OUT.parent / "protocol.json",
               OUT.parent / "inputs.json", OUT.parent / "PILOT_REPORT.json"]
    earlier.extend(HELDOUT / "rows" / f"reinforce_{i:02d}.json" for i in range(16))
    plan = dict(kind="additional_exploratory_final_continuation_comparison", data=original["data"],
        data_sha256=original["data_sha256"], methods=METHODS, checkpoints=checkpoints, order=order,
        source_sha256={str(p):helper.digest(p) for p in sources},
        reused_input_sha256={str(p):helper.digest(p) for p in earlier},
        allowance=32, decoder_steps=8192, heuristic_steps=8192, limits=original["limits"],
        branching="name", nodes=10000, seconds=600, total_work=None, workers=4,
        note="Additional exploratory comparison on the same already queried 16 test puzzles; not a new untouched test. All three final stopping points and training hyperparameters were predeclared. No checkpoint selection, retraining or tuning uses these outcomes.")
    if (OUT / "protocol.json").exists():
        assert helper.read(OUT / "protocol.json") == json.loads(json.dumps(plan))
    else:
        helper.write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir(exist_ok=True)
    print("Frozen all three final checkpoints before any additional test query", flush=True)


def initialize():
    import numpy as np
    import torch
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = helper.read(OUT / "protocol.json")
    assert all(helper.digest(p) == h for p, h in {**plan["source_sha256"], **plan["reused_input_sha256"]}.items())
    assert helper.digest(plan["data"]) == plan["data_sha256"]
    data = helper.read(plan["data"])
    domains, _ = instances(data)
    models = {}
    for name, info in plan["checkpoints"].items():
        assert helper.digest(info["path"]) == info["sha256"]
        saved = torch.load(info["path"], map_location="cpu", weights_only=True)
        assert not instance_hashes(data, domains) & set(saved["training"]["train_instance_hashes"])
        models[name] = runtime.load(info["path"]).cpu().eval()
    raw = data["network"]
    network = ConstraintNetwork({c["key"]:NumpyCore(values=np.asarray(c["values"],dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = partial(network.solve_mac, variable_tie_break="name")
    helper.CONTEXT.update(plan=plan, models=models, network=network, domains=domains)


def evaluate_case(task):
    return helper.evaluate_case(task)  # Same transactional solver, guards and per-episode accounting.


def evaluate():
    plan = helper.read(OUT / "protocol.json")
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=initialize) as pool:
        for future in as_completed([pool.submit(evaluate_case, task) for task in plan["order"]]):
            future.result()
    print("COMPLETE all 48 additional solves", flush=True)


def report():
    import csv
    plan = helper.read(OUT / "protocol.json")
    assert all(helper.digest(p) == h for p,h in {**plan["source_sha256"], **plan["reused_input_sha256"]}.items())
    rows = [helper.read(p) for p in sorted((OUT / "rows").glob("*.json"))]
    assert len(rows) == 48 and len({(r["method"],r["position"]) for r in rows}) == 48
    assert all(r["protocol_sha256"] == helper.digest(OUT / "protocol.json") for r in rows)
    parent = [helper.read(HELDOUT / "rows" / f"reinforce_{i:02d}.json") for i in range(16)]
    totals, pairs = {}, {}
    for name in METHODS:
        group = sorted((r for r in rows if r["method"] == name), key=lambda r:r["position"])
        complete = all(r["status"] == "solved" and r["valid"] for r in group)
        totals[name] = dict(completed=sum(r["status"] == "solved" and r["valid"] for r in group),
            statuses=dict(Counter(r["status"] for r in group)), nodes=sum(r["nodes"] for r in group) if complete else None,
            branch_free=sum(r["status"]=="solved" and r["valid"] and r["nodes"]==1 for r in group),
            contractions=sum(r["work"]["contractions"] for r in group), estimated_work=sum(r["work"]["estimated_work"] for r in group))
        valid = [(a,b) for a,b in zip(group,parent) if a["status"]==b["status"]=="solved" and a["valid"] and b["valid"]]
        pairs[name] = dict(paired_complete=len(valid), wins=sum(a["nodes"]<b["nodes"] for a,b in valid),
            ties=sum(a["nodes"]==b["nodes"] for a,b in valid), losses=sum(a["nodes"]>b["nodes"] for a,b in valid),
            node_difference=sum(a["nodes"]-b["nodes"] for a,b in valid))
    summary = dict(note=plan["note"], protocol_sha256=helper.digest(OUT / "protocol.json"),
        reused_starting_parent=dict(nodes=sum(r["nodes"] for r in parent), branch_free=sum(r["nodes"]==1 for r in parent)),
        results=totals, comparisons_against_starting_parent=pairs,
        row_sha256={str(p):helper.digest(p) for p in sorted((OUT / "rows").glob("*.json"))})
    helper.write(OUT / "summary.json", summary)
    with (OUT / "nodes.csv").open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["position", "method", "status", "nodes", "starting_parent_nodes", "difference"])
        for row in sorted(rows,key=lambda r:(r["position"],METHODS.index(r["method"]))):
            prior = parent[row["position"]]["nodes"]
            writer.writerow([row["position"],row["method"],row["status"],row["nodes"],prior,
                None if row["nodes"] is None else row["nodes"]-prior])
    lines = [plan["note"], "", "All methods: name branching, allowance32, decoder8192, original per-cluster bounds; no cumulative work cap.",
        "Method | DFS nodes | Without branching | Complete", f"Starting parent (reused) | {summary['reused_starting_parent']['nodes']} | {summary['reused_starting_parent']['branch_free']}/16 | 16/16"]
    lines.extend(f"{name} final | {r['nodes']} | {r['branch_free']}/16 | {r['completed']}/16" for name,r in totals.items())
    lines.extend(["", "Paired wins/ties/losses against the frozen starting parent:", json.dumps(pairs,indent=2),
        "", "A small reused-test comparison with one training seed. Equal caps do not mean equal actual contractions or tensor work. Censored cases retain unknown full node counts."])
    with (OUT / "REPORT.txt").open("x") as stream:
        stream.write("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "report"))
    globals()[parser.parse_args().phase]()
