"""Repeat the saved nested allowance sweep with both variable tie-breaks."""
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import runpy
import sys
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement.experiment import instances, solve
from experiments.cluster_selection.reinforcement.train import _state

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "allowance_plateau_20261030.bih66zeb"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


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


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(53)
    original = read(OLD / "protocol.json")
    allowances = original["allowances"] + read(OLD / "extension.json")["allowances"]
    cases = [(a, tie) for a in allowances for tie in ("name", "oldest_pruned")]
    Random(53).shuffle(cases)
    sources = [Path(__file__), OLD / "sweep.py", ROOT / "experiments/constraint_networks/mac.py",
        ROOT / "experiments/constraint_networks/constraint_networks.py",
        ROOT / "experiments/constraint_networks/generalized_arc_consistency.py", Path(runtime.__file__),
        ROOT / "experiments/cluster_selection/candidate_generation.py",
        ROOT / "experiments/cluster_selection/contraction_cost.py",
        ROOT / "experiments/cluster_selection/candidate_evaluation_mac.py",
        ROOT / "experiments/cluster_selection/reinforcement/experiment.py",
        ROOT / "experiments/cluster_selection/reinforcement/train.py"]
    protocol = dict(data=original["data"], data_sha256=original["data_sha256"],
        indices=original["indices"], allowances=allowances, cases=cases, seed=53,
        limits=original["limits"], guards=original["guards"], total_work=None,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        historical_sha256={p.name: digest(p) for p in [OLD / "protocol.json", OLD / "extension.json"]
            + [OLD / f"heuristic_{a}.jsonl" for a in allowances]},
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        no_training=True, no_test_queries=True, threads_per_worker=1,
        generation="Reuse the original nested_generate, stages 32..allowance; 256*quota steps; "
                   "restore generator.max_steps=8192 between generation calls.",
        branching="MRV, highest degree, optional oldest retained domain reduction, name.",
        evaluation="Fresh complete paired name and oldest_pruned full solves; no duplicate root runs. "
                   "A solved one-node result is branch-free. Same original 23 validation puzzle indices.")
    path = OUT / "protocol.json"
    if path.exists():
        if read(path) != json.loads(json.dumps(protocol)):
            raise AssertionError("Frozen protocol changed")
    else:
        write(path, protocol)
    if digest(Path(protocol["data"])) != protocol["data_sha256"]:
        raise AssertionError("Dataset changed")
    (OUT / "rows").mkdir(exist_ok=True)
    if len(sys.argv) == 1:
        print("Prepared", path, flush=True)
        return
    allowance = int(sys.argv[1])
    if allowance not in allowances:
        raise ValueError("Unknown allowance")
    data = read(Path(protocol["data"]))
    selected = {**data, "puzzles": [data["puzzles"][i] for i in protocol["indices"]]}
    initial, _ = instances(selected)
    network = _state({**data, "samples": [dict(domains=[list(range(d))
                     for d in data["network"]["dimensions"]])]}).constraint_network
    model = SimpleNamespace(config=protocol["limits"], reward_colors=data["reward_colors"])
    nested_generate = runpy.run_path(str(OLD / "sweep.py"))["nested_generate"]
    original_attach, original_solve = runtime.attach_heuristic, network.solve_mac

    def staged_attach(state, **options):
        quota = options.pop("max_candidates")
        result = original_attach(state, max_candidates=quota, max_steps=8192, **options)
        generator = state.on_closure.generator
        generator.generate_original = generator.generate
        generator.generate = lambda domains, **generation: nested_generate(generator, domains, quota, **generation)
        return result

    runtime.attach_heuristic = staged_attach
    protocol_hash = digest(path)
    historical = {row["index"]: row for row in (
        json.loads(line) for line in (OLD / f"heuristic_{allowance}.jsonl").read_text().splitlines())
        if row.get("type") == "solve"}
    try:
        for _, tie in (case for case in cases if case[0] == allowance):
            network.solve_mac = partial(original_solve, variable_tie_break=tie)
            rows = []
            for position, (puzzle, domains) in enumerate(zip(protocol["indices"], initial), 1):
                if any(digest(ROOT / p) != expected for p, expected in protocol["source_sha256"].items()):
                    raise AssertionError("Source changed during comparison")
                path = OUT / "rows" / f"{tie}_{allowance}_{puzzle}.json"
                if path.exists():
                    record = read(path)
                    if record["protocol_sha256"] != protocol_hash:
                        raise AssertionError("Saved row provenance changed")
                else:
                    row = solve(network, domains, "heuristic", model, **protocol["guards"],
                                total_work=None, baseline_candidates=allowance)
                    if row["status"] in ("invalid", "unsat", "partial"):
                        raise AssertionError("Unexpected validation result")
                    record = dict(protocol_sha256=protocol_hash, tie=tie,
                                  allowance=allowance, puzzle=puzzle, **row)
                    write(path, record)
                rows.append(record)
                print(tie, allowance, f"{position}/23", puzzle, record["status"], record["nodes"],
                      f"{record['solve_s']:.3f}s", flush=True)
            completed = all(row["solved"] and row["valid"] for row in rows)
            summary = dict(tie=tie, allowance=allowance, instances=len(rows),
                completed=sum(row["solved"] and row["valid"] for row in rows),
                nodes=sum(row["nodes"] for row in rows) if completed else None,
                branch_free=sum(row["solved"] and row["nodes"] == 1 for row in rows),
                nodes_by_puzzle={row["puzzle"]: row["nodes"] for row in rows},
                contractions=sum(row["work"]["contractions"] for row in rows),
                solve_seconds=sum(row["solve_s"] for row in rows))
            if tie == "name":
                summary["historical_node_mismatches"] = [row["puzzle"] for row in rows
                    if (row["status"], row["nodes"]) !=
                    (historical[row["puzzle"]]["status"], historical[row["puzzle"]]["nodes"])]
            path = OUT / f"{tie}_{allowance}.json"
            if path.exists():
                if read(path) != json.loads(json.dumps(summary)):
                    raise AssertionError("Saved summary changed")
            else:
                write(path, summary)
            print("COMPLETE", json.dumps(summary), flush=True)
    finally:
        runtime.attach_heuristic = original_attach
        network.solve_mac = original_solve


if __name__ == "__main__":
    main()
