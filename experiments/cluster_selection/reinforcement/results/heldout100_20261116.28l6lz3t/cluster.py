"""Prepare a separate, paired HTCondor replication of the frozen 100-puzzle test.

One CPU job runs all eight methods for one puzzle. Only complete attempts enter
the report, so an eviction cannot mix workers within a puzzle's comparison.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import mkdtemp

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
OLD = BASE.parent / "learning_diagnostics_20261113.67i11ilh/matched_time_20261115.PW6lJf"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("frozen_cluster_evaluation", OLD / "run.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def verify(plan):
    hashes = {**plan["source_sha256"], **plan["reference_sha256"], plan["data"]: plan["data_sha256"]}
    hashes.update({c["path"]: c["sha256"] for c in plan["checkpoints"].values()})
    assert all(engine.digest(Path(path)) == expected for path, expected in hashes.items()), "Frozen input changed"


def prepare(max_jobs):
    assert max_jobs > 0
    original = engine.read(BASE / "protocol.json")
    old_root = Path(original["data"]).parents[5]

    def relocated(path):
        return str(ROOT / Path(path).relative_to(old_root))

    plan = deepcopy(original)
    for field in ("source_sha256", "reference_sha256"):
        plan[field] = {relocated(path): digest for path, digest in original[field].items()}
    plan["data"] = relocated(original["data"])
    plan["reference_assignments"] = {key: relocated(path) for key, path in original["reference_assignments"].items()}
    for checkpoint in plan["checkpoints"].values():
        checkpoint["path"] = relocated(checkpoint["path"])
    plan["parent_protocol"] = str(BASE / "protocol.json")
    plan["parent_protocol_sha256"] = engine.digest(BASE / "protocol.json")
    plan["kind"] = "cluster_paired_100_matched_time_root_propagation"
    plan["cpu"] = None
    plan["workers"] = max_jobs
    plan.pop("cpu_topology", None)
    plan.pop("original_affinity", None)
    plan["environment"] = {"recorded_per_attempt": True, "torch_threads": 1, "blas_threads": 1}
    plan["timing"] = original["timing"].replace("One serial pinned CPU process;", "One single-thread CPU process per puzzle;")
    plan["interpretation"] = "Parallel replication on the same frozen 100 puzzles and checkpoints. " \
        "All eight methods for each puzzle run sequentially on the same worker; workers may differ " \
        "between puzzles. Record runtime and hardware per attempt, and keep this replication separate " \
        "from local timings. No retraining, checkpoint selection, or performance-based puzzle selection."
    plan["execution"] = "One one-thread CPU job per puzzle; all eight methods on the same worker. " \
        "Respect scheduler affinity; no arbitrary CPU pinning. A restarted job reruns a whole puzzle " \
        "in a separate attempt. Keep cluster timing results separate from the local replication."
    plan["source_sha256"][str(Path(__file__).resolve())] = engine.digest(Path(__file__))
    plan["source_sha256"][str(BASE / "protocol.json")] = engine.digest(BASE / "protocol.json")
    verify(plan)
    out = Path(mkdtemp(prefix="cluster_cpu_", dir=BASE))
    for name in ("attempts", "done", "logs"):
        (out / name).mkdir()
    submission = f'''universe = vanilla
initialdir = {ROOT}
executable = {sys.executable}
transfer_executable = False
should_transfer_files = NO
arguments = "-u '{Path(__file__).resolve()}' task '{out}' $(Process)"
request_cpus = 1
request_gpus = 0
requirements = (TARGET.TotalGPUs == 0)
request_memory = 4GB
request_disk = 1GB
getenv = PATH, LD_LIBRARY_PATH
environment = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cluster-mpl-$(ClusterID)-$(Process)"
output = {out}/logs/$(ClusterID).$(Process).out
error = {out}/logs/$(ClusterID).$(Process).err
log = {out}/logs/$(ClusterID).$(Process).log
stream_output = True
stream_error = True
on_exit_hold = (ExitBySignal =?= True) || (ExitCode =!= 0)
max_materialize = {max_jobs}
queue 100
'''
    with (out / "cpu.sub").open("x") as handle:
        handle.write(submission)
    plan["source_sha256"][str(out / "cpu.sub")] = engine.digest(out / "cpu.sub")
    engine.write(out / "protocol.json", plan)
    print(out)


def initialize(out, plan):
    import numpy as np
    import torch
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from tnreason.engine.workload_to_numpy import NumpyCore
    verify(plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["seed"])
    models = {name: runtime.load(info["path"]).cpu().eval() for name, info in plan["checkpoints"].items()}
    assert all(models[name].config == info["config"] for name, info in plan["checkpoints"].items())
    data = engine.read(Path(plan["data"]))
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(
        values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = engine.forbid
    engine.OUT = out
    engine.CONTEXT.update(plan=plan, models=models, network=network, domains=instances(data)[0], runtime=runtime)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    engine.write(out / "worker.json", dict(host=platform.node(), cpu_model=cpu,
        affinity=sorted(os.sched_getaffinity(0)), python=sys.version, torch=torch.__version__, numpy=np.__version__,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        job_ad=Path(os.environ["_CONDOR_JOB_AD"]).read_text() if "_CONDOR_JOB_AD" in os.environ else None))


def task(out, position):
    plan = engine.read(out / "protocol.json")
    assert position in plan["positions"]
    finished = out / "done" / f"{position:02d}.json"
    if finished.exists():
        assert engine.read(finished)["protocol_sha256"] == engine.digest(out / "protocol.json")
        print(f"Puzzle {position} already complete")
        return
    attempt = Path(mkdtemp(prefix=f"{position:02d}_", dir=out / "attempts"))
    (attempt / "rows").mkdir()
    with (attempt / "protocol.json").open("x") as handle:
        handle.write((out / "protocol.json").read_text())
    initialize(attempt, plan)
    methods = [method for method, i in plan["order"] if i == position]
    assert set(methods) == set(engine.METHODS) and len(methods) == 8
    for method in methods:
        engine.evaluate(method, position)
    for method in methods:
        engine.residual(method, position)
    paths = sorted((attempt / "rows").glob("*.json"))
    assert len(paths) == 16
    record = dict(position=position, protocol_sha256=engine.digest(out / "protocol.json"), attempt=str(attempt),
        row_sha256={p.name: engine.digest(p) for p in paths}, worker_sha256=engine.digest(attempt / "worker.json"))
    try:
        engine.write(finished, record)
    except FileExistsError:  # A duplicate job may have published another complete attempt.
        assert engine.read(finished)["protocol_sha256"] == record["protocol_sha256"]
    print(f"COMPLETE puzzle {position}: eight methods and residual DFS", flush=True)


def collect(out):
    plan = engine.read(out / "protocol.json")
    verify(plan)
    groups, details = defaultdict(list), []
    for position in plan["positions"]:
        marker = engine.read(out / "done" / f"{position:02d}.json")
        assert marker["position"] == position and marker["protocol_sha256"] == engine.digest(out / "protocol.json")
        attempt = Path(marker["attempt"])
        assert engine.digest(attempt / "protocol.json") == marker["protocol_sha256"]
        assert engine.digest(attempt / "worker.json") == marker["worker_sha256"]
        expected_files = {f"{m}_{position:02d}{suffix}" for m in engine.METHODS
                          for suffix in (".json", ".dfs.json")}
        assert set(marker["row_sha256"]) == expected_files
        worker = engine.read(attempt / "worker.json")
        for name, expected in marker["row_sha256"].items():
            assert engine.digest(attempt / "rows" / name) == expected
        for method in engine.METHODS:
            path = attempt / "rows" / f"{method}_{position:02d}.json"
            row, residual = engine.read(path), engine.read(path.with_suffix(".dfs.json"))
            assert row["protocol_sha256"] == residual["protocol_sha256"] == marker["protocol_sha256"]
            assert row["position"] == residual["position"] == position and row["method"] == residual["method"] == method
            assert row["no_added_cores"] and not row["dfs_called"] and row["reference_retained"]
            assert row["decision_level"] == 0 and row["no_pending_queue"] and residual["timed_separately"]
            assert row["affinity"] == worker["affinity"] and residual["root_row_sha256"] == engine.digest(path)
            groups[method].append((row, residual))
            for budget in plan["budgets_s"]:
                prefix, finish = row["prefixes"][str(budget)], residual["budgets"][str(budget)]
                actual = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= budget)
                assert prefix["solved"] == actual and prefix["snapshot"] == finish["snapshot"]
                assert not actual or finish["valid"] and finish["nodes"] == 1
                details.append(dict(method=method, position=position, budget_s=budget, host=worker["host"],
                    cpu_model=worker["cpu_model"], solved_without_branching=int(actual),
                    remaining_dfs_nodes=finish["nodes"] if finish["valid"] and finish["solved"] else None,
                    residual_status=finish["status"], solved_s=row["solved_s"], trajectory_wall_s=row["wall_s"]))
    totals = []
    for method in engine.METHODS:
        for budget in plan["budgets_s"]:
            values = [r for r in details if r["method"] == method and r["budget_s"] == budget]
            assert len(values) == 100
            censored = sum(r["remaining_dfs_nodes"] is None for r in values)
            totals.append(dict(method=method, budget_s=budget, instances=100,
                solved_without_branching=sum(r["solved_without_branching"] for r in values),
                remaining_dfs_nodes=None if censored else sum(r["remaining_dfs_nodes"] for r in values),
                residual_censored=censored))
    for name, rows in (("totals", totals), ("per_puzzle", details)):
        with (out / f"matched_time_{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    engine.write(out / "summary.json", dict(protocol_sha256=engine.digest(out / "protocol.json"), totals=totals))
    spec = importlib.util.spec_from_file_location("frozen_cluster_plot", OLD / "plot.py")
    drawing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drawing)
    drawing.plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 12,
                               "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = drawing.plt.subplots(figsize=(10.8, 6.9))
    drawing.root_curves(ax, plan, groups)
    ax.set_ylim(-1, 103)
    ax.set_yticks(range(0, 101, 10))
    ax.set_title("Solving 100 held-out Sudokus without branching", pad=18)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(.5, .04), frameon=False)
    fig.subplots_adjust(left=.09, right=.975, top=.88, bottom=.24)
    for extension in ("png", "pdf", "svg"):
        path = out / f"matched_time_roots_no_caption.{extension}"
        assert not path.exists()
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.12)
    drawing.plt.close(fig)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--max-jobs", type=int, default=10)
    worker = commands.add_parser("task")
    worker.add_argument("out", type=Path)
    worker.add_argument("position", type=int)
    commands.add_parser("collect").add_argument("out", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "task": task, "collect": collect}[command](**args)
