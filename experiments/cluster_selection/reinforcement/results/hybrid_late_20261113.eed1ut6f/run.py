"""Frozen validation diagnostic: learned clusters after a large structural batch."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
from multiprocessing import get_context
from pathlib import Path
import runpy
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement.experiment import instances, propagate, solve

BASE = OUT.parent / "allowance_extended_20261112.uwmvnag2"
MODELS, NETWORK, DOMAINS, PLAN = {}, None, None, None
INSTALL = runpy.run_path(str(BASE / "run.py"))["install_nested"]
ORIGINAL_ATTACH, ORIGINAL_APPLY = runtime.attach_heuristic, runtime.apply_candidate


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


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


def prepare():
    previous = read(BASE / "protocol.json")
    assert all(digest(ROOT / name) == expected for name, expected in previous["source_sha256"].items()
               if name != "experiments/cluster_selection/runtime.py")
    sources = {
        "teacher": OUT.parent / "growth_exports_20261106.khccwf_d/policy.parent_sampling_grow.pt",
        "reinforce": OUT.parent / "reinforce_sampling_grow_gpu_20261106.q3mc4c0g/job_7906.0/policy.best.pt",
        "q": OUT.parent / "completed_branches_test_20261102.1d29wvmo/q.pt",
    }
    checkpoints = {}
    for name, source in sources.items():
        contents = source.read_bytes()
        checksum = sha256(contents).hexdigest()
        saved = torch.load(BytesIO(contents), map_location="cpu", weights_only=True)
        assert digest(source) == checksum, "Checkpoint changed during snapshot"
        target = OUT / f"{name}.pt"
        with target.open("xb") as stream:
            stream.write(contents)
        checkpoints[name] = dict(source=str(source), path=str(target), sha256=checksum,
                                 config=saved["config"], training=saved.get("training", {}).get("epoch"))
    before = Path("/tmp/cluster_runtime_before_hybrid_20261113.py")
    assert digest(before) == previous["source_sha256"]["experiments/cluster_selection/runtime.py"]
    with (OUT / "runtime_before.py").open("xb") as stream:
        stream.write(before.read_bytes())
    paths = [ROOT / name for name in previous["source_sha256"]]
    paths.extend((Path(__file__), OUT / "runtime_before.py", ROOT / "experiments/cluster_selection/encoder.py",
                  ROOT / "experiments/cluster_selection/data.py",
                  ROOT / "experiments/cluster_selection/reinforcement/model.py"))
    plan = dict(data=previous["data"], data_sha256=previous["data_sha256"],
        indices=[110, 111, 112, 113], checkpoints=checkpoints, source_sha256={str(p): digest(p) for p in paths},
        reference=str(BASE), reference_protocol_sha256=digest(BASE / "protocol.json"),
        heuristic_allowance=4096, learned_allowance=64, limits=previous["limits"],
        root_seconds=1800, solve_seconds=7200, nodes=100000, workers=4,
        branching="name", methods=["teacher", "reinforce", "q", "portfolio", "extra_heuristic"],
        note="Exploratory old validation, not a held-out test. Four cases still branch at H4096. "
             "Root tails start independently from identical H4096 domains; extra heuristic is a fresh nested64 batch. "
             "Full hybrid allows H4096 followed by64 shared learned contractions per propagation episode; allowances add. "
             "Old nineteen root solves are unchanged because no learned phase runs when H solves. No cumulative work cap.")
    write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir()
    print("Prepared", OUT, flush=True)


def initialize():
    global PLAN, NETWORK, DOMAINS, MODELS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PLAN = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in PLAN["source_sha256"].items())
    assert digest(PLAN["data"]) == PLAN["data_sha256"]
    data = read(PLAN["data"])
    DOMAINS, _ = instances(data)
    raw = data["network"]
    NETWORK = ConstraintNetwork({c["key"]: NumpyCore(values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    MODELS = {name: runtime.load(info["path"]) for name, info in PLAN["checkpoints"].items()}
    assert all(digest(info["path"]) == info["sha256"] for info in PLAN["checkpoints"].values())
    runtime.ConstructionEnsemble(MODELS)  # Validate all schemas and per-cluster limits.
    assert all(model.network == raw for model in MODELS.values())


def run_one(puzzle, method, domains, *, full=False):
    path = OUT / "rows" / f"{'solve' if full else 'root'}_{method}_{puzzle}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    model = runtime.ConstructionEnsemble(MODELS) if method in ("portfolio", "hybrid") else MODELS.get(method, MODELS["teacher"])
    captured, applications = {}, []

    def attach(state, **options):
        if full:
            runtime.attach_hybrid(state, model, heuristic_candidates=PLAN["heuristic_allowance"],
                learned_candidates=PLAN["learned_allowance"])
            policy = state.on_closure
            INSTALL(policy.heuristic.generator, PLAN["heuristic_allowance"])
        else:
            ORIGINAL_ATTACH(state, **options)
            policy = state.on_closure
            INSTALL(policy.generator, options["max_candidates"])
        captured["policy"] = policy

    def observed(state, candidate, prepared=None, **limits):
        before = {v: len(state.domains[v]) for v in model.reward_colors}
        result = ORIGINAL_APPLY(state, candidate, prepared, **limits)
        source = getattr(model, "last_source", None) or method
        application = dict(source=source, removed_values=result.removed_values,
            primary_removed=sum(n-len(state.domains[v]) for v,n in before.items()), contradiction=result.contradiction)
        if result.removed_values or result.contradiction:
            application.update(core_keys=candidate.core_keys, open_colors=candidate.open_colors)
        applications.append(application)
        return result

    runtime.attach_heuristic, runtime.apply_candidate = attach, observed
    start = perf_counter()
    try:
        with torch.no_grad():
            if full:
                # Mark structural applications without changing the transaction.
                original_attach = attach
                def attach_observed(state, **options):
                    original_attach(state, **options)
                    captured["policy"].heuristic.on_application = lambda *_: applications[-1].update(source="heuristic")
                runtime.attach_heuristic = attach_observed
                row = solve(NETWORK, domains, "heuristic", model, nodes=PLAN["nodes"], seconds=PLAN["solve_seconds"],
                            baseline_candidates=PLAN["heuristic_allowance"])
            else:
                structural = method in ("heuristic", "extra_heuristic")
                row = propagate(NETWORK, domains, "heuristic" if structural else "trained", model,
                    seconds=PLAN["root_seconds"], baseline_candidates=(PLAN["heuristic_allowance"] if method == "heuristic"
                    else PLAN["learned_allowance"]), cluster_attempts=PLAN["learned_allowance"])
        assert row["status"] not in ("invalid", "unsat"), row["status"]
        if row["status"] != "wall_timeout":
            assert all(set(values) <= set(domains.get(v, range(NETWORK.dimension_dict[v]))) for v, values in row.get("domains", {}).items())
        if captured:
            policy = captured["policy"]
            row["phases"] = (dict(heuristic=dict(policy.heuristic.statistics), learned=dict(policy.learned.statistics))
                             if full else dict(heuristic=dict(policy.statistics)))
        else:
            row["phases"] = dict(learned=dict(row["cluster_statistics"]))
        row.update(protocol_sha256=digest(OUT / "protocol.json"), puzzle=puzzle, method=method,
                   applications=applications, elapsed_s=perf_counter()-start)
        write(path, row)
        print(path.name, row["status"], row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
        return row
    finally:
        runtime.attach_heuristic, runtime.apply_candidate = ORIGINAL_ATTACH, ORIGINAL_APPLY


def run_case(task):
    phase, puzzle = task
    if phase == "full":
        return run_one(puzzle, "hybrid", DOMAINS[puzzle], full=True)
    base = run_one(puzzle, "heuristic", DOMAINS[puzzle])
    if base["status"] == "wall_timeout":
        return base
    assert not base["solved"], "Prespecified remaining case unexpectedly solved by H4096"
    for method in PLAN["methods"]:
        row = run_one(puzzle, method, base["domains"])
        assert row["baseline_entropy"] == base["remaining_entropy"]
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--phase", choices=("roots", "full"), default="roots")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    plan = read(OUT / "protocol.json")
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"), initializer=initialize) as pool:
        jobs = [pool.submit(run_case, (args.phase, i)) for i in plan["indices"]]
        for job in as_completed(jobs):
            job.result()
    print("COMPLETE", args.phase, flush=True)


if __name__ == "__main__":
    main()
