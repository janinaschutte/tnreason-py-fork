"""Reserve a fresh model-free pilot, then compare frozen constructors with MAC."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import sha256
from io import BytesIO
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
BASE = ROOT / "experiments/cluster_selection/reinforcement/results"
sys.path.insert(0, str(ROOT))
LIMITS = dict(max_cores=64, max_open_colors=6, max_work=1_000_000, max_size=100_000)
METHODS = ("plain", "heuristic", "initial", "teacher", "reinforce")
CONTEXT = {}


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
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity(path):
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    stat = Path(path).stat()
    layout = [source.metadata.row_group(i).num_rows for i in range(source.num_row_groups)]
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        rows=source.metadata.num_rows, row_groups=source.num_row_groups,
        layout_sha256=sha256(json.dumps(layout).encode()).hexdigest(),
        schema_sha256=sha256(source.schema.to_arrow_schema().serialize().to_pybytes()).hexdigest(),
        whole_shard_hashed=False, retained_source_rows_replayed=True)


def initialize_preparation(raw):
    import numpy as np
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    network = ConstraintNetwork({c["key"]: NumpyCore(values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    CONTEXT["state"] = network.create_mac_state()
    assert CONTEXT["state"].propagate() and not CONTEXT["state"].is_solved()


def check_batch(rows):
    from experiments.cluster_selection.supervised.collect import sudoku_domains
    from experiments.cluster_selection.reinforcement.scale_data import check_uniqueness
    state, results = CONTEXT["state"], []
    before = state.snapshot_domains()
    for row in rows:
        state.push_level()
        try:
            for color, values in sudoku_domains(row["puzzle"]).items():
                state.restrict_domain(color, values)
            assert state.propagate()
            solved = state.is_solved()
            closure = state.snapshot_domains()
        finally:
            state.pop_level()
        assert state.snapshot_domains() == before and state.decision_level == 0
        if solved:
            results.append(("gac_solved", row))
        else:
            reference = {c: values[0] for c, values in sudoku_domains(row["solution"]).items()}
            proof = check_uniqueness(state.constraint_network, sudoku_domains(row["puzzle"]),
                                    reference, nodes=10_000, seconds=10.)
            assert proof["status"] in ("unique", "multiple", "unknown")
            row = dict(row, proof=proof, gac_domains_sha256=sha256(
                json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            results.append((proof["status"], row))
    return results


def prepare():
    import pyarrow.parquet as pq
    from experiments.cluster_selection.reinforcement.scale_data import normalize, _valid_solution, reserve_validation
    if (OUT / "preparation.json").exists():
        assert digest(OUT / "unique_test.json") == read(OUT / "preparation.json")["data_sha256"]
        print("Reusing completed model-free reservation", flush=True)
        return
    previous = read(BASE / "matched_source_test_20261106._de5ty3e/predeclaration.json")
    paths = {k: ROOT / v["path"] for k, v in previous["inputs"].items()}
    paths.update(matched_test=BASE / "matched_source_test_20261106._de5ty3e/unique_test.json",
        matched_protocol=BASE / "matched_source_test_20261106._de5ty3e/protocol.json",
        earlier57=BASE / "heldout_test_20261030.rcyt1b3z/test_data.json")
    source = ROOT / previous["source"]
    sources = [Path(__file__), ROOT / "experiments/cluster_selection/reinforcement/scale_data.py",
        ROOT / "experiments/cluster_selection/supervised/collect.py",
        ROOT / "experiments/constraint_networks/constraint_networks.py",
        ROOT / "experiments/constraint_networks/mac.py",
        ROOT / "experiments/constraint_networks/generalized_arc_consistency.py"]
    plan = dict(seed=20261113, requested=16, source=source_identity(source),
        input_sha256={str(p): digest(p) for p in paths.values()},
        preparation_source_sha256={str(p): digest(p) for p in sources},
        include="First <=35-given, source-valid, exact-grid/reference-solution-disjoint, GAC-stalled, proven-unique roots in seeded row-group order.",
        exclusions="All declared earlier training/development/evaluation grids and known reference solutions, Nikoli100, source validation grids, and internal duplicates.",
        performance_filter=False, policy_queries=False, nodes_for_each_uniqueness_proof=10000,
        seconds_for_each_uniqueness_proof=10, symmetry_duplicates_excluded=False,
        requested_evaluation=dict(methods=METHODS, allowance=32, decoder="sampling_grow", branching="name",
            decoder_steps=8192, heuristic_steps=8192, nodes=10000, seconds=600, total_work=None, limits=LIMITS),
        note="Fresh 16-instance held-out pilot. It must never become training or tuning data for the concurrent continuations.")
    if (OUT / "predeclaration.json").exists():
        assert read(OUT / "predeclaration.json") == json.loads(json.dumps(plan))
    else:
        write(OUT / "predeclaration.json", plan)
    loaded = {name: read(path) for name, path in paths.items() if path.suffix == ".json"}
    excluded, blocked = set(), set()

    def identities(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "puzzles" and isinstance(item, list):
                    excluded.update(normalize(p) for p in item if isinstance(p, str) and len(p) == 81)
                elif key == "source_solution_sha256" and isinstance(item, str):
                    blocked.add(item)
                elif key == "solution" and isinstance(item, str) and len(item) == 81:
                    blocked.add(sha256(item.encode()).hexdigest())
                else:
                    identities(item)
        elif isinstance(value, list):
            for item in value:
                identities(item)

    for item in loaded.values():
        identities(item)
    nikoli = pq.read_table(paths["nikoli100"], columns=["initial_board", "solution"]).to_pylist()
    excluded.update(normalize(r["initial_board"]) for r in nikoli)
    blocked.update(sha256(r["solution"].encode()).hexdigest() for r in nikoli if isinstance(r["solution"], str))
    excluded.update(reserve_validation(source.parent))
    raw = loaded["full10k"]["network"]
    parquet = pq.ParquetFile(source)
    groups = list(range(parquet.num_row_groups))
    Random(plan["seed"]).shuffle(groups)
    counts, seen, selected, accepted_solutions = Counter(), set(), [], set()

    def batches():
        batch = []
        for group in groups:
            for offset, row in enumerate(parquet.read_row_group(group, columns=["puzzle", "solution"]).to_pylist()):
                counts["considered"] += 1
                puzzle = normalize(row["puzzle"])
                solution_hash = sha256(str(row["solution"]).encode()).hexdigest()
                reason = ("excluded_grid" if puzzle in excluded else "duplicate_grid" if puzzle in seen else
                    "above_35_givens" if sum(c != "." for c in puzzle) > 35 else
                    "invalid_source" if not _valid_solution(puzzle, row["solution"]) else
                    "excluded_solution" if solution_hash in blocked else None)
                if reason:
                    counts[reason] += 1
                    continue
                seen.add(puzzle)
                batch.append(dict(puzzle=puzzle, solution=row["solution"], row_group=group, offset=offset,
                    source_solution_sha256=solution_hash, puzzle_sha256=sha256(puzzle.encode()).hexdigest()))
                if len(batch) == 16:
                    yield batch
                    batch = []

    stream = iter(batches())
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"),
                             initializer=initialize_preparation, initargs=(raw,)) as pool:
        while len(selected) < 16:
            wave = [pool.submit(check_batch, batch) for _ in range(4)
                    if (batch := next(stream, None)) is not None]
            if not wave:
                raise RuntimeError("Source exhausted")
            for job in wave:  # Consume in source order, independent of worker completion.
                for status, row in job.result():
                    counts[status] += 1
                    if status == "unique" and row["source_solution_sha256"] not in accepted_solutions and len(selected) < 16:
                        selected.append(row)
                        accepted_solutions.add(row["source_solution_sha256"])
            print("Reserved", len(selected), "/16; checked", sum(counts[k] for k in ("unique", "multiple", "unknown", "gac_solved")), flush=True)
    assert len(selected) == len(accepted_solutions) == 16
    assert not {r["puzzle"] for r in selected} & excluded and not accepted_solutions & blocked
    assert source_identity(source) == plan["source"]
    assert all(digest(p) == h for p, h in {**plan["input_sha256"], **plan["preparation_source_sha256"]}.items())
    for row in selected:
        original = parquet.read_row_group(row["row_group"], columns=["puzzle", "solution"]).to_pylist()[row["offset"]]
        assert normalize(original["puzzle"]) == row["puzzle"] and original["solution"] == row["solution"]
    write(OUT / "unique_test.json", dict(version=1, network=raw, reward_colors=loaded["full10k"]["reward_colors"],
        puzzles=[r["puzzle"] for r in selected]))
    write(OUT / "preparation.json", dict(predeclaration_sha256=digest(OUT / "predeclaration.json"),
        data_sha256=digest(OUT / "unique_test.json"), counts=dict(counts), excluded_grids=len(excluded),
        excluded_solutions=len(blocked), records=[{k:v for k,v in r.items() if k not in ("puzzle", "solution")} for r in selected],
        no_model_queries=True, uniqueness_scope="81 primary variables; original reference validation and complete blocked UNSAT"))
    print("COMPLETE model-free preparation", flush=True)


def freeze():
    import ast
    import torch
    prep = read(OUT / "preparation.json")
    assert digest(OUT / "unique_test.json") == prep["data_sha256"]
    # Only evaluator accounting changed after the model-free reservation.
    declaration = read(OUT / "predeclaration.json")
    archived = OUT / "preparation_run.py"
    assert digest(archived) == declaration["preparation_source_sha256"][str(Path(__file__))]
    original = {n.name:ast.dump(n) for n in ast.parse(archived.read_text()).body if isinstance(n, ast.FunctionDef)}
    revised = {n.name:ast.dump(n) for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    assert all(original[k] == revised[k] for k in ("initialize_preparation", "check_batch", "prepare"))
    snapshots = {
        "initial": (BASE / "reinforce_sampling_grow_gpu_20261106.q3mc4c0g/job_7906.0/policy.initial.pt", "024d5a176c8fe80b15640e03e8d4c9bf4b31676822a19a13d23bc0dc61373670"),
        "teacher": (BASE / "hybrid_late_20261113.eed1ut6f/teacher.pt", "d340b2f7167079baa5702f739d555afe2213dd450341d19de6b6c0cf45633a4c"),
        "reinforce": (BASE / "hybrid_late_20261113.eed1ut6f/reinforce.pt", "3134d8bd743f381d4f85352fc8f0bdf1dbfc30586d766413c833fc3cd04fe615")}
    checkpoints = {}
    for name, (source, expected) in snapshots.items():
        contents = source.read_bytes()
        assert sha256(contents).hexdigest() == expected
        saved = torch.load(BytesIO(contents), map_location="cpu", weights_only=True)
        destination = OUT / f"{name}.pt"
        if destination.exists():
            assert digest(destination) == expected
        else:
            with destination.open("xb") as stream:
                stream.write(contents)
        assert saved["kind"] == "construction" and saved["beta"] == 0
        assert all(saved["config"][k] == v for k,v in LIMITS.items()) and saved["config"]["decoder"] == "sampling_grow"
        checkpoints[name] = dict(path=str(destination), source=str(source), sha256=expected,
            epoch=saved["training"]["epoch"], config=saved["config"])
    paths = [Path(__file__)]
    for directory in ["experiments/cluster_selection", "experiments/cluster_selection/reinforcement",
                      "experiments/constraint_networks", "tnreason/engine"]:
        paths.extend((ROOT / directory).glob("*.py"))
    order = [(method, i) for i in range(16) for method in METHODS]
    Random(20261113).shuffle(order)
    plan = dict(data=str(OUT / "unique_test.json"), data_sha256=prep["data_sha256"],
        preparation_sha256=digest(OUT / "preparation.json"), checkpoints=checkpoints,
        preparation_driver_archive=dict(path=str(archived), sha256=digest(archived),
            reason="Preserve exact preparation source before fixing incomplete-contraction timeout accounting; preparation function ASTs unchanged."),
        source_sha256={str(p):digest(p) for p in paths}, methods=METHODS, order=order,
        allowance=32, decoder_steps=8192, heuristic_steps=8192, limits=LIMITS,
        branching="name", nodes=10000, seconds=600, total_work=None, workers=4,
        claim="Fresh, prespecified 16-puzzle held-out pilot; one training seed; equal contraction caps, not equal tensor work or time.")
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == json.loads(json.dumps(plan))
    else:
        write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir(exist_ok=True)
    print("FROZEN evaluation; no test solving yet", flush=True)


def initialize_evaluation():
    import numpy as np
    import torch
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(OUT / "protocol.json")
    assert all(digest(p)==h for p,h in plan["source_sha256"].items())
    assert digest(plan["data"]) == plan["data_sha256"]
    data = read(plan["data"])
    domains, _ = instances(data)
    models = {}
    for name, info in plan["checkpoints"].items():
        assert digest(info["path"]) == info["sha256"]
        saved = torch.load(info["path"], map_location="cpu", weights_only=True)
        assert not instance_hashes(data, domains) & set(saved["training"]["train_instance_hashes"])
        model = runtime.load(info["path"]).cpu().eval()
        assert model.network == data["network"] and model.config == info["config"]
        models[name] = model
    assert models["initial"].config == models["reinforce"].config
    raw = data["network"]
    network = ConstraintNetwork({c["key"]:NumpyCore(values=np.asarray(c["values"],dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = partial(network.solve_mac, variable_tie_break="name")
    CONTEXT.update(plan=plan, models=models, network=network, domains=domains)


def evaluate_case(task):
    import torch
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.reinforcement.experiment import solve
    method, position = task
    plan = CONTEXT["plan"]
    path = OUT / "rows" / f"{method}_{position:02d}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    model = CONTEXT["models"].get(method, CONTEXT["models"]["reinforce"])
    original_attach, original_heuristic, original_apply = runtime.attach, runtime.attach_heuristic, runtime.apply_candidate
    applications, episodes = [], []

    def observed(state, candidate, prepared=None, **limits):
        root = state.decision_level == 0
        result = original_apply(state, candidate, prepared, **limits)
        applications.append(dict(root=root, removed=result.removed_values, contradiction=result.contradiction))
        return result

    def attached(function, state, *args, **options):
        result = function(state, *args, **options)
        policy = state.on_closure
        def closure(state, changed):
            start = len(applications)
            try:
                return policy(state, changed)
            finally:
                count = len(applications)-start
                assert count <= 32
                episodes.append(count)
        closure.statistics = policy.statistics
        state.on_closure = closure
        return result

    runtime.apply_candidate = observed
    runtime.attach = partial(attached, original_attach)
    runtime.attach_heuristic = partial(attached, original_heuristic)
    try:
        with torch.no_grad():
            row = solve(CONTEXT["network"], CONTEXT["domains"][position],
                method if method in ("plain", "heuristic") else "trained", model,
                nodes=10000, seconds=600, total_work=None, baseline_candidates=32,
                cluster_attempts=32, baseline_steps=8192)
    finally:
        runtime.attach, runtime.attach_heuristic, runtime.apply_candidate = original_attach, original_heuristic, original_apply
    assert row["status"] != "invalid"
    incomplete = row["work"]["contractions"] - len(applications)
    assert incomplete >= 0 and (incomplete == 0 or row["status"] == "wall_timeout")
    row.update(method=method, position=position, protocol_sha256=digest(OUT / "protocol.json"),
        episode_contractions=episodes, root_contractions=sum(a["root"] for a in applications),
        root_useful=sum(a["root"] and (a["removed"]>0 or a["contradiction"]) for a in applications),
        incomplete_contractions=incomplete)
    write(path, row)
    print(path.name, row["status"], row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
    return row


def evaluate():
    plan = read(OUT / "protocol.json")
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=initialize_evaluation) as pool:
        jobs = [pool.submit(evaluate_case, task) for task in plan["order"]]
        for job in as_completed(jobs):
            job.result()
    print("COMPLETE all 80 held-out solves", flush=True)


def report():
    import csv
    rows = [read(p) for p in sorted((OUT / "rows").glob("*.json"))]
    assert len(rows) == 80 and len({(r["method"],r["position"]) for r in rows}) == 80
    assert all(r["protocol_sha256"] == digest(OUT / "protocol.json") for r in rows)
    totals = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        complete = all(r["status"] == "solved" and r["valid"] for r in group)
        totals[method] = dict(instances=16, completed=sum(r["status"]=="solved" and r["valid"] for r in group),
            statuses=dict(Counter(r["status"] for r in group)), nodes=sum(r["nodes"] for r in group) if complete else None,
            branch_free=sum(r["status"]=="solved" and r["valid"] and r["nodes"]==1 for r in group),
            contractions=sum(r["work"]["contractions"] for r in group), estimated_work=sum(r["work"]["estimated_work"] for r in group),
            solve_seconds=sum(r["solve_s"] for r in group))
    pairs = {}
    for control in ("initial", "teacher", "heuristic", "plain"):
        paired = [(next(r for r in rows if r["method"]=="reinforce" and r["position"]==i),
                   next(r for r in rows if r["method"]==control and r["position"]==i)) for i in range(16)]
        valid = [(a,b) for a,b in paired if a["status"]==b["status"]=="solved" and a["valid"] and b["valid"]]
        pairs[control] = dict(paired_complete=len(valid), wins=sum(a["nodes"]<b["nodes"] for a,b in valid),
            ties=sum(a["nodes"]==b["nodes"] for a,b in valid), losses=sum(a["nodes"]>b["nodes"] for a,b in valid))
    summary = dict(protocol_sha256=digest(OUT / "protocol.json"), results=totals, reinforce_comparisons=pairs,
        row_sha256={str(p):digest(p) for p in sorted((OUT/"rows").glob("*.json"))})
    write(OUT / "summary.json", summary)
    with (OUT / "nodes.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["position", "method", "status", "nodes", "contractions", "estimated_work", "solve_seconds"])
        for row in sorted(rows,key=lambda r:(r["position"],METHODS.index(r["method"]))):
            writer.writerow([row["position"],row["method"],row["status"],row["nodes"],row["work"]["contractions"],row["work"]["estimated_work"],row["solve_s"]])
    text = ["Fresh 16-puzzle held-out pilot; frozen before solver queries.",
        "Same name branching, 32 contractions/episode, 8192 decoder/heuristic steps and per-cluster bounds; no total-work cap.",
        "Equal caps do not imply equal actual contractions, tensor work or time. Concurrent CPU times are descriptive.",
        "Censored results are explicit and prevent a complete total. One training seed and only sixteen new puzzles limit generalization claims.", "", "Method | DFS nodes | Without branching | Completed"]
    text.extend(f"{name} | {r['nodes']} | {r['branch_free']}/16 | {r['completed']}/16" for name,r in totals.items())
    text.extend(["", "REINFORCE wins/ties/losses against controls (complete paired cases only):",json.dumps(pairs,indent=2)])
    with (OUT / "REPORT.txt").open("x") as stream:
        stream.write("\n".join(text)+"\n")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "freeze", "evaluate", "report"))
    globals()[parser.parse_args().phase]()
