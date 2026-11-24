"""Reserve 100 fresh, uniquely solvable, GAC-stalled roots without policy queries."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
BASE = OUT.parent
OLD = BASE / "learning_diagnostics_20261113.67i11ilh/heldout"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("heldout16_preparation", OLD / "preparation_run.py")
HELPERS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPERS
SPEC.loader.exec_module(HELPERS)
read, write, digest = HELPERS.read, HELPERS.write, HELPERS.digest


def initialize(raw):
    HELPERS.initialize_preparation(raw)


def check(rows):
    return HELPERS.check_batch(rows)


def identities(value, grids, solutions, normalize):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "puzzles" and isinstance(item, list):
                grids.update(normalize(p) for p in item if isinstance(p, str) and len(p) == 81)
            elif key == "puzzle" and isinstance(item, str) and len(item) == 81:
                grids.add(normalize(item))
            elif key == "source_solution_sha256" and isinstance(item, str):
                solutions.add(item)
            elif key == "solution" and isinstance(item, str) and len(item) == 81:
                solutions.add(sha256(item.encode()).hexdigest())
            else:
                identities(item, grids, solutions, normalize)
    elif isinstance(value, list):
        for item in value:
            identities(item, grids, solutions, normalize)


def main():
    import pyarrow.parquet as pq
    from experiments.cluster_selection.reinforcement.scale_data import normalize, _valid_solution, reserve_validation
    from experiments.cluster_selection.supervised.collect import sudoku_domains
    if (OUT / "preparation.json").exists():
        completed = read(OUT / "preparation.json")
        assert digest(OUT / "data.json") == completed["data_sha256"]
        assert all(digest(p) == h for p, h in completed["reference_sha256"].items())
        print("Reusing completed 100-puzzle reservation", flush=True)
        return
    old_plan = read(OLD / "predeclaration.json")
    inputs = dict(old_plan["input_sha256"])
    inputs.update({str(p): digest(p) for p in (OLD / "unique_test.json", OLD / "preparation.json",
                                             OLD / "predeclaration.json")})
    assert all(digest(path) == expected for path, expected in inputs.items())
    source = Path(old_plan["source"]["path"])
    source_code = [Path(__file__), OLD / "preparation_run.py", ROOT / "experiments/cluster_selection/reinforcement/scale_data.py",
        ROOT / "experiments/cluster_selection/supervised/collect.py",
        ROOT / "experiments/constraint_networks/constraint_networks.py",
        ROOT / "experiments/constraint_networks/mac.py",
        ROOT / "experiments/constraint_networks/generalized_arc_consistency.py"]
    plan = dict(seed=20261116, requested=100, source=HELPERS.source_identity(source), input_sha256=inputs,
        source_sha256={str(p): digest(p) for p in source_code}, workers=4, batch_size=16,
        include="First <=35-given, source-valid, exact-grid/reference-solution-disjoint, GAC-stalled, proven-unique roots in seeded row-group order.",
        exclusions="All inputs excluded by the earlier 16-puzzle pilot, that pilot itself, all source validation grids, and internal exact-grid/reference-solution duplicates.",
        nodes_per_uniqueness_solve=10000, seconds_per_uniqueness_solve=10,
        policy_queries=False, performance_filter=False, symmetry_duplicates_excluded=False,
        note="Fresh held-out evaluation reservation; do not use for training or checkpoint selection. Uniqueness filtering uses bounded plain MAC, never a learned policy.")
    if (OUT / "predeclaration.json").exists():
        assert read(OUT / "predeclaration.json") == plan
    else:
        write(OUT / "predeclaration.json", plan)
    excluded, blocked = set(), set()
    loaded = {path: read(path) for path in inputs if Path(path).suffix == ".json"}
    for item in loaded.values():
        identities(item, excluded, blocked, normalize)
    nikoli_path = next(path for path in inputs if Path(path).name == "nikoli100.parquet")
    for row in pq.read_table(nikoli_path, columns=["initial_board", "solution"]).to_pylist():
        excluded.add(normalize(row["initial_board"]))
        if isinstance(row["solution"], str):
            blocked.add(sha256(row["solution"].encode()).hexdigest())
    excluded.update(reserve_validation(source.parent))
    full10k = next(item for path, item in loaded.items() if "unique_training_10k_" in path and Path(path).name == "development.json")
    raw = full10k["network"]
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
                if len(batch) == plan["batch_size"]:
                    yield batch
                    batch = []
        if batch:
            yield batch

    stream = iter(batches())
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"),
                             initializer=initialize, initargs=(raw,)) as pool:
        while len(selected) < plan["requested"]:
            wave = [pool.submit(check, batch) for _ in range(plan["workers"])
                    if (batch := next(stream, None)) is not None]
            if not wave:
                raise RuntimeError("Source exhausted before 100 eligible roots")
            for job in wave:  # Source order makes selection independent of worker scheduling.
                for status, row in job.result():
                    counts["checked"] += 1
                    counts[status] += 1
                    if status == "unique":
                        if row["source_solution_sha256"] in accepted_solutions:
                            counts["duplicate_solution"] += 1
                        elif len(selected) < plan["requested"]:
                            selected.append(row)
                            accepted_solutions.add(row["source_solution_sha256"])
                        else:
                            counts["eligible_overshoot"] += 1
            print(f"Reserved {len(selected)}/100; checked {counts['checked']}; GAC-solved {counts['gac_solved']}; "
                  f"multiple {counts['multiple']}; unknown {counts['unknown']}", flush=True)
    assert len(selected) == len(accepted_solutions) == 100
    assert not {row["puzzle"] for row in selected} & excluded and not accepted_solutions & blocked
    initialize(raw)
    network = HELPERS.CONTEXT["state"].constraint_network
    references = {}
    (OUT / "references").mkdir(exist_ok=True)
    for position, row in enumerate(selected):
        original = parquet.read_row_group(row["row_group"], columns=["puzzle", "solution"]).to_pylist()[row["offset"]]
        assert normalize(original["puzzle"]) == row["puzzle"] and original["solution"] == row["solution"]
        state = network.create_mac_state(sudoku_domains(row["solution"]))
        assert state.propagate() and state.is_solved() and state.decision_level == 0
        assignment = state.assignment()
        assert set(assignment) == set(network.dimension_dict)
        assert all(assignment[c] in values for c, values in sudoku_domains(row["puzzle"]).items())
        assert all(core[{c: assignment[c] for c in core.colors}] != 0 for core in network.coresDict.values())
        path = OUT / "references" / f"{position:02d}.json"
        write(path, dict(position=position, assignment=assignment, source_solution_sha256=row["source_solution_sha256"],
                         derived_by="GAC from complete source solution; all original cores checked; no branching"))
        references[str(path)] = digest(path)
    assert HELPERS.source_identity(source) == plan["source"]
    assert all(digest(path) == expected for path, expected in {**inputs, **plan["source_sha256"]}.items())
    write(OUT / "data.json", dict(version=1, network=raw, reward_colors=full10k["reward_colors"],
                                 puzzles=[row["puzzle"] for row in selected]))
    write(OUT / "preparation.json", dict(predeclaration_sha256=digest(OUT / "predeclaration.json"),
        data_sha256=digest(OUT / "data.json"), counts=dict(counts), excluded_grids=len(excluded),
        excluded_solutions=len(blocked), records=[dict(position=i, **row) for i, row in enumerate(selected)],
        reference_sha256=references, no_model_queries=True, no_performance_filter=True,
        uniqueness_scope="81 primary variables; valid reference and complete blocked UNSAT",
        source_rows_replayed=True, all_reference_assignments_valid=True))
    print("COMPLETE: 100 fresh, proven-unique, GAC-stalled roots and validated references", flush=True)


if __name__ == "__main__":
    main()
