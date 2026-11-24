"""Plain MAC and the 51-family expert on the original 55 development puzzles."""
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from math import log
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import sleep

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))

import torch
from demonstrations.sudoku.inferenceClusters import RuleLimits, available_rules
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement.experiment import instances, solve
from experiments.cluster_selection.reinforcement.refine import source_hashes
from experiments.cluster_selection.reinforcement.refine_eval import evaluate_baseline
from experiments.cluster_selection.reinforcement.train import _state


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    gate = BASE / "objective_group_seed0/final.pt"
    while not gate.exists():
        sleep(15)
    torch.set_num_threads(1)
    data_file = BASE.parent / "unique_training_10k_20261106.f9b9837e/development.json"
    parent = BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    engine_path = BASE.parent / "learning_diagnostics_20261113.67i11ilh/matched_time_20261115.PW6lJf/run.py"
    out = Path(mkdtemp(prefix="full55_references_", dir=BASE))
    sources = source_hashes()
    hashes = {str(path): digest(path) for path in (data_file, parent, engine_path, Path(__file__))}
    data, model = json.loads(data_file.read_text()), load(parent)
    assert len(set(data["validation_indices"])) == 55
    plain = evaluate_baseline(model, data, mode="plain", attempts=32, seconds=30.)
    assert plain["totals"]["dfs_complete"] == 55
    write(out / "plain.json", plain)
    print("plain", plain["totals"], flush=True)
    spec = importlib.util.spec_from_file_location("full55_reference_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    (out / "rows").mkdir()
    references = {}
    for position, row in enumerate(plain["rows"]):
        path = out / f"reference_{position:02d}.json"
        write(path, dict(assignment=row["residual"]["assignment"], source="Separate plain-MAC solve; verification only"))
        references[str(position)] = str(path)
    families, limits = list(available_rules()), RuleLimits()
    assert len(families) == 51
    plan = dict(seconds=30., budgets_s=[30.], expert_limits=asdict(limits), expert_families=families,
        reference_assignments=references, source_sha256=hashes, data=str(data_file),
        data_sha256=digest(data_file), validation_indices=data["validation_indices"], dependency_sha256=sources,
        scope="Original development validation; no test100 inputs or outcomes", timing="CPU 1 on a shared host; descriptive wall times")
    write(out / "protocol.json", plan)
    raw = data["network"]
    network = _state({**data, "samples": [dict(domains=[list(range(n)) for n in raw["dimensions"]])]}).constraint_network
    initial = instances(data)[0]
    engine.OUT = out
    engine.CONTEXT.update(plan=plan, network=network, models={}, runtime=runtime,
        domains=[initial[index] for index in data["validation_indices"]])
    rows = []
    for position, index in enumerate(data["validation_indices"]):
        original_solve = network.solve_mac
        network.solve_mac = engine.forbid
        try:
            recorded = engine.evaluate("expert", position)
        finally:
            network.solve_mac = original_solve
        final = recorded["snapshots"][-1]["domains"]
        assert recorded["reference_retained"] and recorded["no_added_cores"] and not recorded["dfs_called"]
        entropy = lambda domains: sum(log(len(domains[color])) for color in data["reward_colors"])
        before = entropy(recorded["snapshots"][0]["domains"])
        root = dict(status=recorded["status"], solved=recorded["status"] == "solved", domains=final,
            baseline_solved=before == 0, progress=1 - entropy(final) / before if before else 1.,
            solve_s=recorded["wall_s"], expert_raw=str(out / "rows" / f"expert_{position:02d}.json"))
        residual = solve(network, final, "plain", model, nodes=10000, seconds=30.)
        assert residual["status"] not in {"invalid", "unsat"}
        rows.append(dict(index=index, root=root, residual=residual))
        print(f"expert {position + 1}/55: roots {sum(row['root']['solved'] for row in rows)}", flush=True)
    complete = sum(row["residual"]["status"] == "solved" for row in rows)
    totals = dict(instances=55, root_solved=sum(row["root"]["solved"] for row in rows), dfs_complete=complete,
        dfs_nodes=sum(row["residual"]["nodes"] for row in rows) if complete == 55 else None,
        root_seconds=sum(row["root"]["solve_s"] for row in rows), residual_seconds=sum(row["residual"]["solve_s"] for row in rows))
    write(out / "expert.json", dict(rows=rows, totals=totals, settings=plan))
    assert sources == source_hashes() and all(digest(path) == value for path, value in hashes.items())
    write(out / "audit.json", dict(sources_unchanged=True, retained_references=55, root_never_called_dfs=True))
    print(out, totals, flush=True)


if __name__ == "__main__":
    main()
