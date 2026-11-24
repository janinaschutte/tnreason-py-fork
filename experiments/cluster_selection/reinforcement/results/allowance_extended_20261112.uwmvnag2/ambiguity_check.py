"""Look for a second valid assignment in the six unresolved allowance-1024 cases."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection.reinforcement.experiment import instances, solve, _valid_assignment
from experiments.cluster_selection.reinforcement.train import _state
from experiments.constraint_networks.mac import ConstraintNetworkSolver

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "allowance_age_20261112.w9v5x1hg"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    target = OUT / "ambiguity_check.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = read(OLD / "protocol.json")
    path = Path(protocol["data"])
    assert digest(path) == protocol["data_sha256"]
    data = read(path)
    indices = [80, 82, 110, 111, 112, 113]
    domains, _ = instances({**data, "puzzles": [data["puzzles"][i] for i in indices]})
    network = _state({**data, "samples": [{"domains": [list(range(d))
                     for d in data["network"]["dimensions"]]}]}).constraint_network
    results = []
    for index, initial in zip(indices, domains):
        source = OLD / "rows" / f"oldest_pruned_1024_{index}.json"
        baseline = read(source)
        known = baseline["assignment"]
        assert baseline["solved"] and baseline["valid"] and baseline["nodes"] > 1
        assert _valid_assignment(network, initial, known)
        state = network.create_mac_state(initial)
        assert state.propagate() and not state.is_solved()
        variable = ConstraintNetworkSolver(network)._select_color(state)
        modified = state.snapshot_domains()
        root_domain = modified[variable]
        forbidden = known[variable]
        modified[variable] = tuple(v for v in root_domain if v != forbidden)
        assert forbidden in root_domain and modified[variable]
        row = solve(network, modified, "plain", None, nodes=2000, seconds=30)
        alternate = row.pop("assignment", {})
        if row["solved"]:
            assert _valid_assignment(network, initial, alternate)
            assert _valid_assignment(network, modified, alternate)
            assert alternate[variable] != forbidden and alternate != known
        result = dict(puzzle=index, baseline_sha256=digest(source), variable=variable,
            gac_domain=root_domain, forbidden_value=forbidden, status=row["status"],
            nodes=row["nodes"], solve_s=row["solve_s"],
            proves_multiple_solutions=bool(row["solved"]),
            interpretation=("Distinct full valid assignment found" if row["solved"] else
                "Selected variable forced; no uniqueness claim" if row["status"] == "unsat" else
                "Unresolved within diagnostic limits"),
            alternate_assignment_delta={k: v for k, v in alternate.items() if known[k] != v})
        results.append(result)
        print(index, result["status"], result["nodes"],
              f"{result['solve_s']:.3f}s", result["proves_multiple_solutions"], flush=True)
    report = dict(method="Forbid the known solution's value at the MRV/degree/name variable "
        "after plain root GAC, then run plain MAC. Validate every full assignment against "
        "all original cores and givens. Witness is reconstructed by applying saved delta "
        "to the referenced baseline assignment. This is a selected-variable check, not "
        "a complete uniqueness test.", data=str(path), data_sha256=digest(path),
        source_sha256=digest(Path(__file__)), guards=dict(nodes=2000, seconds=30),
        no_test_queries=True, threads=1, rows=results)
    with target.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
