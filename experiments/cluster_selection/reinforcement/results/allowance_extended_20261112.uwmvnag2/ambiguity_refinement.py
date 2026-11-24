"""Check whether the forced-variable diagnostic also proves full uniqueness."""
import json
from pathlib import Path

import torch

from ambiguity_check import OUT, OLD, read, digest, instances, _state, _valid_assignment


def main():
    target = OUT / "ambiguity_refinement.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    previous = OUT / "ambiguity_check.json"
    diagnostic = read(previous)
    path = Path(diagnostic["data"])
    assert digest(path) == diagnostic["data_sha256"]
    data = read(path)
    indices = [row["puzzle"] for row in diagnostic["rows"]]
    domains, _ = instances({**data, "puzzles": [data["puzzles"][i] for i in indices]})
    network = _state({**data, "samples": [{"domains": [list(range(d))
                     for d in data["network"]["dimensions"]]}]}).constraint_network
    rows = []
    for row, initial in zip(diagnostic["rows"], domains):
        source = OLD / "rows" / f"oldest_pruned_1024_{row['puzzle']}.json"
        assert digest(source) == row["baseline_sha256"]
        known = read(source)["assignment"]
        assert _valid_assignment(network, initial, known)
        restricted = {**initial, row["variable"]: (row["forbidden_value"],)}
        state = network.create_mac_state(restricted)
        assert state.propagate()
        solved = state.is_solved()
        if solved:
            assignment = state.assignment()
            assert _valid_assignment(network, initial, assignment)
            assert _valid_assignment(network, restricted, assignment)
            assert assignment == known
        result = dict(puzzle=row["puzzle"], variable=row["variable"],
            fixed_value=row["forbidden_value"], gac_solved=solved,
            gac_nonsingleton_domains=sum(len(d) > 1 for d in state.domains.values()),
            proves_unique=bool(solved and row["status"] == "unsat"),
            interpretation=("Complete UNSAT for all other values, and plain GAC with the "
                "remaining value produces the known full valid assignment; unique solution proved"
                if solved and row["status"] == "unsat" else
                "Selected variable forced, but this check does not establish full uniqueness"))
        rows.append(result)
        print(row["puzzle"], result["gac_solved"], result["gac_nonsingleton_domains"],
              result["proves_unique"], flush=True)
    report = dict(method="Original givens plus selected variable fixed to known solution value, "
        "followed by plain GAC only; no additional DFS. Full solutions validated against all "
        "original cores and givens and compared with known assignment.",
        diagnostic_sha256=digest(previous), source_sha256=digest(Path(__file__)), rows=rows)
    with target.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
