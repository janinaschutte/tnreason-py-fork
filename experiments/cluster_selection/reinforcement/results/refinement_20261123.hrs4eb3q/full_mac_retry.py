"""Retry only censored whole-MAC rows at 300s; retain immutable 30s results."""
import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

from full_mac import BASE, digest, write
from experiments.cluster_selection.reinforcement import refine, refine_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    original = args.original.resolve()
    read = lambda path: json.loads(path.read_text())
    plan = read(original / "protocol.json")
    assert plan["sources"] == refine.source_hashes()
    assert all(digest(Path(path)) == wanted for path, wanted in plan["hashes"].items())
    output = Path(mkdtemp(prefix="fullmac300_", dir=BASE))
    inputs = {str(original / name / "result.json"): digest(original / name / "result.json")
              for name in ("plain", "parent", "global_context")}
    write(output / "protocol.json", dict(original=str(original), original_sha256=digest(original / "protocol.json"),
        input_reports=inputs, source_sha256=refine.source_hashes(), driver_sha256=digest(Path(__file__)),
        settings=dict(attempts=32, seconds=300., nodes=10000, seed=100000, full_mac=True),
        note="Retry only initially censored rows. Earlier solved rows are retained because proposal sampling "
             "is deterministic from current domains; the wall ceiling does not alter a completed search path. "
             "Composite timings are descriptive, not a fresh single-pass 300s timing measurement."))
    solver, summaries = refine_eval.solve, {}
    for name in ("plain", "parent", "global_context"):
        before = read(original / name / "result.json")
        censored = [row["index"] for row in before["rows"] if row["status"] not in {"solved", "unsat"}]
        folder = output / name
        folder.mkdir()
        replacements, positions = {}, []
        if censored:
            def observed(*pos, **kw):
                row = solver(*pos, **kw)
                index = censored[len(positions)]
                positions.append(index)
                write(folder / f"row_{index:02d}.json", dict(index=index, **row))
                return row

            refine_eval.solve = observed
            try:
                retried = refine_eval.main([plan["data"], "--checkpoint", plan["checkpoint_paths"][name],
                    "--output", str(folder / "retry.json"), "--full-mac", "--mode", "trained",
                    "--attempts", "32", "--seconds", "300", "--nodes", "10000", "--seed", "100000",
                    "--indices", *map(str, censored)])
            finally:
                refine_eval.solve = solver
            assert positions == censored
            replacements = {row["index"]: row for row in retried["rows"]}
        rows = [replacements.get(row["index"], row) for row in before["rows"]]
        complete = sum(row["status"] in {"solved", "unsat"} for row in rows)
        totals = dict(instances=len(rows), dfs_complete=complete,
            root_solved=sum(row["solved"] and row["nodes"] == 1 for row in rows),
            dfs_nodes=sum(row["nodes"] for row in rows) if complete == len(rows) else None,
            composite_solve_seconds=sum(row["solve_s"] for row in rows), retried_indices=censored,
            actual_evaluation_seconds_including_initial_censoring=before["totals"]["solve_seconds"]
                + sum(row["solve_s"] for row in replacements.values()))
        write(folder / "combined.json", dict(rows=rows, totals=totals,
            retained_from=str(original / name / "result.json"), scope="Full MAC with 300s ceiling; only censored rows rerun"))
        summaries[name] = totals
        print(name, json.dumps(totals), flush=True)
    assert plan["sources"] == refine.source_hashes()
    assert all(digest(Path(path)) == wanted for path, wanted in inputs.items())
    write(output / "summary.json", summaries)
    print(output, flush=True)


if __name__ == "__main__":
    main()
