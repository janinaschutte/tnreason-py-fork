"""Fresh 100-puzzle replication of the frozen eight-method timed comparison.

Reuse the measured propagation and residual-DFS routines without changing them.
Each published row is resumable; startup and residual DFS remain separately timed.
"""
import argparse
from copy import deepcopy
import csv
import importlib.util
from io import StringIO
import json
from pathlib import Path
from random import Random
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
OLD = OUT.parent / "learning_diagnostics_20261113.67i11ilh/matched_time_20261115.PW6lJf"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("frozen_timed_solver", OLD / "run.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
solver.OUT = OUT


def freeze():
    import torch
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
    parent = solver.read(OLD / "protocol.json")
    data = solver.read(OUT / "data.json")
    preparation = solver.read(OUT / "preparation.json")
    overlap = solver.read(OUT / "overlap_audit.json")
    assert overlap["status"] == "passed" and overlap["instances"] == 100
    assert overlap["data_sha256"] == preparation["data_sha256"]
    assert solver.digest(OUT / "data.json") == preparation["data_sha256"]
    assert solver.digest(OUT / "predeclaration.json") == preparation["predeclaration_sha256"]
    declaration = solver.read(OUT / "predeclaration.json")
    assert not declaration["policy_queries"] and not declaration["performance_filter"]
    assert all(solver.digest(Path(path)) == expected for path, expected in
               {**declaration["input_sha256"], **declaration["source_sha256"]}.items())
    assert all(solver.digest(Path(path)) == expected for path, expected in preparation["reference_sha256"].items())
    assert len(data["puzzles"]) == len(set(data["puzzles"])) == 100
    assert data["network"] == solver.read(Path(parent["data"]))["network"]
    test_hashes = instance_hashes(data, instances(data)[0])
    for info in parent["checkpoints"].values():
        assert solver.digest(Path(info["path"])) == info["sha256"]
        saved = torch.load(info["path"], map_location="cpu", weights_only=True)
        assert not test_hashes & set(saved["training"]["train_instance_hashes"])
        assert saved["config"] == info["config"] and saved["beta"] == info["beta"]
    assert set(overlap["checkpoints"]) == set(parent["checkpoints"])
    assert all(item["train_fingerprint_overlap"] == 0 and item["all_validation_candidates_excluded"]
               and item["sha256"] == parent["checkpoints"][name]["sha256"]
               for name, item in overlap["checkpoints"].items())
    references = {str(i): str(OUT / "references" / f"{i:02d}.json") for i in range(100)}
    plan = deepcopy(parent)
    plan.update(kind="fresh_100_matched_time_root_propagation", data=str(OUT / "data.json"),
        data_sha256=solver.digest(OUT / "data.json"), positions=list(range(100)), seed=2026111601,
        parent_protocol=str(OLD / "protocol.json"), parent_protocol_sha256=solver.digest(OLD / "protocol.json"),
        preparation_sha256=solver.digest(OUT / "preparation.json"), reference_assignments=references,
        reference_sha256={path: solver.digest(Path(path)) for path in references.values()},
        interpretation="Fresh 100 exact-grid/reference-solution-disjoint, GAC-stalled, proven-unique "
        "Sudokus sampled before any policy evaluation. Same selected checkpoints, decoders, source, "
        "single pinned CPU/thread, cluster limits, 1/5/15/60-second root propagation prefixes, "
        "and separately timed residual plain MAC as the earlier 16-puzzle comparison. "
        "No mix, retraining, checkpoint selection, or filtering on cluster/solver performance. "
        "Sudoku symmetries are not canonicalized. Expert retains its different, larger limits.")
    plan["order"] = [(method, i) for method in solver.METHODS for i in plan["positions"]]
    Random(plan["seed"]).shuffle(plan["order"])
    for path in (Path(__file__), OUT / "prepare.py", OUT / "overlap_audit.json", OLD / "protocol.json"):
        plan["source_sha256"][str(path)] = solver.digest(path)
    assert all(solver.digest(Path(path)) == expected for path, expected in plan["source_sha256"].items())
    if (OUT / "protocol.json").exists():
        assert solver.read(OUT / "protocol.json") == json.loads(json.dumps(plan))
    else:
        solver.write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir(exist_ok=True)
    print("FROZEN: 100 fresh puzzles, eight fixed methods, 800 timed roots", flush=True)


def report():
    plan = solver.read(OUT / "protocol.json")
    expected = set(map(tuple, plan["order"]))
    found, totals, details = set(), [], []
    for method in solver.METHODS:
        group = []
        for position in plan["positions"]:
            path = OUT / "rows" / f"{method}_{position:02d}.json"
            row, residual = solver.read(path), solver.read(path.with_suffix(".dfs.json"))
            assert row["protocol_sha256"] == residual["protocol_sha256"] == solver.digest(OUT / "protocol.json")
            assert residual["root_row_sha256"] == solver.digest(path)
            assert row["method"] == residual["method"] == method
            assert row["position"] == residual["position"] == position
            assert row["reference_retained"] and row["no_added_cores"] and not row["dfs_called"]
            assert row["affinity"] == [plan["cpu"]] and row["decision_level"] == 0
            found.add((method, position))
            group.append((row, residual))
        for budget in plan["budgets_s"]:
            nodes, censored, solved = [], 0, 0
            for row, residual in group:
                prefix, finish = row["prefixes"][str(budget)], residual["budgets"][str(budget)]
                assert prefix["snapshot"] == finish["snapshot"]
                actual = bool(row["valid"] and row["solved_s"] is not None and row["solved_s"] <= budget)
                assert prefix["solved"] == actual
                if actual:
                    assert finish["valid"] and finish["nodes"] == 1
                complete = finish["solved"] and finish["valid"] and finish["nodes"] is not None
                if complete:
                    nodes.append(finish["nodes"])
                else:
                    censored += 1
                solved += actual
                details.append(dict(method=method, position=row["position"], budget_s=budget,
                    solved_without_branching=int(actual), remaining_dfs_nodes=finish["nodes"],
                    residual_status=finish["status"], residual_s=finish.get("solve_s"),
                    solved_s=row["solved_s"], trajectory_wall_s=row["wall_s"], trajectory_cpu_s=row["cpu_s"]))
            totals.append(dict(method=method, budget_s=budget, instances=len(group),
                solved_without_branching=solved, remaining_dfs_nodes=None if censored else sum(nodes),
                residual_censored=censored))
    assert found == expected and len(found) == 800
    for name, records in (("totals", totals), ("per_puzzle", details)):
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        path = OUT / f"matched_time_{name}.csv"
        if path.exists():
            assert path.read_text() == buffer.getvalue()
        else:
            with path.open("x") as handle:
                handle.write(buffer.getvalue())
    paths = sorted((OUT / "rows").glob("*.json"))
    assert len(paths) == 1600
    solver.write(OUT / "summary.json", dict(protocol_sha256=solver.digest(OUT / "protocol.json"),
        totals=totals, row_sha256={p.name: solver.digest(p) for p in paths}))
    print(json.dumps(totals, indent=2), flush=True)


def plot():
    spec = importlib.util.spec_from_file_location("frozen_timed_plot", OLD / "plot.py")
    drawing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drawing)
    plan = solver.read(OUT / "protocol.json")
    summary = solver.read(OUT / "summary.json")
    assert summary["protocol_sha256"] == solver.digest(OUT / "protocol.json")
    assert all(solver.digest(OUT / "rows" / name) == expected
               for name, expected in summary["row_sha256"].items())
    rows = {method: [(solver.read(OUT / "rows" / f"{method}_{i:02d}.json"),
                     solver.read(OUT / "rows" / f"{method}_{i:02d}.dfs.json"))
                    for i in plan["positions"]] for method in solver.METHODS}
    drawing.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 13, "axes.titlesize": 15, "legend.fontsize": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = drawing.plt.subplots(figsize=(10.8, 6.9))
    drawing.root_curves(ax, plan, rows)
    ax.set_ylim(-1, 103)
    ax.set_yticks(range(0, 101, 10))
    ax.set_title("Solving 100 held-out Sudokus without branching", pad=18)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .04),
               frameon=False, columnspacing=1.4, handlelength=2.7)
    fig.subplots_adjust(left=.09, right=.975, top=.88, bottom=.24)
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"matched_time_roots_no_caption.{extension}"
        if not path.exists():
            fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.12)
    drawing.plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "report", "plot"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "report":
        report()
    elif args.phase == "plot":
        plot()
    else:
        solver.initialize()
        for method, position in solver.CONTEXT["plan"]["order"]:
            solver.evaluate(method, position)
        for method, position in solver.CONTEXT["plan"]["order"]:
            solver.residual(method, position)
        if not (OUT / "summary.json").exists():
            report()
        if not all((OUT / f"matched_time_roots_no_caption.{extension}").exists()
                   for extension in ("png", "pdf", "svg")):
            plot()
