"""Descriptive paired analysis of the small, existing-development RL pilot.

No solver or training is run. Missing final evaluations remain pending. Each
invocation writes a fresh analysis directory, preserving previous summaries.
"""
import argparse
import csv
from hashlib import sha256
import json
from math import exp, floor, log
from pathlib import Path
from random import Random
from statistics import mean, median
from tempfile import mkdtemp


HERE = Path(__file__).resolve().parent
BOOTSTRAPS, BOOTSTRAP_SEED = 10_000, 20261123


def interval(values):
    """Percentile interval for paired puzzle means, with a fixed resampling RNG."""
    if not values:
        return None
    rng, n = Random(BOOTSTRAP_SEED), len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(BOOTSTRAPS))

    def percentile(probability):
        position = (len(samples) - 1) * probability
        lower = floor(position)
        return samples[lower] + (position - lower) * (samples[min(lower + 1, len(samples) - 1)] - samples[lower])

    return [percentile(.025), percentile(.975)]


def compact(row, originals):
    residual, root = row["residual"], row["root"]
    if root["status"] in {"invalid", "unsat"} or residual["status"] in {"invalid", "unsat"}:
        raise ValueError("Invalid or contradictory result on a known satisfiable development puzzle")
    complete = residual["status"] == "solved" and residual.get("nodes") is not None
    return dict(index=row["index"], original_index=originals[row["index"]],
        root_solved=int(root["solved"]), dfs_complete=complete,
        dfs_nodes=residual["nodes"] if complete else None,
        root_seconds=root["solve_s"], residual_seconds=residual["solve_s"],
        progress=root.get("progress"), root_status=root["status"], residual_status=residual["status"])


def paired(rows, baseline):
    shared = sorted(set(rows) & set(baseline))
    complete = [idx for idx in shared if rows[idx]["dfs_complete"] and baseline[idx]["dfs_complete"]]
    deltas = [rows[idx]["dfs_nodes"] - baseline[idx]["dfs_nodes"] for idx in complete]
    root_deltas = [rows[idx]["root_solved"] - baseline[idx]["root_solved"] for idx in shared]
    ratios = [rows[idx]["dfs_nodes"] / baseline[idx]["dfs_nodes"] for idx in complete
              if min(rows[idx]["dfs_nodes"], baseline[idx]["dfs_nodes"]) > 0]
    largest = max(complete, key=lambda idx: abs(rows[idx]["dfs_nodes"] - baseline[idx]["dfs_nodes"])) if complete else None
    outlier = None
    if largest is not None:
        delta = rows[largest]["dfs_nodes"] - baseline[largest]["dfs_nodes"]
        outlier = dict(index=largest, original_index=rows[largest]["original_index"],
            baseline_nodes=baseline[largest]["dfs_nodes"], method_nodes=rows[largest]["dfs_nodes"],
            delta=delta, absolute_delta_share=abs(delta) / sum(map(abs, deltas)) if any(deltas) else 0.,
            total_delta_without_this_puzzle=sum(deltas) - delta,
            note="Largest absolute paired node difference; descriptive only. All puzzles remain in the criterion.")
    return dict(paired_puzzles=len(shared), complete_node_pairs=len(complete),
        censored_node_pairs=len(shared) - len(complete), nodes_improved=sum(x < 0 for x in deltas),
        nodes_equal=sum(x == 0 for x in deltas), nodes_worsened=sum(x > 0 for x in deltas),
        paired_baseline_nodes=sum(baseline[idx]["dfs_nodes"] for idx in complete),
        paired_method_nodes=sum(rows[idx]["dfs_nodes"] for idx in complete),
        total_node_delta=sum(deltas) if len(complete) == len(shared) else None,
        mean_node_delta=mean(deltas) if deltas else None, median_node_delta=median(deltas) if deltas else None,
        mean_node_delta_ci95=interval(deltas),
        geometric_node_ratio=exp(mean(map(log, ratios))) if ratios and len(ratios) == len(complete) else None,
        root_wins=sum(x > 0 for x in root_deltas), root_ties=sum(x == 0 for x in root_deltas),
        root_losses=sum(x < 0 for x in root_deltas), root_delta=sum(root_deltas),
        mean_root_delta=mean(root_deltas), mean_root_delta_ci95=interval(root_deltas),
        largest_node_difference=outlier)


def training_metrics(folder, final):
    path = folder / "training.jsonl"
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        return None
    steps = sum(record["optimizer_steps"] for record in records)
    return dict(records=len(records), fresh_trials=records[-1].get("fresh_trials", records[-1]["trials"]),
        batch_wall_seconds=sum(record["seconds"] for record in records),
        elapsed_seconds_from_file_times=max(0., final.stat().st_mtime - (folder / "protocol.json").stat().st_mtime),
        optimizer_steps=steps, gradient_clipped_steps=sum(record["grad_clipped"] for record in records),
        gradient_clipped_fraction=sum(record["grad_clipped"] for record in records) / max(1, steps),
        mean_gradient_norm=mean(record["grad_norm"] for record in records),
        skipped_groups=sum(record["skipped_groups"] for record in records),
        groups=sum(record["groups"] for record in records),
        note="Batch wall time excludes validation; file-time span includes initialization and validation. CPU contention affects both.")


def sampled_pair(report, reference):
    """Compare single-cluster trials only when their evaluation seed/count match."""
    settings, control = report.get("settings", {}), reference.get("settings", {})
    if any(settings.get(key) != control.get(key) for key in ("seed", "samples")):
        return None
    rows = {row["index"]: row for row in report["rows"]}
    baseline = {row["index"]: row for row in reference["rows"]}
    if set(rows) != set(baseline) or any(not row.get("sampled", {}).get("trials")
                                       for row in [*rows.values(), *baseline.values()]):
        return None
    usefulness, rewards = [], []
    for index, row in rows.items():
        left, right = row["sampled"], baseline[index]["sampled"]
        usefulness.append(left["useful"] / left["trials"] - right["useful"] / right["trials"])
        rewards.append(left["reward"] / left["trials"] - right["reward"] / right["trials"])
    return dict(seed=settings.get("seed"), samples_per_puzzle=settings.get("samples"),
        reference_useful=sum(row["sampled"]["useful"] for row in baseline.values()),
        method_useful=sum(row["sampled"]["useful"] for row in rows.values()),
        reference_trials=sum(row["sampled"]["trials"] for row in baseline.values()),
        method_trials=sum(row["sampled"]["trials"] for row in rows.values()),
        mean_puzzle_usefulness_delta=mean(usefulness), usefulness_delta_ci95=interval(usefulness),
        mean_puzzle_reward_delta=mean(rewards), reward_delta_ci95=interval(rewards))


def analyze(root=HERE, *, baseline_run=None, output=None, trials=2048, evaluations=()):
    root = Path(root).resolve()
    inputs = {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        inputs[str(path)] = sha256(raw).hexdigest()
        return json.loads(raw)

    data = read(root / "pilot_data.json")
    expected = set(data["validation_indices"])
    originals = data["pilot_preparation"]["original_indices"]
    baseline_dirs = [Path(baseline_run)] if baseline_run else sorted(root.glob("baselines32_*"))
    if len(baseline_dirs) != 1:
        raise ValueError("Choose one baseline directory with --baseline-run")
    baseline_dir = baseline_dirs[0]
    paths = {name: baseline_dir / f"{name}.json"
             for name in ("plain", "expert", "heuristic", "fixed_init", "untrained")}
    pending = []
    for folder in sorted(root.glob("*_seed*")):
        protocol = folder / "protocol.json"
        if not protocol.exists() or "data_digest" not in json.loads(protocol.read_text()):
            continue
        final = folder / f"validation_{trials:09d}.json"
        if final.exists():
            paths[folder.name] = final
        else:
            pending.append(folder.name)
    for path in sorted(root.glob("progressive30_*/heuristic_progressive_time30.json")):
        paths["progressive_heuristic_" + path.parent.name] = path
    overrides = {}
    for item in evaluations:
        label, separator, filename = item.partition("=")
        if not separator or not label:
            raise ValueError("Evaluation overrides must use LABEL=PATH")
        paths[label] = Path(filename)
        overrides[label] = filename
        if label in pending:
            pending.remove(label)
    reports, rows, summaries = {}, {}, {}
    for name, path in paths.items():
        report = read(path)
        indices = [row["index"] for row in report["rows"]]
        if set(indices) != expected or len(indices) != len(expected):
            raise ValueError(f"{name} has missing, duplicated or unexpected evaluation puzzles")
        reports[name] = report
        rows[name] = {row["index"]: compact(row, originals) for row in report["rows"]}
        complete = [row for row in rows[name].values() if row["dfs_complete"]]
        sampled = report.get("totals", {})
        summaries[name] = dict(instances=len(indices), root_solved=sum(row["root_solved"] for row in rows[name].values()),
            dfs_complete=len(complete), dfs_nodes=sum(row["dfs_nodes"] for row in complete)
                if len(complete) == len(indices) else None,
            median_dfs_nodes=median(row["dfs_nodes"] for row in complete) if complete else None,
            root_seconds=sum(row["root_seconds"] for row in rows[name].values()),
            residual_seconds=sum(row["residual_seconds"] for row in rows[name].values()),
            sampled_trials=sampled.get("sampled_trials"), sampled_useful=sampled.get("sampled_useful"),
            sampled_seconds=sampled.get("sampled_seconds"), sampled_mean_reward=sampled.get("sampled_mean_reward"),
            source=str(Path(path).resolve()), evaluation_settings=report.get("settings", {}),
            common_reevaluation_override=name in overrides)
        folder = root / name
        if folder.is_dir() and (folder / "protocol.json").exists():
            summaries[name]["training_protocol"] = read(folder / "protocol.json")
            final = folder / f"validation_{trials:09d}.json"
            if final.exists():
                summaries[name]["training"] = training_metrics(folder, final)
                history = folder / "training.jsonl"
                inputs[str(history.resolve())] = sha256(history.read_bytes()).hexdigest()
    comparisons = {name: paired(method_rows, rows["fixed_init"]) for name, method_rows in rows.items()
                   if name != "fixed_init"}
    sampled_comparisons = {}
    for name, report in reports.items():
        if name == "fixed_init":
            continue
        reference, reference_name = reports["fixed_init"], "fixed_init"
        comparison = sampled_pair(report, reference)
        initial = root / name / "validation_000000000.json"
        if comparison is None and initial.exists():
            reference, reference_name = read(initial), str(initial)
            comparison = sampled_pair(report, reference)
        if comparison is not None:
            sampled_comparisons[name] = dict(reference=reference_name, **comparison)
    caveats = [
        "Exploratory mechanism pilot: 32 training roots, 16 existing development-validation roots; no heldout100 test inputs.",
        "Final stopping points use the prescribed fresh-policy-trial budget; no best-checkpoint selection is performed here.",
        "Bootstrap intervals resample paired puzzles 10,000 times, seed20261123. They do not quantify training-seed or checkpoint-selection uncertainty.",
        "Single-cluster sampled changes are compared only at matching evaluation seeds/sample counts; training arms otherwise use their own initial evaluation.",
        "DFS intervals include only complete node pairs; censored outcomes are counted explicitly and never treated as zero.",
        "Original structural baseline has a 32-attempt bounded search. Progressive heuristic has a 30s root ceiling without that attempt cap.",
        "Sudoku expert uses 51 problem-specific families and larger cluster limits; it remains a reference rather than a matched generic arm.",
        "Initial/final validation and baseline timings were collected at different contention levels. Common final reevaluation should precede timing claims.",
        "Times exclude model loading. Root propagation, sampled single-cluster trials, residual DFS and training have separate recorded times.",
        "Largest paired node-difference contribution is diagnostic; all16 puzzles remain in the evaluation criterion.",
    ]
    summary = dict(status="partial" if pending else "available_final_evaluations_collected", pending=pending,
        bootstrap=dict(resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED, unit="paired puzzle", interval="percentile95"),
        methods=summaries, versus_fixed_initialization=comparisons,
        sampled_vs_matching_initialization=sampled_comparisons, caveats=caveats,
        evaluation_overrides=overrides, input_sha256=inputs, analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(output) if output else Path(mkdtemp(prefix="analysis_", dir=root))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty or new analysis output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "per_puzzle.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *next(iter(rows["fixed_init"].values())).keys(),
            "baseline_dfs_nodes", "node_delta", "baseline_root_solved", "root_delta"], lineterminator="\n")
        writer.writeheader()
        for name, method_rows in rows.items():
            for index, row in sorted(method_rows.items()):
                baseline = rows["fixed_init"][index]
                writer.writerow(dict(method=name, **row, baseline_dfs_nodes=baseline["dfs_nodes"],
                    node_delta=row["dfs_nodes"] - baseline["dfs_nodes"] if row["dfs_complete"] and baseline["dfs_complete"] else None,
                    baseline_root_solved=baseline["root_solved"], root_delta=row["root_solved"] - baseline["root_solved"]))
    with (output / "totals.csv").open("x", newline="") as stream:
        fields = ["method", "instances", "root_solved", "dfs_complete", "dfs_nodes", "median_dfs_nodes",
                  "root_seconds", "residual_seconds", "sampled_trials", "sampled_useful", "sampled_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, row in summaries.items():
            writer.writerow(dict(method=name, **{key: row.get(key) for key in fields[1:]}))
    text = ["Exploratory constructive-RL pilot; fixed final evaluations", "",
            "Method                         Root solves   DFS nodes  Median  Root time(s)"]
    for name, row in summaries.items():
        text.append(f"{name:30} {row['root_solved']:2}/{row['instances']:<3}       {str(row['dfs_nodes']):>6}   "
                    f"{str(row['median_dfs_nodes']):>5}   {row['root_seconds']:10.2f}")
    text.extend(["", "Paired differences versus the same frozen old RF initialization; negative DFS delta is better."])
    for name, result in comparisons.items():
        text.append(f"{name}: nodes improved/equal/worse {result['nodes_improved']}/{result['nodes_equal']}/{result['nodes_worsened']}; "
            f"total delta {result['total_node_delta']}; mean delta {result['mean_node_delta']}, "
            f"95% interval {result['mean_node_delta_ci95']}; root wins/losses {result['root_wins']}/{result['root_losses']}; "
            f"mean root delta {result['mean_root_delta']},95% interval {result['mean_root_delta_ci95']}.")
        largest = result["largest_node_difference"]
        if largest:
            text.append(f"  Largest absolute node contribution: pilot{largest['index']}/source{largest['original_index']}, "
                f"{largest['baseline_nodes']} -> {largest['method_nodes']}; delta{largest['delta']}; "
                f"remaining-puzzle delta{largest['total_delta_without_this_puzzle']} (diagnostic only).")
        if name in sampled_comparisons:
            sampled = sampled_comparisons[name]
            text.append(f"  Single-cluster useful trials at matching seed{sampled['seed']}: "
                f"{sampled['reference_useful']}/{sampled['reference_trials']} -> "
                f"{sampled['method_useful']}/{sampled['method_trials']}; reference {sampled['reference']}.")
    if pending:
        text.extend(["", "Pending final evaluations: " + ", ".join(pending)])
    text.extend(["", *caveats])
    (output / "REPORT.txt").write_text("\n".join(text) + "\n")
    print(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=2048)
    parser.add_argument("--evaluation", action="append", default=[], help="Override a result using LABEL=PATH")
    args = parser.parse_args()
    analyze(args.root, baseline_run=args.baseline_run, output=args.output, trials=args.trials,
            evaluations=args.evaluation)
