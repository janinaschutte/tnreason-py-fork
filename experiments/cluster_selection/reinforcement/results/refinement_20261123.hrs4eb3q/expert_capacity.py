"""Inspect saved development expert proofs; no contraction or training is run."""
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from math import prod
from pathlib import Path
from statistics import median
from tempfile import mkdtemp
from types import SimpleNamespace

import torch

from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.contraction_cost import estimate_candidate_cost
from experiments.cluster_selection.reinforcement.model import load


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"


def distribution(values):
    return dict(min=min(values), median=median(values), max=max(values),
                counts=dict(sorted(Counter(values).items())))


def representation(model, candidate, domains):
    sizes = [len(domains[color]) for color in model.network["colors"]]
    options, failures = [], Counter()
    for anchor in candidate.open_colors:
        try:
            trace = model.candidate_trace(candidate, domains, target_color=anchor)
        except ValueError as error:
            failures[str(error)] += 1
            continue
        keys, opened, phase, blocks = [], [], False, []
        for action in trace:
            actions = model.actions(keys, opened, phase, sizes)
            following = [actions[idx] for idx in model._continuations(actions, keys, opened, phase)]
            if action not in following:
                blocks.append(dict(action=action, phase="outputs" if phase else "cores",
                                   legal_growth_choices=len(following)))
            kind, index = action
            if kind == "core":
                keys.append(index)
            elif kind == "open":
                opened.append(index)
            elif kind == "finish":
                phase = True
        options.append(dict(anchor=anchor, growth_blocks=blocks))
    return dict(teacher_trace_legal=bool(options), growth_trace_legal=any(not x["growth_blocks"] for x in options),
        core_stop_blocked=bool(options) and all(any(b["phase"] == "cores" for b in x["growth_blocks"]) for x in options),
        output_stop_blocked=bool(options) and all(any(b["phase"] == "outputs" for b in x["growth_blocks"]) for x in options),
        anchors=options, teacher_failures=dict(failures))


def reduce_tautologies(model, candidate, domains):
    """Remove only factors identically true on the current Cartesian domains."""
    raw = {core["key"]: core for core in model.network["cores"]}
    kept, dropped, inputs = [], [], {}
    for key in candidate.core_keys:
        core = raw[key]
        colors = [model.network["colors"][idx] for idx in core["colors"]]
        strides = [prod(core["shape"][idx + 1:]) for idx in range(len(colors))]
        tautology = all(core["values"][sum(value * stride for value, stride in zip(values, strides))]
                       for values in product(*(domains[color] for color in colors)))
        (dropped if tautology else kept).append(key)
        if not tautology:
            inputs[key] = SimpleNamespace(colors=colors, shape=tuple(len(domains[color]) for color in colors))
    scope = set().union(*(set(core.colors) for core in inputs.values()))
    outputs = tuple(color for color in candidate.open_colors if color in scope)
    reduced = SummaryClusterCandidate("reduced_saved_expert", tuple(kept), outputs)
    estimate = estimate_candidate_cost(inputs, reduced)
    expanded = prod(model.network["dimensions"][model.color_idx[color]] for color in outputs)
    affordable = (len(kept) <= model.config["max_cores"] and len(outputs) <= model.config["max_open_colors"]
        and estimate.work <= model.config["max_work"] and max(estimate.max_size, expanded) <= model.config["max_size"])
    return dict(cores=len(kept), outputs=len(outputs), dropped_cores=dropped,
        dropped_outputs=sorted(set(candidate.open_colors) - set(outputs)), scope_size=len(scope),
        work=estimate.work, max_size=max(estimate.max_size, expanded), within_learner_budgets=affordable,
        representation=representation(model, reduced, domains) if affordable else None)


def main():
    torch.set_num_threads(1)
    data_file = HERE / "pilot_data.json"
    data, model = json.loads(data_file.read_text()), load(PARENT)
    primary = set(data["reward_colors"])
    dimensions = dict(zip(model.network["colors"], model.network["dimensions"]))
    indices = data["validation_indices"]
    sources, rows = {}, []
    paths = sorted(HERE.glob("baselines32_*/expert_raw/rows/expert_*.json"))
    if len(paths) != len(indices):
        raise ValueError("Expected all sixteen saved expert trajectories")
    for path in paths:
        saved = json.loads(path.read_text())
        sources[str(path)] = sha256(path.read_bytes()).hexdigest()
        for number, proof in enumerate(saved["proofs"]):
            domains = saved["snapshots"][proof["before"]]["domains"]
            cores, outputs = proof["core_keys"], proof["open_colors"]
            scope = set().union(*(model.scope_sets[model.key_idx[key]] for key in cores))
            within = (len(cores) <= model.config["max_cores"] and len(outputs) <= model.config["max_open_colors"]
                and proof["estimated_work"] <= model.config["max_work"] and proof["max_size"] <= model.config["max_size"]
                and prod(dimensions[color] for color in outputs) <= model.config["max_size"])
            candidate = SummaryClusterCandidate("saved_expert", tuple(cores), tuple(outputs))
            role = ("primary_only" if all(color in primary for color in outputs) else
                    "auxiliary_only" if all(color not in primary for color in outputs) else "mixed")
            row = dict(position=saved["position"], pilot_index=indices[saved["position"]], proof=number,
                family=proof["family"], cores=len(cores), outputs=len(outputs), output_role=role,
                output_primary=sum(color in primary for color in outputs),
                output_auxiliary_binary=sum(color not in primary and dimensions[color] == 2 for color in outputs),
                output_auxiliary_other=sum(color not in primary and dimensions[color] != 2 for color in outputs),
                work=proof["estimated_work"], max_size=proof["max_size"], scope_size=len(scope),
                expanded_output_size=prod(dimensions[color] for color in outputs),
                removed=proof["removed"], within_learner_budgets=within,
                inactive_cores=sum(sum(len(domains[model.network["colors"][idx]]) > 1
                    for idx in model.scope_sets[model.key_idx[key]]) < model.config.get("min_active", 1) for key in cores))
            row["representation"] = representation(model, candidate, domains) if within else None
            row["tautology_reduced"] = reduce_tautologies(model, candidate, domains)
            rows.append(row)
    legal = [row for row in rows if row["representation"] and row["representation"]["teacher_trace_legal"]]
    reduced = [row["tautology_reduced"] for row in rows]
    reduced_legal = [row for row in reduced if row["representation"] and row["representation"]["teacher_trace_legal"]]
    summary = dict(trajectories=len(paths), proofs=len(rows), limits={key: model.config[key] for key in
        ("max_cores", "max_open_colors", "max_work", "max_size")},
        families=dict(Counter(row["family"] for row in rows)),
        cores=distribution([row["cores"] for row in rows]), outputs=distribution([row["outputs"] for row in rows]),
        work=distribution([row["work"] for row in rows]), max_size=distribution([row["max_size"] for row in rows]),
        output_roles=dict(Counter(row["output_role"] for row in rows)),
        output_axis_counts={key: sum(row[key] for row in rows) for key in
                            ("output_primary", "output_auxiliary_binary", "output_auxiliary_other")},
        within_learner_budgets=sum(row["within_learner_budgets"] for row in rows),
        legal_with_free_stopping=len(legal), legal_with_actual_growth=sum(row["representation"]["growth_trace_legal"] for row in legal),
        legal_but_core_stopping_blocked=sum(row["representation"]["core_stop_blocked"] for row in legal),
        legal_but_output_stopping_blocked=sum(row["representation"]["output_stop_blocked"] for row in legal),
        affordable_trace_failures=dict(Counter(error for row in rows if row["representation"]
            for error in row["representation"]["teacher_failures"] if not row["representation"]["teacher_trace_legal"])),
        after_safe_tautology_reduction=dict(cores=distribution([row["cores"] for row in reduced]),
            outputs=distribution([row["outputs"] for row in reduced]),
            within_learner_budgets=sum(row["within_learner_budgets"] for row in reduced),
            legal_with_free_stopping=len(reduced_legal),
            legal_with_actual_growth=sum(row["representation"]["growth_trace_legal"] for row in reduced_legal),
            legal_but_core_stopping_blocked=sum(row["representation"]["core_stop_blocked"] for row in reduced_legal),
            legal_but_output_stopping_blocked=sum(row["representation"]["output_stop_blocked"] for row in reduced_legal),
            teacher_failures=dict(Counter(error for row in reduced if row["representation"]
                for error in row["representation"]["teacher_failures"] if not row["representation"]["teacher_trace_legal"]))),
        parent_sha256=sha256(PARENT.read_bytes()).hexdigest(), data_sha256=sha256(data_file.read_bytes()).hexdigest(),
        source_sha256=sources, script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows,
        caveats=["Existing development validation only; expert proof domains are sequential expert states, not matched learner states.",
            "Work and tensor entries are the saved measured estimates; expanded output size is additionally checked against the actor mask.",
            "All possible retained-output anchors are checked using the exact current construction masks. No neural forward, contraction or training is performed.",
            "Reduction checks raw tensor entries, removes only domain-tautological factors and vacuous output axes, then replans without contraction. This preserves the conditional relation on recorded domains.",
            "These are exact saved core/output sets. Larger clusters containing those cores may reproduce their deductions; this audit does not test that."])
    output = Path(mkdtemp(prefix="expert_capacity_", dir=HERE))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(output)
    print(json.dumps({key: value for key, value in summary.items() if key not in
        {"source_sha256", "rows", "work", "max_size"}}, indent=2))


if __name__ == "__main__":
    main()
