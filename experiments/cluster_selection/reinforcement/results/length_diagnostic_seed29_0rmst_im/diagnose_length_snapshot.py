"""Paired, retrospective core-prefix trials on existing development validation.

Same-output prefixes isolate core count. Anchor-only prefixes have their own
full-length comparator. This measures one application, not an early-stop solver.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import log
from pathlib import Path
from random import Random
from tempfile import mkdtemp
from time import perf_counter

import torch

from ..candidate_evaluation_mac import evaluate_candidate
from ..candidate_evaluation_tnreason import SummaryClusterCandidate
from .experiment import instances, solve
from .model import load
from .train import _state


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def summarize(cases):
    groups = {}
    for variant in ("same_outputs", "anchor_only"):
        selected = [c for c in cases if c["candidate"] is not None]
        useful, earlier, equivalent, worse_dfs, equal_dfs, better_dfs = [], [], [], 0, 0, 0
        censored_pairs = 0
        for case in selected:
            rows = [r for r in case["trials"] if r["variant"] == variant and r["status"] == "measured"]
            full = next((r for r in rows if r["cores"] == case["full_cores"]), None)
            if full is None or not full["removed_all"]:
                continue
            useful.append(case)
            short = [r for r in rows if r["cores"] < full["cores"] and r["finish_legal"] and r["removed_all"]]
            if short:
                first = min(short, key=lambda r: r["cores"])
                earlier.append(dict(puzzle=case["puzzle"], draw=case["draw"], cores=first["cores"],
                    full_cores=full["cores"], retained_information=first["information"] / full["information"],
                    evaluation_s=first["evaluation_s"], full_evaluation_s=full["evaluation_s"]))
                if first["dfs"]["status"] == full["dfs"]["status"] == "solved":
                    difference = first["dfs"]["nodes"] - full["dfs"]["nodes"]
                    worse_dfs += difference > 0
                    equal_dfs += difference == 0
                    better_dfs += difference < 0
                else:
                    censored_pairs += 1
            matching = [r for r in short if r["changed_domains"] == full["changed_domains"]]
            if matching:
                equivalent.append(dict(puzzle=case["puzzle"], draw=case["draw"],
                    cores=min(r["cores"] for r in matching), full_cores=full["cores"]))
        all_rows = [r for c in selected for r in c["trials"] if r["variant"] == variant]
        groups[variant] = dict(constructions=len(selected), full_useful=len(useful),
            shorter_legal_useful=len(earlier), shorter_legal_identical_domains=len(equivalent),
            shortest_tested_useful_vs_full_dfs=dict(better=better_dfs, equal=equal_dfs, worse=worse_dfs,
                comparable=better_dfs + equal_dfs + worse_dfs, censored=censored_pairs),
            rejected=sum(r["status"] == "rejected" for r in all_rows),
            full_rejected=sum(r["status"] == "rejected" and r["cores"] == c["full_cores"]
                              for c in selected for r in c["trials"] if r["variant"] == variant),
            earlier=earlier, equivalent=equivalent)
    return dict(cases=len(cases), skipped=sum(c["candidate"] is None for c in cases), groups=groups)


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = json.loads(args.data.read_text())
    model = load(args.checkpoint).cpu().eval()
    assert data["network"] == model.network and model.config.get("target_first")
    validation = list(data["validation_indices"])
    assert not set(validation) & set(data["train_indices"])
    Random(args.seed).shuffle(validation)
    subset = {**data}
    field = "initial_domains" if "initial_domains" in data else "puzzles"
    subset[field] = [data[field][i] for i in validation]
    initial, _ = instances(subset)
    base = _state({**data, "samples": data.get("samples") or [{}]})
    network = base.constraint_network
    selected = []
    for puzzle, domains in zip(validation, initial):
        state = network.create_mac_state(domains)
        assert state.propagate(), "Contradictory validation input"
        if not state.is_solved():
            selected.append((puzzle, state))
        if len(selected) == args.states:
            break
    assert len(selected) == args.states, "Insufficient GAC-stalled validation states"
    out = Path(mkdtemp(prefix=f"length_diagnostic_seed{args.seed}_", dir=Path(__file__).parent / "results"))
    sources = [Path(__file__).resolve(), args.data.resolve(), args.checkpoint.resolve(),
               Path(__file__).with_name("model.py"), Path(__file__).with_name("experiment.py"),
               Path(__file__).parents[1] / "candidate_evaluation_mac.py",
               Path(__file__).parents[3] / "experiments/constraint_networks/mac.py"]
    hashes = {str(p): digest(p) for p in sources}
    with (out / "diagnose_length_snapshot.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    protocol = dict(seed=args.seed, validation_indices=[p for p, _ in selected],
        samples_per_state=args.samples, greedy_first=True, inference_mode="eval; beta=0; no exploration mixture",
        cuts="every prefix containing all outputs" if args.all_prefixes else
             [1, 2, 4, 8, 16, 32, 48, "full", "first prefix containing all outputs"],
        config=model.config, source_sha256=hashes,
        interpretation="Existing validation only; no training or heldout100 input. Fixed-output, independent "
            "one-cluster trials with GAC and rollback. Anchor-only has a separate full comparator. "
            "Prefix tests are retrospective; construction timestamps exclude stop/output decisions. "
            "No learned summaries during residual plain-MAC DFS. No claim of early-stop solver speed.")
    write(out / "protocol.json", protocol)
    print(f"Results: {out}", flush=True)
    cases = []
    for puzzle, state in selected:
        before = state.snapshot_domains()
        counters, core_keys = asdict(state.statistics), tuple(state._cores)
        targets = set(model.reward_colors)
        auxiliary_weight = min(1., len(targets) / max(1, len(before) - len(targets)))
        weights = {c: 1. if c in targets else auxiliary_weight for c in before}
        remaining = sum(weights[c] * log(len(v)) for c, v in before.items())
        dfs_cache = {}

        def dfs(domains):
            signature = tuple(domains[c] for c in model.network["colors"])
            if signature not in dfs_cache:
                result = solve(network, domains, "plain", model, nodes=10000, seconds=30.)
                assert result["status"] not in {"invalid", "unsat"}
                dfs_cache[signature] = {k: result[k] for k in ("status", "nodes", "solve_s")}
            return dfs_cache[signature]

        write(out / f"state_{puzzle}.json", dict(puzzle=puzzle, domains=before, baseline_dfs=dfs(before)))
        for draw in range(args.samples):
            seed = args.seed + 1000 * puzzle + draw
            torch.manual_seed(seed)
            ticks, original = [], model.distribution
            began = perf_counter()

            def timed(*a, **kw):
                result = original(*a, **kw)
                ticks.append(perf_counter() - began)
                return result

            model.distribution = timed
            try:
                with torch.no_grad():
                    candidate, _, _, trace = model.rollout(state.domains, beta=0., greedy=draw == 0)
            finally:
                del model.__dict__["distribution"]
            assert len(ticks) == len(trace)
            construction_s = perf_counter() - began
            case = dict(puzzle=puzzle, draw=draw, seed=seed, greedy=draw == 0, trace=trace,
                candidate=None if candidate is None else asdict(candidate), construction_s=construction_s, trials=[])
            if candidate is not None:
                ordered = [idx for kind, idx in trace if kind == "core"]
                anchor = next(idx for kind, idx in trace if kind == "open")
                case["full_cores"] = len(ordered)
                sizes = [len(before[c]) for c in model.network["colors"]]
                for variant, outputs in (("same_outputs", candidate.open_colors),
                                         ("anchor_only", (model.network["colors"][anchor],))):
                    covered, first_covered = set(), None
                    for count, idx in enumerate(ordered, 1):
                        covered.update(model.network["colors"][c] for c in model.scope_sets[idx])
                        if set(outputs) <= covered and first_covered is None:
                            first_covered = count
                    assert first_covered is not None
                    cuts = sorted({min(k, len(ordered)) for k in (1, 2, 4, 8, 16, 32, 48)}
                                  | {len(ordered), first_covered})
                    if args.all_prefixes:
                        cuts = range(first_covered, len(ordered) + 1)
                    measured = []
                    for count in cuts:
                        if count < first_covered:
                            continue
                        keys = tuple(model.keys[i] for i in sorted(ordered[:count]))
                        prefix = SummaryClusterCandidate(f"length_{puzzle}_{draw}_{variant}_{count}", keys, tuple(outputs))
                        row = dict(variant=variant, cores=count, outputs=len(outputs),
                            finish_legal=("finish", -1) in model.actions(ordered[:count], [anchor], False, sizes),
                            construction_to_last_core_s=ticks[next(i for i, action in enumerate(trace)
                                if action == ("core", ordered[count - 1]))])
                        try:
                            result = evaluate_candidate(state, prefix, reward_colors=model.reward_colors,
                                compact_domains=model.config.get("compact_domains", False),
                                **{k: model.config[k] for k in ("max_work", "max_size")})
                        except ValueError as error:
                            if str(error) not in {"Candidate exceeds the contraction work or tensor-size budget",
                                                  "NumPy contraction supports at most 52 distinct colors"}:
                                raise
                            row.update(status="rejected", reason=str(error))
                        else:
                            assert not result.contradiction
                            after = result.domains
                            assert all(set(after[c]) <= set(before[c]) for c in before)
                            information = sum(weights[c] * log(len(before[c]) / len(after[c])) for c in before)
                            row.update(status="measured", work=result.estimate.work, max_size=result.estimate.max_size,
                                evaluation_s=result.cost, information=information, reward=information / remaining,
                                removed_targets=result.removed_domain_values,
                                removed_all=sum(len(before[c]) - len(after[c]) for c in before),
                                changed_domains={c: v for c, v in after.items() if v != before[c]},
                                solved=all(len(v) == 1 for v in after.values()), dfs=dfs(after))
                            measured.append((count, after))
                        assert state.snapshot_domains() == before and asdict(state.statistics) == counters
                        assert tuple(state._cores) == core_keys and not state._added_core_keys
                        assert state.decision_level == 0 and not state._queue and not state._queued
                        case["trials"].append(row)
                    for (_, shorter), (_, longer) in zip(measured, measured[1:]):
                        assert all(set(longer[c]) <= set(shorter[c]) for c in before), "Nonmonotone fixed-output prefix"
            write(out / f"case_{puzzle}_{draw}.json", case)
            cases.append(case)
            print(f"puzzle {puzzle}, draw {draw}: cores {case.get('full_cores', 0)}, "
                  f"useful trials {sum(r.get('removed_all', 0) > 0 for r in case['trials'])}", flush=True)
    assert all(digest(Path(p)) == value for p, value in hashes.items()), "An input changed during diagnosis"
    result = summarize(cases)
    write(out / "summary.json", result)
    print(json.dumps({v: {k: value for k, value in row.items() if k not in {"earlier", "equivalent"}}
                      for v, row in result["groups"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--all-prefixes", action="store_true", help="Check every valid core-prefix length")
    args = parser.parse_args()
    if min(args.states, args.samples) < 1:
        parser.error("states and samples must be positive")
    run(args)
