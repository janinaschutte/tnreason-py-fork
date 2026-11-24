"""Serial, puzzle-interleaved time-only comparison; no training or test100 access."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from statistics import median
import sys
from tempfile import mkdtemp

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    import torch
    from experiments.cluster_selection.reinforcement import refine, refine_eval
    from experiments.cluster_selection.reinforcement.actor import load
    from experiments.cluster_selection.reinforcement.experiment import instances
    from experiments.cluster_selection import runtime
    from analyze import compact, paired

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_path, refs_path = BASE / "pilot_data.json", BASE / "baselines32_wgk4zatg/plain.json"
    expert_path = BASE / "baselines32_wgk4zatg/expert.json"
    prior_protocol = read(BASE / "prior_gpu_replay_arli8b26/protocol.json")
    random_protocol = read(BASE / "time30_untrained_1j78ql6z/protocol.json")
    rms_protocol = read(BASE / "time30_global_rms_seed0_ux5j8phj/protocol.json")
    checkpoints = dict(prior_gpu=Path(prior_protocol["checkpoint"]),
        untrained=BASE / "common_v2_2wlie6na/untrained.pt", global_rms=BASE / "global_rms_seed0/final.pt")
    expected = dict(prior_gpu=prior_protocol["hashes"][str(checkpoints["prior_gpu"])],
        untrained=random_protocol["hashes"][str(checkpoints["untrained"])],
        global_rms=rms_protocol["hashes"]["checkpoint"])
    assert all(digest(path) == expected[name] for name, path in checkpoints.items())
    data = read(data_path)
    references = {row["index"]: row["residual"]["assignment"] for row in read(refs_path)["rows"]}
    initial, _ = instances(data)
    models = {name: load(path).cpu().eval() for name, path in checkpoints.items()}
    for model in models.values():
        assert model.network == data["network"] and model.reward_colors == data["reward_colors"]
        assert model.beta == 0. and all(model.config[key] == models["untrained"].config[key]
            for key in ("max_cores", "max_open_colors", "max_work", "max_size", "decoder", "compact_domains"))
    rng = random.Random(20261123)
    indices, order = list(data["validation_indices"]), list(models)
    rng.shuffle(indices)
    rng.shuffle(order)
    schedule = [(index, name) for position, index in enumerate(indices)
                for name in order[position % 3:] + order[:position % 3]]
    output = Path(mkdtemp(prefix="serial_time30_", dir=BASE))
    print(output, flush=True)
    paths = dict(data=data_path, references=refs_path, expert=expert_path, driver=Path(__file__),
        analyzer=BASE / "analyze.py", **checkpoints)
    hashes, sources = {name: digest(path) for name, path in paths.items()}, refine.source_hashes()
    settings = dict(attempts=0, samples=0, seconds=args.seconds, nodes=10000, seed=100000)
    write(output / "protocol.json", dict(paths={name: str(path) for name, path in paths.items()},
        hashes=hashes, sources=sources, settings=settings, cpu=sorted(os.sched_getaffinity(0)),
        model_configuration={name: dict(config=model.config, actor_config=model.actor_config, beta=model.beta)
                             for name, model in models.items()},
        schedule=schedule, schedule_seed=20261123,
        scope="Same16 development roots, no heldout100. Serial single CPU, rotated method order per shuffled puzzle.",
        timing="Root wall ceilings exclude loading/setup; separate plain residual MAC. No other coordinated heavy jobs."))
    rows = {name: [] for name in models}
    entries = givens = 0
    colors = data["network"]["colors"]
    for position, (index, name) in enumerate(schedule):
        shapes, original_contract = [], runtime.contract_candidate

        def observed_contract(candidate, prepared):
            shapes.append(dict(cores=len(candidate.core_keys), outputs=len(candidate.open_colors),
                               work=int(prepared[0].work)))
            return original_contract(candidate, prepared)

        runtime.contract_candidate = observed_contract
        try:
            result = refine_eval.evaluate(models[name], data, [index], **settings)
        finally:
            runtime.contract_candidate = original_contract
        row, = result["rows"]
        assert len(shapes) == row["root"]["work"]["contractions"]
        row["construction_sizes"] = shapes
        write(output / f"row_{position:02d}_{name}.json", row)
        assert row["index"] == index and row["residual"]["status"] == "solved"
        assignment = row["residual"]["assignment"]
        assert set(assignment) == set(colors)
        assert all(value in row["root"]["domains"][color] for color, value in references[index].items())
        assert all(assignment[color] in values for color, values in row["root"]["domains"].items())
        for core in data["network"]["cores"]:
            offset = 0
            for color, size in zip(core["colors"], core["shape"]):
                value = assignment[colors[color]]
                assert 0 <= value < size
                offset = offset * size + value
            assert core["values"][offset]
            entries += 1
        for color, allowed in initial[index].items():
            assert assignment[color] in allowed
            givens += len(allowed) == 1
        rows[name].append(row)
        print(f"{position+1}/48 index{index} {name}: root {row['root']['status']}, DFS{row['residual']['nodes']}", flush=True)
    reports, normalized = {}, {}
    originals = data["pilot_preparation"]["original_indices"]
    for name, method_rows in rows.items():
        assert len(method_rows) == len({row["index"] for row in method_rows}) == 16
        method_rows.sort(key=lambda row: row["index"])
        totals = dict(instances=16, root_solved=sum(row["root"]["solved"] for row in method_rows),
            dfs_nodes=sum(row["residual"]["nodes"] for row in method_rows), dfs_complete=16,
            root_seconds=sum(row["root"]["solve_s"] for row in method_rows),
            residual_seconds=sum(row["residual"]["solve_s"] for row in method_rows))
        work = dict(root_statuses=dict(Counter(row["root"]["status"] for row in method_rows)),
            **{key: sum(row["root"]["cluster_statistics"].get(key, 0) for row in method_rows)
               for key in ("candidates", "selected", "useful", "estimated_work")})
        for key in ("cores", "outputs"):
            values = [shape[key] for row in method_rows for shape in row["construction_sizes"]]
            work[key] = dict(mean=sum(values)/len(values), median=median(values), min=min(values), max=max(values))
        reports[name] = dict(totals=totals, work=work)
        normalized[name] = {row["index"]: compact(row, originals) for row in method_rows}
        write(output / f"{name}.json", dict(rows=method_rows, totals=totals, settings=settings))
    assert hashes == {name: digest(path) for name, path in paths.items()} and sources == refine.source_hashes()
    summary = dict(methods=reports, paired_vs_untrained={name: paired(item, normalized["untrained"])
        for name, item in normalized.items() if name != "untrained"}, expert_reference=read(expert_path)["totals"],
        audit=dict(sources_inputs_checkpoints_unchanged=True, valid_assignments=48,
            original_tensor_entries_checked=entries, original_givens_checked=givens,
            reference_assignments_retained=48),
        caveat="One development set, one checkpoint/seed each. Expert reference reused with larger problem-specific limits.")
    write(output / "summary.json", summary)
    table = ["Serial single-core time30, root propagation then separate plain residual MAC",
             "Method          Root solves  Residual DFS  Root seconds  Residual seconds"]
    for name, report in {**reports, "expert_reused": dict(totals=summary["expert_reference"])}.items():
        row = report["totals"]
        table.append(f"{name:15} {row['root_solved']:2}/16 {row['dfs_nodes']:14} "
                     f"{row['root_seconds']:13.2f} {row['residual_seconds']:17.2f}")
    table.extend(["", summary["caveat"], "Shared-host activity outside this experiment remains uncontrolled."])
    (output / "TABLES.txt").write_text("\n".join(table) + "\n")
    print(output, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
