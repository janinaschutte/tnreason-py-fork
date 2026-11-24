"""Profile one measured four-construction training group; no checkpoints change."""
import argparse
import cProfile
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import pstats
from random import Random
from tempfile import mkdtemp
from time import perf_counter

import torch

from experiments.cluster_selection.reinforcement.learn import training_states
from experiments.cluster_selection.reinforcement.refine import collect, initialize, source_hashes
from experiments.cluster_selection.reinforcement.updates import advantages, update


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"


def main(index=24, seed=0, first_active=False):
    data_file = HERE / "pilot_data.json"
    data = json.loads(data_file.read_text())
    torch.set_num_threads(1)
    base, state, pools, _ = training_states(data)
    if index not in pools:
        raise ValueError("Choose an unresolved training puzzle")
    model = initialize(base, initial=PARENT, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    order, probes = list(pools), []
    Random(seed).shuffle(order)
    candidates = [order.pop() for _ in range(min(4, len(order)))] if first_active else [index]
    for index in candidates:
        sample = next(item for item in pools[index] if item.get("root"))
        profiler = cProfile.Profile()
        with profiler:
            began = perf_counter()
            group, outcomes, distinct = collect(model, state, sample, rollouts=4)
            measured = perf_counter()
            for item, value in zip(group["rollouts"], advantages(
                    [item["reward"] for item in group["rollouts"]], "rms", .05)):
                item["advantage"] = value
            metrics = update(model, [group], optimizer)
            finished = perf_counter()
        probes.append(dict(index=index, useful=sum(item["useful"] for item in outcomes),
                           optimizer_steps=metrics["optimizer_steps"]))
        if metrics["optimizer_steps"]:
            break
    output = Path(mkdtemp(prefix="batch_profile_", dir=HERE))
    profiler.dump_stats(output / "profile.pstats")
    report = io.StringIO()
    stats = pstats.Stats(profiler, stream=report)
    stats.sort_stats("cumulative").print_stats(35)
    stats.sort_stats("tottime").print_stats(35)
    (output / "profile.txt").write_text(report.getvalue())
    top = sorted(stats.stats.items(), key=lambda pair: pair[1][3], reverse=True)[:35]
    summary = dict(index=index, original_index=data["pilot_preparation"]["original_indices"][index],
        seed=seed, affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        torch_version=str(torch.__version__), cuda_available=torch.cuda.is_available(),
        collected_trials=4, probes=probes, first_active_search=first_active,
        useful=sum(item["useful"] for item in outcomes), unique=distinct,
        rewards=[item["reward"] for item in group["rollouts"]], update=metrics,
        profiled_collection_seconds=measured - began, profiled_update_seconds=finished - measured,
        top_cumulative=[dict(file=key[0], line=key[1], function=key[2], primitive_calls=value[0],
            calls=value[1], self_seconds=value[2], cumulative_seconds=value[3]) for key, value in top],
        data_sha256=sha256(data_file.read_bytes()).hexdigest(),
        parent_sha256=sha256(PARENT.read_bytes()).hexdigest(), source_hashes=source_hashes(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=["Four actual measured constructions on a fixed existing training root; zero-advantage groups skip the requested update.",
            "Profile overhead and CPU contention affect these durations; this is not a solver timing comparison.",
            "Network setup and checkpoint loading are excluded. This is the first Adam step, including moment allocation.",
            "Only CPU execution is profiled; GPU speed or utilization is not measured.",
            "The updated weights exist only in this process and are discarded."])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(output)
    print(json.dumps({key: summary[key] for key in ("useful", "rewards", "update",
        "profiled_collection_seconds", "profiled_update_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-active", action="store_true",
                        help="Try the first four fixed pilot-order roots until one group has a real gradient")
    args = parser.parse_args()
    main(args.index, args.seed, args.first_active)
