"""Measure excluded startup separately from the frozen root-propagation study.

Three fresh Python processes, one pinned CPU/thread; filesystem caches are not
cleared. No propagation, model predictions, or changes to the solver/results.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent


def trial():
    plan = json.loads((HERE / "protocol.json").read_text())
    os.sched_setaffinity(0, {plan["cpu"]})
    times = {}
    started = perf_counter()
    import numpy as np
    from experiments.cluster_selection import runtime
    from experiments.cluster_selection.supervised.collect import sudoku_domains
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from tnreason.engine.workload_to_numpy import NumpyCore
    times["shared_imports"] = perf_counter() - started
    assert "torch" not in sys.modules, "Shared solver imports unexpectedly load Torch"

    started = perf_counter()
    data = json.loads(Path(plan["data"]).read_text())
    times["read_dataset"] = perf_counter() - started
    raw = data["network"]
    started = perf_counter()
    network = ConstraintNetwork({core["key"]: NumpyCore(
        values=np.asarray(core["values"], dtype=bool).reshape(core["shape"]),
        colors=[raw["colors"][i] for i in core["colors"]])
        for core in raw["cores"]}, maintain_indices=True)
    domains = sudoku_domains(data["puzzles"][0])
    times["network_and_first_puzzle_domains"] = perf_counter() - started
    started = perf_counter()
    state = network.create_mac_state(domains)
    times["first_mac_state"] = perf_counter() - started
    assert len(state._cores) == len(raw["cores"])

    started = perf_counter()
    import torch
    from experiments.cluster_selection.reinforcement.model import Policy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["seed"])
    times["neural_imports_and_thread_setup"] = perf_counter() - started
    models = {}
    for name in ("teacher", "reinforce", "replay", "q", "initial"):
        started = perf_counter()
        models[name] = runtime.load(plan["checkpoints"][name]["path"])
        times[f"load_{name}"] = perf_counter() - started
        assert isinstance(models[name], Policy)
        assert models[name].config == plan["checkpoints"][name]["config"]
    for repeat in range(2):
        started = perf_counter()
        model = runtime.load(plan["checkpoints"]["teacher"]["path"])
        times[f"repeat_teacher_{repeat + 1}"] = perf_counter() - started
        del model
    times["single_teacher_neural_startup"] = (
        times["neural_imports_and_thread_setup"] + times["load_teacher"])
    times["mixed_neural_startup"] = times["neural_imports_and_thread_setup"] + sum(
        times[f"load_{name}"] for name in ("teacher", "reinforce", "replay", "q"))
    return dict(seconds=times, cpu=plan["cpu"], torch=torch.__version__,
                checkpoint_bytes={name: Path(item["path"]).stat().st_size
                                  for name, item in plan["checkpoints"].items()})


if __name__ == "__main__":
    if "--trial" in sys.argv:
        print(json.dumps(trial()))
    else:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", MPLCONFIGDIR="/tmp/cluster-mpl-cache")
        trials = []
        for _ in range(3):
            started = perf_counter()
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trial"],
                                    env=env, check=True, text=True, capture_output=True)
            row = json.loads(result.stdout)
            row["whole_process_seconds"] = perf_counter() - started
            trials.append(row)
        summary = {key: dict(median=statistics.median(values), min=min(values), max=max(values))
                   for key in trials[0]["seconds"]
                   for values in [[row["seconds"][key] for row in trials]]}
        output = dict(note="Fresh processes, existing filesystem caches; no solving. "
                      "Shared setup and extra neural startup are measured separately. "
                      "No provenance-hash verification is timed.",
                      protocol_sha256=hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      summary=summary, trials=trials)
        with Path(__file__).with_suffix(".json").open("x") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")
        print(json.dumps(summary, indent=2))
