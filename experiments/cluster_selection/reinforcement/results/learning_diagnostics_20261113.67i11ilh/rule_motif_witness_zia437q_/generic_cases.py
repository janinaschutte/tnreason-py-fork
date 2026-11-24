"""Reproduce sparse/dense generic Hall witnesses without importing Sudoku code."""
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate

OUT = Path(__file__).resolve().parent


def forbidden(*args, **kwargs):
    raise AssertionError("Search and assignments are forbidden")


def main():
    fixture = json.loads((OUT / "generic_variant.json").read_text())
    source = (OUT / "candidate_generation.before.py").read_text()
    module = ModuleType("experiments.cluster_selection.generic_hall_before")
    module.__package__ = "experiments.cluster_selection"
    exec(compile(source, str(OUT / "candidate_generation.before.py"), "exec"), module.__dict__)
    results = {}
    for name, initial in [("sparse", fixture["initial"]), ("dense", {})]:
        cores = {c["key"]: NumpyCore(values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
                                   colors=c["colors"]) for c in fixture["network"]}
        network = ConstraintNetwork(cores, maintain_indices=True)
        network.solve_mac = forbidden
        state = network.create_mac_state(initial)
        state.assign = forbidden
        assert state.propagate()
        before = state.snapshot_domains()
        candidate = SummaryClusterCandidate("generic_hall_rectangle", tuple(cores), fixture["target_outputs"])
        projection = evaluate_candidate(state, candidate, compact_domains=True, cost_weight=0)
        assert projection.removed_domain_values == 3
        result = dict(removed=projection.removed_domain_values, work=projection.estimate.work,
                      max_size=projection.estimate.max_size, old_cycle_generator={})
        for length in (8, 12):
            generator = module.ClusterCandidateGenerator(network, max_cores=64, max_open_colors=6,
                max_cycle_length=length, max_candidates=512, max_steps=131072)
            candidates = generator.generate(state.domains, cyclic_only=True, compact_domains=True)
            useful = sum(evaluate_candidate(state, c, compact_domains=True, cost_weight=0)
                         .removed_domain_values > 0 for c in candidates)
            assert useful == 0 if length == 8 else useful > 0
            result["old_cycle_generator"][length] = dict(candidates=len(candidates), useful=useful)
        assert state.snapshot_domains() == before
        results[name] = result
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
