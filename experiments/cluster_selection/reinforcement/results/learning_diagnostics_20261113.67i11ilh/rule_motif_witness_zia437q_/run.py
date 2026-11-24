"""A checked Sudoku naked-triple witness for the old generic cycle proposals."""
from hashlib import sha256
import json
from pathlib import Path
import signal
from time import perf_counter
from types import ModuleType

from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.constraint_networks.sudoku_tests.standard_constraints import get_sudoku_constraint_network
from demonstrations.sudoku.inferenceClusters import RuleContext
from demonstrations.sudoku.inferenceClusters.subsets_fish import detect_subsets
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate

OUT = Path(__file__).resolve().parent


def main():
    source = (OUT / "candidate_generation.before.py").read_text()
    module = ModuleType("experiments.cluster_selection.motif_witness_before")
    module.__package__ = "experiments.cluster_selection"
    exec(compile(source, str(OUT / "candidate_generation.before.py"), "exec"), module.__dict__)
    network = ConstraintNetwork(get_sudoku_constraint_network(num=3))
    naming = RuleContext(network, num=3)
    positions = tuple(naming.pos(0, c) for c in (0, 3, 6))
    units = tuple(naming.unit("row", 0, d) for d in range(3))
    initial = {p: tuple(sorted((i, (i + 1) % 3))) for i, p in enumerate(positions)}
    state = network.create_mac_state(initial)
    assert state.propagate() and all(state.domains[p] == initial[p] for p in positions)
    before = state.snapshot_domains()
    context = RuleContext(network, state.domains, num=3)
    expert = next(c for c in detect_subsets(context, sizes=(3,)).clusters
                  if set(c.open_colors) == set(positions + units))
    candidate = SummaryClusterCandidate("naked_triple_target_units", expert.core_keys, units)
    trial = evaluate_candidate(state, candidate, compact_domains=True, cost_weight=0)
    assert trial.removed_domain_values > 0 and state.snapshot_domains() == before
    print("EXPERT", len(candidate.core_keys), "work", trial.estimate.work,
          "size", trial.estimate.max_size, "removed", trial.removed_domain_values, flush=True)
    small = ConstraintNetwork({k: network.coresDict[k] for k in candidate.core_keys}, maintain_indices=True)
    small_state = small.create_mac_state({c: state.domains[c] for c in small.dimension_dict})
    assert small_state.propagate()
    options = dict(max_cores=64, max_open_colors=6, max_candidates=512,
                   max_work=1_000_000, max_size=100_000, max_steps=131072)
    rows = {}
    for length in (8, 12):
        started = perf_counter()
        generator = module.ClusterCandidateGenerator(small, max_cycle_length=length, **options)
        candidates = generator.generate(small_state.domains, cyclic_only=True, compact_domains=True)
        successful = []
        for c in candidates:
            result = evaluate_candidate(small_state, c, compact_domains=True, cost_weight=0)
            if result.removed_domain_values:
                successful.append(dict(cores=c.core_keys, outputs=c.open_colors,
                    removed=result.removed_domain_values, work=result.estimate.work,
                    size=result.estimate.max_size))
        rows[f"isolated_cycle_{length}"] = dict(candidates=len(candidates),
            useful=len(successful), seconds=perf_counter()-started, successful=successful)
        print("SMALL", length, len(candidates), len(successful), flush=True)

    def timeout(*_):
        raise TimeoutError("90-second exploratory full-network guard")

    previous = signal.signal(signal.SIGALRM, timeout)
    started, full, candidates = perf_counter(), [], ()
    try:
        signal.setitimer(signal.ITIMER_REAL, 90)
        generator = module.ClusterCandidateGenerator(network, **options)
        candidates = generator.generate(state.domains, cyclic_only=True, compact_domains=True)
        for i, c in enumerate(candidates):
            result = evaluate_candidate(state, c, compact_domains=True, cost_weight=0)
            full.append(dict(index=i, removed=result.removed_domain_values,
                             keys=c.core_keys, outputs=c.open_colors))
        status = "complete"
    except TimeoutError:
        status = "guard"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert state.snapshot_domains() == before
    rows["full_sudoku_cycle_8"] = dict(status=status, candidates=len(candidates),
        evaluated=len(full), useful=sum(r["removed"] > 0 for r in full),
        seconds=perf_counter()-started, candidate_rows=full)
    result = dict(kind="Exploratory naked-triple witness, old generic contraction limits",
        generator_source_sha256=sha256(source.encode()).hexdigest(), initial=initial,
        ordinary_gac_closed_domains=before, original_cores=candidate.core_keys, outputs=candidate.open_colors,
        expert_result=dict(removed=trial.removed_domain_values, work=trial.estimate.work,
            max_size=trial.estimate.max_size, domains={c: trial.domains[c] for c in units}),
        options=options, results=rows, no_dfs=True)
    with (OUT / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("RESULT", OUT, status, len(candidates), len(full), rows["full_sudoku_cycle_8"]["useful"], flush=True)


if __name__ == "__main__":
    main()
