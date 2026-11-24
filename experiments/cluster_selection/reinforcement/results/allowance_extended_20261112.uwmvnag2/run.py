"""Resume a paired larger-allowance sweep; save every puzzle atomically."""
import argparse
from functools import partial
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
from random import Random
import runpy
import sys
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection import runtime
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
from experiments.cluster_selection.reinforcement.experiment import instances, solve
from experiments.cluster_selection.reinforcement.train import _state

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "allowance_age_20261112.w9v5x1hg"
NETWORK, DOMAINS, MODEL, ORIGINAL_SOLVE, TRACE = None, None, None, None, []
ORIGINAL_ATTACH = runtime.attach_heuristic


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def signature(candidates):
    return tuple((candidate.core_keys, candidate.open_colors) for candidate in candidates)


def install_nested(generator, allowance):
    """Use the existing doubling generator, retaining the old cache limits."""
    original = generator.generate

    def nested(domains, **options):
        options.pop("max_candidates", None)
        steps, wrapper = generator.max_steps, generator.generate
        entry = dict(allowance=allowance, proposed=None, stages=[])
        generator.last_generation = entry

        def stage(domains, **generation):
            entry["stages"].append([generation["max_candidates"], generator.max_steps])
            return original(domains, **generation)

        generator.generate, generator.max_steps = stage, 256 * allowance
        try:
            candidates = generator.generate_nested(domains, max_candidates=allowance, **options)
            expected, quota = [], 32
            while quota <= allowance:
                expected.append([quota, 256 * quota])
                quota *= 2
            assert entry["stages"] == expected
            assert len(candidates) <= allowance
            entry.update(proposed=len(candidates),
                         prefix_1024_sha256=sha256(repr(signature(candidates[:1024])).encode()).hexdigest())
            return candidates
        finally:
            generator.generate, generator.max_steps = wrapper, steps
            assert generator.max_steps == 8192

    generator.generate = nested


def staged_attach(state, **options):
    quota = options.pop("max_candidates")
    result = ORIGINAL_ATTACH(state, max_candidates=quota, max_steps=8192, **options)
    policy = state.on_closure
    install_nested(policy.generator, quota)

    def observed(state, changed):
        before = dict(policy.statistics)
        policy.generator.last_generation = dict(allowance=quota, proposed=None, stages=[])
        try:
            return policy(state, changed)
        finally:
            if policy.statistics["calls"] > before["calls"]:
                entry = dict(policy.generator.last_generation)
                entry.update({key: policy.statistics[key] - before[key]
                              for key in ("applied", "useful", "skipped", "rejected")})
                if entry["proposed"] is not None:
                    assert entry["applied"] <= entry["proposed"]
                TRACE.append(entry)

    observed.statistics = policy.statistics
    state.on_closure = observed
    return result


def init_worker():
    global NETWORK, DOMAINS, MODEL, ORIGINAL_SOLVE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = read(OUT / "protocol.json")
    data = read(protocol["data"])
    selected = {**data, "puzzles": [data["puzzles"][i] for i in protocol["indices"]]}
    initial, _ = instances(selected)
    DOMAINS = dict(zip(protocol["indices"], initial))
    NETWORK = _state({**data, "samples": [dict(domains=[list(range(d))
                     for d in data["network"]["dimensions"]])]}).constraint_network
    ORIGINAL_SOLVE = NETWORK.solve_mac
    MODEL = SimpleNamespace(config=protocol["limits"], reward_colors=data["reward_colors"])
    runtime.attach_heuristic = staged_attach


def audit():
    """Compare real, GAC-closed Sudoku domains against the frozen old adapter."""
    protocol = read(OUT / "protocol.json")
    old_path = OUT.parent / "allowance_plateau_20261030.bih66zeb" / "sweep.py"
    old_nested = runpy.run_path(str(old_path))["nested_generate"]
    state = NETWORK.create_mac_state(DOMAINS[113])
    state.propagate()
    limits = {key: value for key, value in protocol["limits"].items() if key != "compact_domains"}
    options = dict(changed_colors=set(state.domains), cyclic_only=True, compact_domains=True)
    original = ClusterCandidateGenerator(state._cores, max_candidates=1024, max_steps=8192, **limits)
    original.generate_original = original.generate
    previous = signature(old_nested(original, state.domains, 1024, **options))
    checks = []
    for quota in (1024, 2048, 4096):
        generator = ClusterCandidateGenerator(state._cores, max_candidates=quota, max_steps=8192, **limits)
        install_nested(generator, quota)
        current = signature(generator.generate(state.domains, **options))
        assert current[:len(previous)] == previous
        assert generator.max_steps == 8192
        checks.append(dict(allowance=quota, proposals=len(current), previous_prefix=len(previous),
                           stages=generator.last_generation["stages"]))
        previous = current
        print("PREFIX AUDIT", checks[-1], flush=True)
    value = dict(protocol_sha256=digest(OUT / "protocol.json"), puzzle=113,
                 domains="Root after ordinary GAC", checks=checks)
    write(OUT / "prefix_audit.json", value)


def run_case(case):
    global TRACE
    tie, allowance, puzzle = case
    protocol = read(OUT / "protocol.json")
    protocol_hash = digest(OUT / "protocol.json")
    path = OUT / "rows" / f"{tie}_{allowance}_{puzzle}.json"
    if path.exists():
        record = read(path)
        assert record["protocol_sha256"] == protocol_hash
        return record
    assert all(digest(ROOT / name) == expected for name, expected in protocol["source_sha256"].items())
    NETWORK.solve_mac = partial(ORIGINAL_SOLVE, variable_tie_break=tie)
    TRACE = []
    try:
        row = solve(NETWORK, DOMAINS[puzzle], "heuristic", MODEL, **protocol["guards"],
                    total_work=None, baseline_candidates=allowance)
    finally:
        NETWORK.solve_mac = ORIGINAL_SOLVE
    assert row["status"] in ("solved", "wall_timeout", "node_limit")
    assert not row["solved"] or row["valid"]
    record = dict(protocol_sha256=protocol_hash, tie=tie, allowance=allowance,
                  puzzle=puzzle, budget_trace=TRACE, **row)
    write(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    previous = read(PREVIOUS / "protocol.json")
    backend = {name: expected for name, expected in previous["source_sha256"].items()
               if "/results/" not in name}
    assert all(digest(ROOT / name) == expected for name, expected in backend.items())
    assert digest(previous["data"]) == previous["data_sha256"]
    cases = [(tie, a, i) for tie in ("name", "oldest_pruned") for a in (2048, 4096)
             for i in previous["indices"]]
    Random(67).shuffle(cases)
    protocol = dict(previous={"folder": str(PREVIOUS), "protocol_sha256": digest(PREVIOUS / "protocol.json"),
                    "comparison_sha256": digest(PREVIOUS / "comparison.json")},
        data=previous["data"], data_sha256=previous["data_sha256"], indices=previous["indices"],
        allowances=[2048, 4096], cases=cases, seed=67, workers=4,
        limits=previous["limits"], guards=dict(nodes=100000, seconds=1800), total_work=None,
        source_sha256={**backend, str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        no_training=True, no_test_queries=True, threads_per_worker=1,
        generation="Existing generate_nested with stage budget 256*quota; restore max_steps8192 between calls.",
        wall_guard_note="New paired arms both allow1800s; previous arms used600s. Completed old node counts remain valid; timeouts stay censored.")
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == json.loads(json.dumps(protocol)), "Frozen protocol changed"
    else:
        write(path, protocol)
    (OUT / "rows").mkdir(exist_ok=True)
    if args.prepare:
        print("Prepared", path, flush=True)
        return
    if not (OUT / "prefix_audit.json").exists():
        init_worker()
        audit()
    else:
        assert read(OUT / "prefix_audit.json")["protocol_sha256"] == digest(path)
    if args.audit:
        return
    with mp.get_context("spawn").Pool(protocol["workers"], initializer=init_worker) as pool:
        for n, row in enumerate(pool.imap_unordered(run_case, cases, chunksize=1), 1):
            print(n, "/", len(cases), row["tie"], row["allowance"], row["puzzle"],
                  row["status"], row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
    print("COMPLETE", len(cases), "saved puzzle solves", flush=True)


if __name__ == "__main__":
    main()
