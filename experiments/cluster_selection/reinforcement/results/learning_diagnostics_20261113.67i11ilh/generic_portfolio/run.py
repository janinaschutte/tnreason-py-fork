"""Frozen exploratory portfolio comparison; production inference is unchanged."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from types import SimpleNamespace

from experiments.cluster_selection import runtime
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
BASE = OUT.parent
SOURCES = ("heuristic", "mask", "teacher", "reinforce", "replay", "q")
METHODS = {
    "portfolio32": dict(sources=SOURCES, allowance=32, steps=8192),
    "portfolio192": dict(sources=SOURCES, allowance=192, steps=49152),
    "q32": dict(sources=("q",), allowance=32, steps=8192),
    "teacher192": dict(sources=("teacher",), allowance=192, steps=49152),
    "replay192": dict(sources=("replay",), allowance=192, steps=49152),
    "heuristic192": dict(sources=("heuristic",), allowance=192, steps=49152, batch=True),
    "structural32": dict(sources=("heuristic",), allowance=32, steps=8192),
    "structural192": dict(sources=("heuristic",), allowance=192, steps=49152),
}
CONTEXT = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


class Structural:
    """Structural proposals under each new domain snapshot, without oracle scoring."""
    kind = "construction"

    def __init__(self, template, allowance):
        self.network, self.config = template.network, dict(template.config)
        self.reward_colors, self.allowance = template.reward_colors, allowance
        self.cores = {c["key"]: SimpleNamespace(shape=c["shape"],
            colors=[self.network["colors"][i] for i in c["colors"]]) for c in self.network["cores"]}

    def proposals(self, domains, *, max_candidates=32, max_steps=8192):
        limits = {k: self.config[k] for k in ("max_cores", "max_open_colors", "max_work", "max_size")}
        generator = ClusterCandidateGenerator(self.cores, max_candidates=min(max_candidates, self.allowance),
                                              max_steps=max_steps, **limits)
        yield from generator.generate(domains, cyclic_only=True, compact_domains=True)


class Portfolio(runtime.ConstructionEnsemble):
    """Make the aggregate decoder budget explicit; reuse existing round-robin logic."""
    def __init__(self, members, steps):
        super().__init__(members)
        self.steps = steps

    def proposals(self, domains, *, max_candidates=32, max_steps=8192):
        return super().proposals(domains, max_candidates=max_candidates, max_steps=self.steps)


def freeze():
    import torch
    from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
    prior = read(BASE / "heldout/protocol.json")
    assert all(digest(p) == h for p, h in prior["source_sha256"].items())
    assert digest(prior["data"]) == prior["data_sha256"]
    data = read(prior["data"])
    domains, _ = instances(data)
    source = {
        "teacher": BASE / "heldout/teacher.pt",
        "reinforce": BASE / "heldout/reinforce.pt",
        "replay": BASE / "continuations/final_test/self_imitation.pt",
        "mask": BASE / "expert_supervised/mask.pt",
        "q": BASE.parent / "hybrid_late_20261113.eed1ut6f/q.pt",
    }
    checkpoints = {}
    for name, path in source.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["network"] == data["network"]
        assert not instance_hashes(data, domains) & set(saved["training"]["train_instance_hashes"])
        if name == "q":
            assert digest(path) == "d97586675993ca6ab92ace20d010c4876ff9cccb8cc961caacfaf092c6994fc8"
        destination = OUT / f"{name}.pt"
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        checkpoints[name] = dict(source=str(path), path=str(destination), sha256=digest(destination),
            config=saved["config"], epoch=saved["training"]["epoch"], training_overlap=0)
    refs = [BASE / folder / name for folder in ["heldout", "continuations/final_test", "expert_supervised"]
            for name in ["protocol.json", "summary.json", "audit.json"]]
    refs += [BASE / "expert_supervised/comparison.json"]
    sources = {p: h for p, h in prior["source_sha256"].items()}
    sources[str(Path(__file__))] = digest(__file__)
    order = [(method, i) for i in range(16) for method in METHODS]
    Random(2026111302).shuffle(order)
    write(OUT / "protocol.json", dict(kind="exploratory_all_generic_portfolio", data=prior["data"],
        data_sha256=prior["data_sha256"], source_sha256=sources,
        reference_sha256={str(p): digest(p) for p in refs}, checkpoints=checkpoints,
        methods=METHODS, order=order, workers=4, nodes=10000, seconds=600, branching="name",
        limits=prior["limits"], compact_domains=True, mask_anchors=8, total_work=None,
        note="Same reused16 test puzzles; no training/checkpoint selection. Six-source round-robin, "
             "two frozen REINFORCE variants. Limits are per episode/refresh: global contraction and "
             "decoder budgets shared; unused/exhausted streams redistribute. Regenerate all source "
             "proposals after useful pruning. Original selected Q decoder unchanged. Structural-only "
             "refreshed controls distinguish regeneration from combining learned proposals. "
             "Larger192 arms receive49152 global steps (six times8192); equal caps are not equal actual work."))
    (OUT / "rows").mkdir()
    print("FROZEN128solves; no solver queries yet", flush=True)


def initialize():
    import numpy as np
    import torch
    from tnreason.engine.workload_to_numpy import NumpyCore
    from experiments.constraint_networks.constraint_networks import ConstraintNetwork
    from experiments.cluster_selection.reinforcement.experiment import instances
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(OUT / "protocol.json")
    assert all(digest(p) == h for p, h in {**plan["source_sha256"], **plan["reference_sha256"]}.items())
    assert digest(plan["data"]) == plan["data_sha256"]
    models = {}
    for name, info in plan["checkpoints"].items():
        assert digest(info["path"]) == info["sha256"]
        models[name] = runtime.load(info["path"]).cpu().eval()
        models[name].config["compact_domains"] = True
        if name == "mask":
            models[name].config["anchor_candidates"] = 8
    data = read(plan["data"])
    domains, _ = instances(data)
    raw = data["network"]
    network = ConstraintNetwork({c["key"]: NumpyCore(values=np.asarray(c["values"], dtype=bool).reshape(c["shape"]),
        colors=[raw["colors"][i] for i in c["colors"]]) for c in raw["cores"]}, maintain_indices=True)
    network.solve_mac = partial(network.solve_mac, variable_tie_break="name")
    CONTEXT.update(plan=plan, models=models, network=network, domains=domains)


def evaluate(task):
    import torch
    from experiments.cluster_selection.reinforcement.experiment import solve
    method, position = task
    path = OUT / "rows" / f"{method}_{position:02d}.json"
    if path.exists():
        row = read(path)
        assert row["protocol_sha256"] == digest(OUT / "protocol.json")
        return row
    plan, models = CONTEXT["plan"], CONTEXT["models"]
    settings = plan["methods"][method]
    members = {name: Structural(models["teacher"], settings["allowance"])
               if name == "heuristic" else models[name] for name in settings["sources"]}
    model = Portfolio(members, settings["steps"])
    original_apply, original_attach, original_h = runtime.apply_candidate, runtime.attach, runtime.attach_heuristic
    applications, episodes, root_trace = [], [], []
    counts = defaultdict(Counter)
    def snapshot(state):
        return sha256(json.dumps(list(state.domains.items()), separators=(",", ":")).encode()).hexdigest()
    def applied(state, candidate, prepared=None, **options):
        source = "heuristic" if settings.get("batch") else model.last_source
        root = state.decision_level == 0
        before = snapshot(state) if root else None
        result = original_apply(state, candidate, prepared, **options)
        useful = bool(result.removed_values or result.contradiction)
        applications.append(dict(source=source, root=root, useful=useful))
        counts[source].update(contractions=1, useful=int(useful), removed=result.removed_values,
                              root_contractions=int(root), root_useful=int(root and useful))
        if root and useful:
            root_trace.append(dict(source=source, core_keys=candidate.core_keys, open_colors=candidate.open_colors,
                removed=result.removed_values, contradiction=result.contradiction,
                before=before, after=snapshot(state)))
        return result
    def attached(function, state, *args, **options):
        result = function(state, *args, **options)
        policy = state.on_closure
        def closure(state, changed):
            start = len(applications)
            try:
                return policy(state, changed)
            finally:
                count = len(applications)-start
                assert count <= settings["allowance"]
                episodes.append(count)
        closure.statistics = policy.statistics
        state.on_closure = closure
        return result
    runtime.apply_candidate = applied
    runtime.attach = partial(attached, original_attach)
    runtime.attach_heuristic = partial(attached, original_h)
    try:
        with torch.no_grad():
            row = solve(CONTEXT["network"], CONTEXT["domains"][position],
                "heuristic" if settings.get("batch") else "trained", model,
                nodes=plan["nodes"], seconds=plan["seconds"], total_work=None,
                baseline_candidates=settings["allowance"], baseline_steps=settings["steps"],
                cluster_attempts=settings["allowance"])
    finally:
        runtime.apply_candidate, runtime.attach, runtime.attach_heuristic = original_apply, original_attach, original_h
    assert row["status"] not in ("invalid", "unsat"), row
    incomplete = row["work"]["contractions"]-len(applications)
    assert incomplete >= 0 and (not incomplete or row["status"] == "wall_timeout")
    row.update(method=method, position=position, protocol_sha256=digest(OUT / "protocol.json"),
        episode_contractions=episodes, source_statistics=dict(counts), root_trace=root_trace,
        root_contractions=sum(a["root"] for a in applications), incomplete_contractions=incomplete)
    write(path, row)
    print(path.name, row["status"], row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
    return row


def report():
    rows = [read(p) for p in sorted((OUT / "rows").glob("*.json"))]
    assert len(rows) == 128 and len({(r["method"], r["position"]) for r in rows}) == 128
    result = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        complete = sum(r["status"] == "solved" and r["valid"] for r in group)
        result[method] = dict(completed=complete, nodes=sum(r["nodes"] for r in group) if complete == 16 else None,
            branch_free=sum(r["status"] == "solved" and r["nodes"] == 1 for r in group),
            root_positions=[r["position"] for r in group if r["status"] == "solved" and r["nodes"] == 1],
            solve_seconds=sum(r["solve_s"] for r in group),
            contractions=sum(r["work"]["contractions"] for r in group),
            estimated_work=sum(r["work"]["estimated_work"] for r in group),
            censored=[r["position"] for r in group if r["status"] != "solved"])
    write(OUT / "summary.json", dict(protocol_sha256=digest(OUT / "protocol.json"), results=result,
        row_sha256={p.name: digest(p) for p in sorted((OUT / "rows").glob("*.json"))}))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "evaluate", "report"])
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    elif phase == "evaluate":
        plan = read(OUT / "protocol.json")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"), initializer=initialize) as pool:
            for job in as_completed([pool.submit(evaluate, task) for task in plan["order"]]):
                job.result()
    else:
        report()
