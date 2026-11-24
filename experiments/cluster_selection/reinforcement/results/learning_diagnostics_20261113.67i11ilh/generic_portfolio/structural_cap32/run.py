"""Post-hoc structural breadth diagnostic; reuse the frozen portfolio evaluator."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
ROOT = PARENT.parents[5]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("frozen_portfolio_parent", PARENT / "run.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
METHOD = "portfolio192cap32"
SETTINGS = dict(sources=parent.SOURCES, allowance=192, steps=49152)


class Structural32(parent.Structural):
    def __init__(self, template, allowance):
        super().__init__(template, 32)


parent.Structural = Structural32
parent.OUT = OUT


def freeze():
    import torch
    original = parent.read(PARENT / "protocol.json")
    assert all(parent.digest(p) == h for p, h in
               {**original["source_sha256"], **original["reference_sha256"]}.items())
    plan = deepcopy(original)
    plan.update(kind="posthoc_structural_list_cap32_diagnostic", methods={METHOD: SETTINGS},
                structural_list_cap=32, order=[(METHOD, i) for i in range(16)])
    Random(2026111303).shuffle(plan["order"])
    plan["source_sha256"][str(Path(__file__))] = parent.digest(__file__)
    plan["reference_sha256"].update({str(PARENT / name): parent.digest(PARENT / name)
        for name in ("protocol.json", "summary.json", "audit.json")})
    plan["note"] = ("Post-hoc diagnostic after the original portfolio's structural source yielded no "
        "contractions. Same reused16 test puzzles and frozen five learned models; no model selection "
        "or training. Six-source round-robin order and192 global contraction allowance unchanged; "
        "49152 aggregate steps give8192 per source. Only the structural proposal-list cap changes "
        "from192 to32, preserving H32 traversal breadth. Refresh all streams after useful pruning. "
        "Original experiment files are unchanged. Equal caps are not equal actual computation.")
    models = {}
    for name, info in plan["checkpoints"].items():
        assert parent.digest(info["path"]) == info["sha256"]
        models[name] = parent.runtime.load(info["path"]).cpu().eval()
        models[name].config["compact_domains"] = True
        if name == "mask":
            models[name].config["anchor_candidates"] = 8
    members = {name: Structural32(models["teacher"], 192) if name == "heuristic" else models[name]
               for name in parent.SOURCES}
    parent.Portfolio(members, 49152)
    structural = members["heuristic"]
    limits = {key: structural.config[key] for key in
              ("max_cores", "max_open_colors", "max_work", "max_size")}
    generator = parent.ClusterCandidateGenerator(structural.cores, max_candidates=32,
                                                max_steps=8192, **limits)
    assert generator.max_candidates == 32 and generator.max_steps == 8192
    assert all(m.network == models["teacher"].network for m in members.values())
    parent.write(OUT / "protocol.json", plan)
    (OUT / "rows").mkdir()
    print("FROZEN16solves; six-source config/shape preflight passed; no solver queries", flush=True)


def initialize():
    parent.initialize()


def evaluate(task):
    return parent.evaluate(task)


def report():
    plan = parent.read(OUT / "protocol.json")
    rows = [parent.read(p) for p in sorted((OUT / "rows").glob("*.json"))]
    assert len(rows) == 16 and {r["position"] for r in rows} == set(range(16))
    complete = sum(r["status"] == "solved" and r["valid"] for r in rows)
    counts = defaultdict(Counter)
    for row in rows:
        for source, values in row["source_statistics"].items():
            counts[source].update(values)
    result = dict(completed=complete,
        nodes=sum(r["nodes"] for r in rows) if complete == 16 else None,
        branch_free=sum(r["status"] == "solved" and r["nodes"] == 1 for r in rows),
        root_positions=[r["position"] for r in rows if r["status"] == "solved" and r["nodes"] == 1],
        solve_seconds=sum(r["solve_s"] for r in rows),
        contractions=sum(r["work"]["contractions"] for r in rows),
        estimated_work=sum(r["work"]["estimated_work"] for r in rows),
        censored=[r["position"] for r in rows if r["status"] != "solved"],
        source_statistics=dict(counts))
    parent.write(OUT / "summary.json", dict(protocol_sha256=parent.digest(OUT / "protocol.json"),
        results={METHOD: result}, row_sha256={p.name: parent.digest(p)
            for p in sorted((OUT / "rows").glob("*.json"))}))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate", "report"))
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    elif phase == "evaluate":
        plan = parent.read(OUT / "protocol.json")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=get_context("spawn"),
                                 initializer=initialize) as pool:
            for job in as_completed([pool.submit(evaluate, task) for task in plan["order"]]):
                job.result()
    else:
        report()
