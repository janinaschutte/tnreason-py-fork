"""Summarize frozen hybrid trials without rerunning inference."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
digest = lambda p: sha256(p.read_bytes()).hexdigest()
plan = read(folder / "protocol.json")
reference = Path(plan["reference"])
old = read(reference / "protocol.json")
assert digest(reference / "protocol.json") == plan["reference_protocol_sha256"]
assert digest(Path(plan["data"])) == plan["data_sha256"]
assert all(digest(Path(p)) == h for p, h in plan["source_sha256"].items())
assert all(digest(Path(c["path"])) == c["sha256"] for c in plan["checkpoints"].values())
before = folder / "runtime_before.py"
assert digest(before) == old["source_sha256"]["experiments/cluster_selection/runtime.py"]
current = Path(next(p for p in plan["source_sha256"] if p.endswith("cluster_selection/runtime.py")))
tree = ast.parse(current.read_text())
tree.body = [n for n in tree.body if getattr(n, "name", None) not in ("HybridPolicy", "attach_hybrid")]
assert ast.dump(tree) == ast.dump(ast.parse(before.read_text()))
hashes = {str(Path(__file__)): digest(Path(__file__)), str(folder / "protocol.json"): digest(folder / "protocol.json")}


def row(name):
    p = folder / "rows" / name
    r = read(p)
    assert r["protocol_sha256"] == hashes[str(folder / "protocol.json")]
    assert r["status"] != "wall_timeout", "Censored trial needs a separate extension, not a fabricated total"
    assert len(r["applications"]) == r["work"]["contractions"]
    hashes[str(p)] = digest(p)
    return r


def previous(puzzle):
    filename = f"name_4096_{puzzle}.json"
    retry = reference / "retries_7200" / filename.removesuffix(".json") / "rows" / filename
    p = retry if retry.exists() else reference / "rows" / filename
    r = read(p)
    assert r["solved"] and r["valid"]
    hashes[str(p)] = digest(p)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("roots", "full"), default="roots")
    args = parser.parse_args()
    report, root_summary = [], {}
    report += ["Late learned clusters after structural allowance4096", "",
        "Exploratory original validation; fixed name branching, unchanged original tensor cores.",
        "Structural4096 then a shared learned64 allowance. Each summary is temporary; pruning is retained.",
        "The three sources are teacher growth, new REINFORCE growth (chunk52), and Q (epoch52).",
        "Root tails start independently from identical structural-domain snapshots.",
        "Extra heuristic64 is a fresh nested search under updated domains, not the next64 of the old list.",
        "Root trials alone do not establish a DFS-node improvement.", "",
        "Puzzle | Teacher | REINFORCE | Q | Portfolio | Extra heuristic (primary values removed; *solves)"]
    for i in plan["indices"]:
        h = row(f"root_heuristic_{i}.json")
        baseline = previous(i)
        for key in ("applied", "useful", "skipped", "rejected"):
            assert h["cluster_statistics"][key] == baseline["budget_trace"][0][key]
        tails = {}
        for method in plan["methods"]:
            r = row(f"root_{method}_{i}.json")
            assert r["baseline_entropy"] == h["remaining_entropy"]
            assert all(set(ds) <= set(h["domains"][v]) for v, ds in r["domains"].items())
            removed = sum(len(ds) - len(r["domains"][v]) for v, ds in h["domains"].items())
            assert removed == sum(a["removed_values"] for a in r["applications"])
            assert r["work"]["contractions"] <= plan["learned_allowance"]
            tails[method] = dict(solved=r["solved"], valid=r["valid"], removed_values=removed,
                primary_removed=sum(a["primary_removed"] for a in r["applications"]),
                contractions=r["work"]["contractions"], seconds=r["solve_s"],
                useful_applications=[dict(attempt=k, **a) for k, a in enumerate(r["applications"], 1) if a["removed_values"]])
        root_summary[i] = tails
        report.append(str(i) + " | " + " | ".join(str(tails[m]["primary_removed"]) + ("*" if tails[m]["solved"] else "")
                      for m in plan["methods"]))
    summary = dict(protocol_sha256=hashes[str(folder / "protocol.json")], roots=root_summary)
    if args.phase == "full":
        results = {i: row(f"solve_hybrid_{i}.json") for i in plan["indices"]}
        assert all(r["solved"] and r["valid"] for r in results.values())
        assert all(previous(i)["nodes"] == 1 for i in old["indices"] if i not in plan["indices"])
        reused = len(old["indices"]) - len(results)
        summary["full"] = dict(nodes=reused + sum(r["nodes"] for r in results.values()),
            branch_free=reused + sum(r["nodes"] == 1 for r in results.values()),
            reused_root_solves=reused, newly_measured={i: r["nodes"] for i, r in results.items()})
        report += ["", "Full hybrid DFS (structural4096 + portfolio64 at every propagation episode):"]
        report += [f"  {i}: heuristic {previous(i)['nodes']} -> hybrid {r['nodes']}" for i, r in results.items()]
        report += [f"Total23: {summary['full']['nodes']} nodes, {summary['full']['branch_free']} without branching.",
            f"Reuses {reused} earlier structural root solves; the hybrid stops before its learned phase on those states.",
            "The hybrid has64 additional learned contraction slots; this is an additive hybrid comparison."]
    summary["source_sha256"] = hashes
    for suffix, contents in (("json", json.dumps(summary, indent=2, allow_nan=False) + "\n"),
                             ("txt", "\n".join(report) + "\n")):
        with (folder / f"{args.phase.upper()}_REPORT.{suffix}").open("x") as stream:
            stream.write(contents)
    print("\n".join(report))


if __name__ == "__main__":
    main()
