"""Validate completed rows and extend the saved validation figures; no solver runs."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

folder = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
digest = lambda p: sha256(p.read_bytes()).hexdigest()
protocol = read(folder / "protocol.json")
previous = Path(protocol["previous"]["folder"])
if not previous.is_absolute():
    previous = folder.parent / previous
assert digest(previous / "protocol.json") == protocol["previous"]["protocol_sha256"]
assert digest(previous / "comparison.json") == protocol["previous"]["comparison_sha256"]
old = read(previous / "protocol.json")
old_comparison = read(previous / "comparison.json")
assert old["indices"] == protocol["indices"] and old["limits"] == protocol["limits"]
assert old["data_sha256"] == protocol["data_sha256"] and old["guards"]["nodes"] == protocol["guards"]["nodes"]
indices, ties = protocol["indices"], ("name", "oldest_pruned")
allowances = old["allowances"] + protocol["allowances"]
assert allowances == sorted(set(allowances))
labels = {"name": "Lexicographical", "oldest_pruned": "Oldest-pruned"}
hashes = {str(p): digest(p) for p in (folder / "protocol.json", previous / "protocol.json",
                                    previous / "comparison.json", Path(__file__))}
records, summaries, attempts, retries = {}, {}, {}, {}
for path in sorted(folder.glob("retries_*/*/protocol.json")):
    spec = read(path)
    assert len(spec["cases"]) == 1
    assert all(spec[k] == protocol[k] for k in ("data_sha256", "indices", "limits"))
    assert all(spec["source_sha256"].get(k) == v for k, v in protocol["source_sha256"].items())
    assert spec["guards"]["nodes"] == protocol["guards"]["nodes"]
    assert spec["guards"]["seconds"] >= protocol["guards"]["seconds"]
    assert Path(spec["retry_of"]["folder"]).resolve() == folder
    assert spec["retry_of"]["protocol_sha256"] == hashes[str(folder / "protocol.json")]
    retries.setdefault(tuple(spec["cases"][0]), []).append((path, spec))
for a in allowances:
    source, spec = (folder, protocol) if a in protocol["allowances"] else (previous, old)
    for t in ties:
        rows = []
        for i in indices:
            path = source / "rows" / f"{t}_{a}_{i}.json"
            r = read(path)
            hashes[str(path)] = digest(path)
            if source == previous:
                assert hashes[str(path)] == old_comparison["source_sha256"][str(path.relative_to(previous))]
            assert (r["tie"], r["puzzle"], r["allowance"]) == (t, i, a)
            assert r["protocol_sha256"] == hashes[str(source / "protocol.json")]
            assert r["status"] in ("solved", "wall_timeout", "node_limit")
            assert r["solved"] == (r["status"] == "solved")
            assert not r["solved"] or (r["valid"] and isinstance(r["nodes"], int) and r["nodes"] >= 1)
            guard = spec["guards"]["seconds"]
            def note(row, p, seconds):
                return dict(path=str(p), row_sha256=digest(p), wall_guard_seconds=seconds,
                    **{k: row[k] for k in ("status", "solved", "nodes", "valid", "solve_s", "protocol_sha256")})
            for retry_path, retry in sorted(retries.get((t, a, i), []), key=lambda x: x[1]["guards"]["seconds"]):
                assert source == folder and read(path)["status"] == "wall_timeout"
                assert retry["retry_of"]["row_sha256"] == hashes[str(path)]
                key = f"{t}_{a}_{i}"
                attempts.setdefault(key, [note(r, path, guard)])
                p = retry_path.parent / "rows" / path.name
                r = read(p)  # A launched retry must finish before any output is written.
                hashes.update({str(retry_path): digest(retry_path), str(p): digest(p)})
                assert r["protocol_sha256"] == hashes[str(retry_path)]
                assert (r["tie"], r["puzzle"], r["allowance"]) == (t, i, a)
                assert r["status"] in ("solved", "wall_timeout", "node_limit") and r["solved"] == (r["status"] == "solved")
                assert not r["solved"] or (r["valid"] and isinstance(r["nodes"], int) and r["nodes"] >= 1)
                guard = retry["guards"]["seconds"]
                attempts[key].append(note(r, p, guard))
            r["wall_guard_seconds"] = guard
            records[t, i, a] = r
            rows.append(r)
        traces = [e for r in rows for e in r.get("budget_trace", [])]
        if a in protocol["allowances"]:
            assert all("budget_trace" in r for r in rows)
            assert all(e["allowance"] == a and 0 <= e["applied"] <= a and
                       (e["proposed"] is None or e["applied"] <= e["proposed"] <= a) for e in traces)
        complete = [e for e in traces if e["proposed"] is not None]
        summaries[t, a] = dict(completed=sum(r["solved"] for r in rows),
            nodes=sum(r["nodes"] for r in rows) if all(r["solved"] for r in rows) else None,
            branch_free=sum(r["solved"] and r["nodes"] == 1 for r in rows), initial_wall_guard_seconds=spec["guards"]["seconds"],
            wall_guard_seconds_by_puzzle={r["puzzle"]: r["wall_guard_seconds"] for r in rows},
            censored=[dict(puzzle=r["puzzle"], status=r["status"], partial_nodes=r["nodes"]) for r in rows if not r["solved"]],
            cluster_totals={k: sum(r["cluster_statistics"].get(k, 0) for r in rows)
                            for k in ("calls", "proposed", "applied", "useful")},
            budget_usage=dict(episodes=len(traces), completed_generation=len(complete),
                interrupted_generation=len(traces)-len(complete),
                proposal_cap_hits=sum(e["proposed"] == a for e in complete),
                application_cap_hits=sum(e["applied"] == a for e in traces),
                max_proposed=max((e["proposed"] for e in complete), default=None),
                max_applied=max((e["applied"] for e in traces), default=None)))

assert len(attempts) == len(retries), "Retry case is outside the completed base sweep"
order = old_comparison["row_order"]
assert set(order) == set(indices)
comparison = dict(source_sha256=hashes, retry_attempts=attempts, row_order=order, allowances={}, oldest_allowance_changes={})
for a in allowances:
    paired = [i for i in indices if all(records[t, i, a]["solved"] for t in ties)]
    delta = {i: records["oldest_pruned", i, a]["nodes"] - records["name", i, a]["nodes"] for i in paired}
    comparison["allowances"][a] = dict(summaries={t: summaries[t, a] for t in ties}, solved_pairs=len(paired),
        wins=sum(d < 0 for d in delta.values()), ties=sum(d == 0 for d in delta.values()),
        losses=sum(d > 0 for d in delta.values()), paired_node_delta=sum(delta.values()), paired_deltas=delta,
        sensitivity_puzzle_113={t: dict(puzzle={k: records[t, 113, a][k] for k in ("nodes", "status", "solve_s")},
            other_22_nodes=sum(records[t, i, a]["nodes"] for i in indices if i != 113)
                if all(records[t, i, a]["solved"] for i in indices if i != 113) else None) for t in ties})
for before, after in zip(allowances, allowances[1:]):
    delta = {i: records["oldest_pruned", i, after]["nodes"] - records["oldest_pruned", i, before]["nodes"]
             for i in indices if all(records["oldest_pruned", i, a]["solved"] for a in (before, after))}
    comparison["oldest_allowance_changes"][f"{before}->{after}"] = dict(pairs=len(delta),
        improved=sum(d < 0 for d in delta.values()), tied=sum(d == 0 for d in delta.values()),
        worsened=sum(d > 0 for d in delta.values()), paired_node_delta=sum(delta.values()),
        changed_puzzles={i: d for i, d in delta.items() if d})

stems = ("dfs_nodes_oldest_pruned", "dfs_nodes_branching_comparison")
outputs = ["nodes.csv", "comparison.json", "REPORT.txt"] + [f"{s}.{e}" for s in stems for e in ("png", "pdf")]
for name in outputs:
    if (folder / name).exists():
        raise FileExistsError(folder / name)
with (folder / "nodes.csv").open("x", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["puzzle_index", "allowance", "branching", "status", "dfs_nodes", "partial_dfs_nodes",
                     "solve_seconds", "wall_guard_seconds", "proposed", "applied", "useful"])
    for i in indices:
        for a in allowances:
            for t in ties:
                r = records[t, i, a]
                writer.writerow([i, a, t, r["status"], r["nodes"] if r["solved"] else None,
                    None if r["solved"] else r["nodes"], r["solve_s"], r["wall_guard_seconds"],
                    *[r["cluster_statistics"].get(k, 0) for k in ("proposed", "applied", "useful")]])
with (folder / "comparison.json").open("x") as stream:
    json.dump(comparison, stream, indent=2, allow_nan=False)
    stream.write("\n")

maximum = max(r["nodes"] for r in records.values() if r["solved"])
norm = LogNorm(vmin=1, vmax=max(2, maximum))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#ececec")
guard_note = (f"{min(old['allowances']):,}–{max(old['allowances']):,}: {old['guards']['seconds']:,} s/solve; "
    f"{min(protocol['allowances']):,}–{max(protocol['allowances']):,}: {protocol['guards']['seconds']:,} s/solve. "
    f"Retries: up to {max(r['wall_guard_seconds'] for r in records.values()):,} s. Hatched cells: censored counts.")
for selected, stem in [(("oldest_pruned",), stems[0]), (ties, stems[1])]:
    paired = len(selected) == 2
    left, right = (.055, .91) if paired else (.11, .85)
    fig, axes = plt.subplots(1, len(selected), figsize=(18 if paired else 10, 10.5), squeeze=False)
    fig.subplots_adjust(left=left, right=right, bottom=.12, top=.81, wspace=.13)
    for ax, t in zip(axes[0], selected):
        values = np.array([[records[t, i, a]["nodes"] if records[t, i, a]["solved"] else np.nan
                            for a in allowances] for i in order])
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(allowances)), labels=[f"{a:,}" for a in allowances])
        ax.set_yticks(range(len(order)), labels=order)
        ax.tick_params(axis="both", which="both", length=0, pad=7)
        ax.xaxis.tick_top()
        ax.set_title(labels[t], pad=27, fontsize=13, fontweight="bold")
        ax.set_ylabel("Sudoku index")
        ax.set_xticks(np.arange(-.5, len(allowances), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row, i in enumerate(order):
            for col, a in enumerate(allowances):
                r = records[t, i, a]
                if not r["solved"]:
                    ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor="#e6e6e6",
                                          edgecolor="#8c8c8c", hatch="///", linewidth=0))
                text = str(r["nodes"]) if r["solved"] else ("Timeout" if r["status"] == "wall_timeout" else "Node cap")
                ax.text(col, row, text, ha="center", va="center", fontsize=9 if paired else 11,
                        color="white" if r["solved"] and norm(r["nodes"]) > .55 else "#17212b")
    ticks = [n for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if n <= maximum]
    cax = fig.add_axes([.935 if paired else .88, .38, .013 if paired else .019, .37])
    cb = fig.colorbar(image, cax=cax, ticks=ticks)
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([str(n) for n in ticks])
    cb.set_label("DFS nodes (shared log colour scale)", labelpad=10)
    fig.suptitle("DFS nodes for each Sudoku", x=left, y=.965, ha="left", fontsize=18)
    fig.text(left, .925, "Structural heuristic · same 23 validation puzzles · original row order", fontsize=11)
    fig.text(left, .89, "Columns: cluster allowance per propagation episode", fontsize=11)
    fig.text(left, .065, "1 node = solved without branching. Same row order and colour scale in both figures.", fontsize=10)
    fig.text(left, .039, guard_note, fontsize=9)
    for suffix in ("png", "pdf"):
        with (folder / f"{stem}.{suffix}").open("xb") as stream:
            fig.savefig(stream, format=suffix, dpi=180, facecolor="white")
    plt.close(fig)

report = ["Extended structural cluster allowance sweep", "",
    "Same original 23 validation Sudoku puzzles; no retraining and no final-test queries.",
    "32–1024 are immutable previous paired runs; 2048 and 4096 are new paired runs.", guard_note,
    "Wall guards cover generation, contractions, GAC and DFS. Both sweeps have a 100000-node guard.",
    "Wall times reflect concurrent CPU jobs; method comparisons here use DFS nodes.",
    "Three unfinished 7200s retries (name/4096/112, name/4096/113, oldest_pruned/4096/113) were restarted after interruption; their times come from the saved fresh attempts.",
    "Both rules use MRV, highest degree, then name or oldest retained pruning followed by name.",
    "Same nested proposal generator and per-cluster limits; no total-work cap. Defaults remain unchanged.",
    "Totals require all 23 valid solved rows; censored partial counts are not completed solve counts.",
    "Wins/ties/losses compare solved pairs only, with oldest-pruned minus name node deltas.", "",
    "Allowance | Name nodes | Oldest nodes | Branch-free name/oldest | Solved pairs | Wins/ties/losses"]
for a, c in comparison["allowances"].items():
    s = c["summaries"]
    report.append(f"{a} | {s['name']['nodes']} | {s['oldest_pruned']['nodes']} | "
        f"{s['name']['branch_free']}/{s['oldest_pruned']['branch_free']} | {c['solved_pairs']} | {c['wins']}/{c['ties']}/{c['losses']}")
    for t in ties:
        sens, totals = c["sensitivity_puzzle_113"][t], s[t]["cluster_totals"]
        report.append(f"  {t}: completed {s[t]['completed']}/23; puzzle 113 {sens['puzzle']}; other 22 nodes {sens['other_22_nodes']}; "
                      f"cluster calls/proposed/applied/useful {totals}; censored {s[t]['censored']}")
        if a in protocol["allowances"]:
            report.append(f"    Per-episode cap usage: {s[t]['budget_usage']}")
report += ["", "Oldest-pruned allowance changes (delta is larger minus smaller allowance):"]
for transition, c in comparison["oldest_allowance_changes"].items():
    report.append(f"  {transition}: solved pairs {c['pairs']}; improve/tie/worsen {c['improved']}/{c['tied']}/{c['worsened']}; "
                  f"paired node delta {c['paired_node_delta']:+d}; changed puzzles {c['changed_puzzles']}")
report += ["", "Retries replace only original wall-time-censored cases, using the largest launched guard.",
           "Initial attempts and all retry hashes are retained; highest-guard finished attempts are used even if still censored."]
for case, history in attempts.items():
    report.append(f"  {case}: {history}")
report += ["", f"Validated {len(records)} solve rows; original and new source hashes are in comparison.json.",
    "Allowance is a cap, not a promise that every proposed cluster is applied or useful.",
    "A flat observed node count does not prove that no useful clusters exist outside this structural generator.",
    "CSV includes whole-solve time and its guard, plus actual proposal/application/usefulness counts."]
with (folder / "REPORT.txt").open("x") as stream:
    stream.write("\n".join(report) + "\n")
print("\n".join(report))
