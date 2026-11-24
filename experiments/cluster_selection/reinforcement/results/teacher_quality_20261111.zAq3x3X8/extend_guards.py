"""Retry all wall-censored solves with a common 600-second guard, retaining originals."""
import fcntl
import resume as continued

frozen = continued.frozen
OUT = frozen.OUT
read, write = continued.read, continued.write


def main():
    with (OUT / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extend()


def extend():
    frozen.torch.set_num_threads(1)
    frozen.torch.set_num_interop_threads(1)
    protocol = read(OUT / "protocol.json")
    extension = read(OUT / "guard_extension_protocol.json")
    if extension["code_sha256"] != frozen.digest(__file__):
        raise AssertionError("Guard-extension source changed")
    for name, expected in protocol["code_sha256"].items():
        if frozen.digest(frozen.ROOT / name) != expected:
            raise AssertionError(f"Frozen code changed: {name}")
    if not (OUT / "summary.json").exists():
        raise ValueError("Complete the original comparison first")
    if frozen.digest(frozen.DATA) != protocol["data_sha256"]:
        raise AssertionError("Frozen data changed")
    data = read(frozen.DATA)
    initial, _ = frozen.instances(data)
    parent = frozen.load(frozen.PARENT).cpu().eval()
    state = frozen._state({**data, "samples": [dict(domains=[list(range(d))
                          for d in data["network"]["dimensions"]])]})
    summary = {}
    for name in read(OUT / "summary.json"):
        source = OUT / f"evaluation_{name}.json"
        original = read(source)
        rows, retried = [], []
        checkpoint = OUT / f"{name}.pt" if name in frozen.ARMS else frozen.PARENT
        model = frozen.load(checkpoint).cpu().eval()
        model.beta = 0.
        model.config.pop("max_total_work", None)
        model.config["decoder"] = "discrepancy"
        if model.network != parent.network:
            raise AssertionError("Runtime network changed")
        provenance = dict(original_evaluation_sha256=frozen.digest(source),
            checkpoint_sha256=frozen.digest(checkpoint),
            extension_protocol_sha256=frozen.digest(OUT / "guard_extension_protocol.json"))
        for row in original["rows"]:
            puzzle = row["puzzle"]
            if row["status"] == "wall_timeout":
                retried.append(puzzle)
                output = OUT / f"extended_{name}_{puzzle}.json"
                if output.exists():
                    saved = read(output)
                    if saved["provenance"] != provenance:
                        raise AssertionError("Retry provenance changed")
                    row = saved["row"]
                else:
                    mode = "plain" if name == "plain" else "heuristic" if name == "heuristic32" else "trained"
                    with frozen.torch.no_grad():
                        row = dict(puzzle=puzzle, **frozen.solve(state.constraint_network, initial[puzzle],
                            mode, model, nodes=10000, seconds=600, cluster_attempts=32,
                            baseline_candidates=32, baseline_steps=8192, stop_on_noop=False))
                    if row["status"] in ("invalid", "unsat"):
                        raise AssertionError("Invalid validation solution")
                    write(output, dict(provenance=provenance, row=row))
                print("retry", name, puzzle, row["status"], row["nodes"],
                      f"{row['solve_s']:.2f}s", flush=True)
            rows.append(row)
        complete = all(r["solved"] and r["valid"] for r in rows)
        compact = {k: v for k, v in original.items() if k not in ("rows", "greedy_roots")}
        compact.update(complete=complete, nodes=sum(r["nodes"] for r in rows) if complete else None,
            branch_free=sum(r["solved"] and r["nodes"] == 1 for r in rows),
            retried_puzzles=retried, statuses=[r["status"] for r in rows],
            seconds_guard=600, node_guard=10000)
        summary[name] = compact
        print("extended evaluation", name, compact["nodes"], compact["branch_free"], flush=True)
    if not (OUT / "summary_extended.json").exists():
        write(OUT / "summary_extended.json", summary)


if __name__ == "__main__":
    main()
