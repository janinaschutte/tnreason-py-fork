"""Continue the frozen experiment; save each completed solve before proceeding."""
from copy import deepcopy
import fcntl
import json
from random import Random
import tempfile
from time import perf_counter

import run as frozen

torch = frozen.torch
OUT = frozen.OUT


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = frozen.Path(stream.name)
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    # A second continuation must not race over checkpoints or completed rows.
    with (OUT / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resume()


def resume():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = read(OUT / "protocol.json")
    plan = read(OUT / "training_plan.json")

    def verify():
        for name, expected in protocol["code_sha256"].items():
            if frozen.digest(frozen.ROOT / name) != expected:
                raise AssertionError(f"Frozen code changed: {name}")
        for name, path in (("parent", frozen.PARENT), ("data", frozen.DATA),
                           ("states", frozen.OLD / "states.json"),
                           ("old_protocol", frozen.OLD / "protocol.json")):
            if frozen.digest(path) != protocol[name + "_sha256"]:
                raise AssertionError(f"Frozen input changed: {name}")

    verify()
    manifest = dict(version=1, protocol_sha256=frozen.digest(OUT / "protocol.json"),
        plan_sha256=frozen.digest(OUT / "training_plan.json"),
        resume_code_sha256=frozen.digest(__file__), torch_version=str(torch.__version__),
        preserved_evaluations=sorted(p.name for p in OUT.glob("evaluation_*.json")),
        note="Session interrupted during direct64 validation. Existing weights and completed results preserved. "
             "Missing solves restart from original domains; completed resume rows are reused. "
             "Pending arms start from the prescribed parent with fresh Adam and the frozen seed/order.")
    manifest_path = OUT / "resume_20261112.json"
    if manifest_path.exists():
        previous = read(manifest_path)
        for key in ("protocol_sha256", "plan_sha256", "resume_code_sha256", "torch_version"):
            if previous[key] != manifest[key]:
                raise AssertionError(f"Resume provenance changed: {key}")
    else:
        write(manifest_path, manifest)

    data = read(frozen.DATA)
    initial, _ = frozen.instances(data)
    validation = [r for r in read(frozen.OLD / "states.json") if r["split"] == "validation"]
    records = {idx: read(OUT / "pools" / f"state_{idx:02d}.json")
               for idx in plan["shared_state_ids"]}
    order = plan["update_order"]
    if len(records) != 24 or len(order) != 240:
        raise AssertionError("Expected frozen 24 states and 240 updates")
    parent = frozen.load(frozen.PARENT).cpu().eval()
    parent.beta = 0.
    parent.config.pop("max_total_work", None)
    parent.config["decoder"] = "discrepancy"
    saved_parent = torch.load(frozen.PARENT, map_location="cpu", weights_only=True)
    inherited = saved_parent["training"]["train_instance_hashes"]
    ids = protocol["controls"]["validation_puzzles"]
    if set(inherited) & frozen.instance_hashes(data, [initial[i] for i in ids]):
        raise AssertionError("Parent validation exposure")
    state = frozen._state({**data, "samples": [dict(domains=[list(range(d))
                          for d in data["network"]["dimensions"]])]})
    exposure = sorted(set(inherited) | frozen.instance_hashes(data,
                      [initial[r["puzzle"]] for r in records.values()]))

    def evaluate(model, name, checkpoint):
        output = OUT / f"evaluation_{name}.json"
        if output.exists():
            return read(output)
        progress = OUT / f"progress_{name}"
        progress.mkdir(exist_ok=True)
        provenance = dict(checkpoint_sha256=frozen.digest(checkpoint),
                          protocol_sha256=frozen.digest(OUT / "protocol.json"),
                          resume_code_sha256=frozen.digest(__file__))
        roots_path = progress / "greedy_roots.json"
        if roots_path.exists():
            stored = read(roots_path)
            if stored["provenance"] != provenance:
                raise AssertionError("Greedy probe provenance changed")
            roots = stored["rows"]
        else:
            roots = []
            with torch.no_grad():
                for sample in validation:
                    with frozen._closed(state, sample, model.network["colors"]):
                        candidate, _, _, trace = model.rollout(state.domains, greedy=True, beta=0.)
                        roots.append(dict(puzzle=sample["puzzle"], steps=len(trace),
                            **frozen.learn.measure(model, state, candidate, objective="pruning",
                                                   known_satisfiable=True)))
            write(roots_path, dict(provenance=provenance, rows=roots))
        rows = []
        for index, puzzle in enumerate(ids, 1):
            path = progress / f"puzzle_{puzzle}.json"
            if path.exists():
                stored = read(path)
                if stored["provenance"] != provenance or stored["row"]["puzzle"] != puzzle:
                    raise AssertionError("Saved solve provenance changed")
                row = stored["row"]
            else:
                with torch.no_grad():
                    row = dict(puzzle=puzzle, **frozen.solve(state.constraint_network, initial[puzzle],
                        "trained", model, nodes=10000, seconds=120, cluster_attempts=32,
                        baseline_candidates=32, baseline_steps=8192, stop_on_noop=False))
                if row["status"] in ("invalid", "unsat"):
                    raise AssertionError("Invalid validation solution")
                write(path, dict(provenance=provenance, row=row))
            rows.append(row)
            print("solve", name, f"{index}/{len(ids)}", puzzle, row["status"],
                  row["nodes"], f"{row['solve_s']:.2f}s", flush=True)
        complete = all(row["solved"] and row["valid"] for row in rows)
        result = dict(complete=complete, nodes=sum(r["nodes"] for r in rows) if complete else None,
            branch_free=sum(r["solved"] and r["nodes"] == 1 for r in rows),
            greedy_useful=sum(r["useful"] for r in roots), greedy_states=len(roots),
            greedy_mean_reward=sum(r["reward"] for r in roots) / len(roots),
            greedy_roots=roots, rows=rows)
        write(output, result)
        print("evaluation", name, "nodes", result["nodes"], "branchfree", result["branch_free"],
              "greedy", result["greedy_useful"], "/", len(roots), flush=True)
        return result

    results = {name: read(OUT / f"evaluation_{name}.json")
               for name in ("plain", "heuristic32", "parent")}
    for arm in protocol["controls"]["arm_order"]:
        verify()
        checkpoint = OUT / f"{arm}.pt"
        if not checkpoint.exists():
            model = deepcopy(parent).train()
            torch.manual_seed(29)
            optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
            rng, history, started = Random(29), [], perf_counter()
            pool, fraction, greedy = frozen.ARMS[arm]
            for update, idx in enumerate(order, 1):
                record = records[idx]
                teachers = frozen.teachers_for(record["pools"][pool])
                with frozen._closed(state, record, model.network["colors"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = frozen.learn.teacher_loss(model, state.domains, teachers, rng,
                                                     greedy=greedy, quality_fraction=fraction)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    history.append(dict(update=update, state=idx, loss=float(loss.detach()),
                                        gradient_norm=float(norm)))
                if update % 24 == 0:
                    print("train", arm, update, "/240", history[-1]["loss"], flush=True)
            metadata = dict(method="controlled_teacher_quality_imitation", parent=str(frozen.PARENT),
                parent_sha256=frozen.digest(frozen.PARENT), train_instance_hashes=exposure,
                train_puzzles=sorted({r["puzzle"] for r in records.values()}), arm=arm, updates=240,
                seed=29, lr=.0003, fresh_optimizer=True, quality_fraction=fraction,
                greedy_imitation=greedy, protocol_sha256=frozen.digest(OUT / "protocol.json"),
                plan_sha256=frozen.digest(OUT / "training_plan.json"))
            frozen._save(checkpoint, model, history, 0., 1., training=metadata, cost_metric="pruning_fraction")
            write(OUT / f"history_{arm}.json", dict(seconds=perf_counter() - started, updates=240, rows=history))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["training"]["updates"] != 240 or len(saved["history"]) != 240:
            raise AssertionError("Incomplete checkpoint")
        model = frozen.load(checkpoint).cpu().eval()
        if model.network != parent.network or model.config != parent.config:
            raise AssertionError("Runtime configuration changed")
        results[arm] = evaluate(model, arm, checkpoint)
    summary = {name: {k: v for k, v in result.items() if k not in ("rows", "greedy_roots")}
               for name, result in results.items()}
    if not (OUT / "summary.json").exists():
        write(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
