"""Current-protocol checks of the stronger archived GPU success-replay policy."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
from tempfile import mkdtemp

import torch

from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prior_replay_helpers", BASE / "full_validation.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
CHECKPOINT = BASE.parent / "learning_diagnostics_20261113.67i11ilh/continuations/gpu_self_imitation_8051.2/policy.best.pt"


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(mkdtemp(prefix="prior_gpu_replay_", dir=BASE))
    settings = dict(seconds=30., samples=8, attempts=32, seed=100000, nodes=10000)
    phases = dict(full55=(helpers.DATA, settings), time16=(BASE / "pilot_data.json", dict(settings, attempts=0, samples=0)))
    sources = refine.source_hashes()
    hashes = {str(path): helpers.digest(path) for path in (CHECKPOINT, helpers.DATA, BASE / "pilot_data.json", Path(__file__), BASE / "full_validation.py")}
    assert hashes[str(CHECKPOINT)] == "7d4e9e57244cda6041de7a1538181432a5ca437186b2bb2ee0713f2ed2c64049"
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    exposure = set(saved["training"]["train_instance_hashes"])
    references = {row["index"]: row["residual"]["assignment"] for row in json.loads((BASE / "full55_q13qt7s1/parent.json").read_text())["rows"]}
    helpers.write(output / "protocol.json", dict(checkpoint=str(CHECKPOINT), hashes=hashes, source_sha256=sources,
        phases={name: dict(data=str(path), settings=options) for name, (path, options) in phases.items()},
        timing="CPU0 on a shared host; root and plain residual MAC have separate time limits",
        scope="Original reused development validation only; no heldout100 inputs or outcomes"))
    for phase, (data_path, options) in phases.items():
        data = json.loads(data_path.read_text())
        initial = instances(data)[0]
        selected = data["validation_indices"]
        assert len(selected) == (55 if phase == "full55" else 16)
        assert not exposure & instance_hashes(data, [initial[i] for i in selected])
        model = load(CHECKPOINT)
        helpers.write(output / f"{phase}_before.json", dict(checkpoint_sha256=helpers.digest(CHECKPOINT),
            data_sha256=helpers.digest(data_path), source_sha256=refine.source_hashes(),
            config=model.config, actor_config=model.actor_config, training_fingerprint_overlap=0))
        original_root, original_solve = refine_eval.propagate, refine_eval.solve
        roots, residuals = [], []

        def record_root(*a, **kw):
            row = original_root(*a, **kw)
            roots.append(row)
            return row

        def record_residual(*a, **kw):
            row = original_solve(*a, **kw)
            position = len(residuals)
            residuals.append(row)
            helpers.write(output / f"{phase}_row_{position:02d}.json", dict(index=selected[position], root=roots[position], residual=row))
            print(f"GPU replay {phase} {position+1}/{len(selected)}: roots {sum(r['solved'] for r in roots)}, "
                  f"residual {row['status']}, completed DFS {sum(r['nodes'] for r in residuals if r['status'] in {'solved', 'unsat'})}", flush=True)
            return row

        refine_eval.propagate, refine_eval.solve = record_root, record_residual
        try:
            result = refine_eval.evaluate(model, data, **options)
        finally:
            refine_eval.propagate, refine_eval.solve = original_root, original_solve
        assert sources == refine.source_hashes() and all(helpers.digest(Path(path)) == value for path, value in hashes.items())
        checked_entries = checked_givens = 0
        colors = data["network"]["colors"]
        for row in result["rows"]:
            assignment = row["residual"]["assignment"]
            assert row["residual"]["status"] == "solved" and set(assignment) == set(colors)
            assert all(value in row["root"]["domains"][color] for color, value in references[row["index"]].items())
            for core in data["network"]["cores"]:
                offset = 0
                for color, size in zip(core["colors"], core["shape"]):
                    assert assignment[colors[color]] in range(size)
                    offset = offset * size + assignment[colors[color]]
                assert core["values"][offset]
                checked_entries += 1
            for index, digit in enumerate(data["puzzles"][row["index"]]):
                if digit not in "0.":
                    r, c = divmod(index, 9)
                    assert assignment[f"pos_{r//3}_{r%3}_{c//3}_{c%3}"] == int(digit) - 1
                    checked_givens += 1
            assert all(assignment[color] in values for color, values in row["root"]["domains"].items())
            if row["root"]["solved"]:
                assert row["residual"]["nodes"] == 1 and all(len(values) == 1 for values in row["root"]["domains"].values())
        helpers.write(output / f"{phase}.json", result)
        helpers.write(output / f"{phase}_summary.json", dict(totals=result["totals"], settings=result["settings"],
            root_statuses=dict(Counter(row["root"]["status"] for row in result["rows"])),
            assignments_checked=len(result["rows"]), original_tensor_entries_checked=checked_entries,
            original_givens_checked=checked_givens, retained_original_references=len(result["rows"]),
            all_solutions_valid=True, source_inputs_checkpoint_unchanged=True, training_fingerprint_overlap=0))
        print(phase, result["totals"], flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
