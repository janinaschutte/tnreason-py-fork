"""Post-hoc separation of the learned global head and co-adapted base weights."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp

import torch

from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement.experiment import instances, instance_hashes
from experiments.cluster_selection.reinforcement.train import _save

BASE = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(mkdtemp(prefix="global_parts_", dir=BASE))
    data_path, reference_path = BASE / "pilot_data.json", BASE / "baselines32_wgk4zatg/plain.json"
    parent_path = BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    trained_path = BASE / "global_seed0/final.pt"
    hashes = {str(path): digest(path) for path in (data_path, reference_path, parent_path, trained_path, Path(__file__))}
    sources = refine.source_hashes()
    data = json.loads(data_path.read_text())
    references = {row["index"]: row["residual"]["assignment"] for row in json.loads(reference_path.read_text())["rows"]}
    indices = data["validation_indices"]
    assert len(indices) == 16 and all(index in references for index in indices)
    parent, trained = load(parent_path), load(trained_path)
    assert trained.actor_config == dict(global_context=True, distance_features=False, objective_features=False)
    assert parent.network == trained.network == data["network"] and parent.beta == trained.beta == 0.
    saved = torch.load(trained_path, map_location="cpu", weights_only=True)
    exposure = set(saved["training"]["train_instance_hashes"])
    initial = instances(data)[0]
    assert not exposure & instance_hashes(data, [initial[i] for i in indices])
    base_only, head_only = deepcopy(trained), deepcopy(trained)
    with torch.no_grad():
        base_only.anchor_context[-1].weight.zero_()
        base_only.anchor_context[-1].bias.zero_()
    merged = head_only.state_dict()
    merged.update(parent.state_dict())
    head_only.load_state_dict(merged, strict=True)
    original_weights, parent_weights = trained.state_dict(), parent.state_dict()
    assert all(torch.equal(head_only.state_dict()[key], value) for key, value in parent_weights.items())
    assert all(torch.equal(base_only.state_dict()[key], value) for key, value in original_weights.items()
               if not key.startswith("anchor_context."))
    assert all(torch.equal(head_only.state_dict()[key], value) for key, value in original_weights.items()
               if key.startswith("anchor_context."))
    settings = dict(attempts=32, samples=0, seconds=30., nodes=10000, seed=100000)
    write(output / "protocol.json", dict(source_sha256=sources, input_sha256=hashes, settings=settings,
        data=str(data_path), indices=indices, canonical_config=trained.config, actor_config=trained.actor_config,
        beta=trained.beta, parent=str(parent_path), trained=str(trained_path),
        scope="Post-hoc architecture diagnostic on16reused development puzzles; no test100 or extra training",
        timing="Sequential arms on CPU0 on a shared host; separate root and plain residual MAC limits",
        interpretation="Head-only can expose co-adaptation; it is not a separately trained competing model. Both arms retain the global-head architecture and categorical renormalization."))
    for name, model in dict(base_only=base_only, head_only=head_only).items():
        path = output / f"{name}.pt"
        _save(path, model, [], 0., 1., actor_config=model.actor_config,
              training=saved["training"], ablation=dict(name=name, parent_sha256=digest(parent_path), trained_sha256=digest(trained_path)))
        before = digest(path)
        restored = load(path)
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
        write(output / f"{name}_before.json", dict(checkpoint_sha256=before, source_sha256=refine.source_hashes(),
            config=restored.config, actor_config=restored.actor_config, fingerprint_overlap=0))
        original_root, original_solve = refine_eval.propagate, refine_eval.solve
        roots, residuals = [], []

        def record_root(*args, **kwargs):
            row = original_root(*args, **kwargs)
            roots.append(row)
            return row

        def record_residual(*args, **kwargs):
            row = original_solve(*args, **kwargs)
            position = len(residuals)
            residuals.append(row)
            write(output / f"{name}_row_{position:02d}.json", dict(index=indices[position], root=roots[position], residual=row))
            print(f"{name} {position+1}/16: roots{sum(r['solved'] for r in roots)}, "
                  f"residual{row['status']}, nodes{sum(r['nodes'] for r in residuals if r['status'] in {'solved','unsat'})}", flush=True)
            return row

        refine_eval.propagate, refine_eval.solve = record_root, record_residual
        try:
            result = refine_eval.evaluate(restored, data, **settings)
        finally:
            refine_eval.propagate, refine_eval.solve = original_root, original_solve
        write(output / f"{name}.json", result)
        assert before == digest(path) and sources == refine.source_hashes()
        assert all(digest(Path(path)) == value for path, value in hashes.items())
        entries = givens = 0
        colors = data["network"]["colors"]
        for row in result["rows"]:
            assert row["residual"]["status"] == "solved"
            assignment = row["residual"]["assignment"]
            assert set(assignment) == set(colors)
            assert all(value in row["root"]["domains"][color] for color, value in references[row["index"]].items())
            assert all(assignment[color] in values for color, values in row["root"]["domains"].items())
            for core in data["network"]["cores"]:
                offset = 0
                for color, size in zip(core["colors"], core["shape"]):
                    assert assignment[colors[color]] in range(size)
                    offset = offset * size + assignment[colors[color]]
                assert core["values"][offset]
                entries += 1
            for index, digit in enumerate(data["puzzles"][row["index"]]):
                if digit not in "0.":
                    r, c = divmod(index, 9)
                    assert assignment[f"pos_{r//3}_{r%3}_{c//3}_{c%3}"] == int(digit) - 1
                    givens += 1
        write(output / f"{name}_summary.json", dict(totals=result["totals"], settings=result["settings"],
            root_statuses=dict(Counter(row["root"]["status"] for row in result["rows"])),
            assignments_valid=16, retained_references=16, original_tensor_entries_checked=entries,
            original_givens_checked=givens, checkpoint_and_sources_unchanged=True))
        print(name, result["totals"], flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
