"""On-policy constructive RL with explicit state sampling and resumable updates.

This is a separate training entry point: existing experiments keep their source
and checkpoint formats. Contraction and GAC, including rollback, are reused.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite, sqrt
import os
from pathlib import Path
import platform
import random
import signal
from time import perf_counter

import torch

from .actor import Policy, load
from .experiment import instance_hashes, instances
from .learn import measure, training_states
from .train import _closed, _save
from .updates import advantages, update


def digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = []
    for folder in (root / "tnreason", root / "experiments/cluster_selection", root / "experiments/constraint_networks"):
        for directory, subdirs, files in os.walk(folder):
            subdirs[:] = sorted(set(subdirs) - {"tests", "results", "__pycache__"})
            paths.extend(Path(directory) / name for name in files if name.endswith(".py"))
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def atomic_save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def initialize(data, *, seed=0, initial=None, global_context=False, distance_features=False,
               objective_features=False, learned_stopping=None,
               max_cores=None, max_open_colors=None, max_work=None, max_size=None):
    """Add zero-initialized adapters without changing the parent's predictions."""
    torch.manual_seed(seed)
    if initial is None:
        config = dict(width=32, rounds=2, contextual=True,
            context_rounds=2, exploration=0., target_first=True, cyclic_only=True,
            compact_domains=True, min_active=2, max_cores=64, max_open_colors=6,
            max_work=1_000_000, max_size=100_000, heuristic_candidates=0,
            decoder="sampling_grow")
        parent = None
    else:
        parent = load(initial)
        if (parent.network != data["network"]
                or parent.reward_colors != data.get("reward_colors", data["network"]["colors"])):
            raise ValueError("Initial checkpoint and training network/reward variables differ")
        config = dict(parent.config)
    config.update({key: value for key, value in dict(max_cores=max_cores,
        max_open_colors=max_open_colors, max_work=max_work, max_size=max_size).items() if value is not None})
    flags = {key: value or bool(parent and parent.actor_config.get(key)) for key, value in
             dict(global_context=global_context, distance_features=distance_features,
                  objective_features=objective_features).items()}
    flags["learned_stopping"] = (bool(parent and parent.actor_config.get("learned_stopping"))
                                 if learned_stopping is None else learned_stopping)
    model = Policy(data["network"], **config, **flags)
    if parent:
        missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
        added = tuple(name + "." for name in ("anchor_context", "core_distance", "objective_role")
                      if not hasattr(parent, name) and hasattr(model, name))
        if unexpected or any(not key.startswith(added) for key in missing):
            raise ValueError("Unexpected initial-checkpoint parameter mismatch")
    model.reward_colors = data.get("reward_colors", data["network"]["colors"])
    model.beta = 0.
    model.config["exploration"] = 0.
    model.config["heuristic_candidates"] = 0
    # Adapter construction consumes random numbers; paired arms sample from
    # the same RNG state before their first parameter update.
    torch.manual_seed(seed)
    return model.train()


def collect(model, state, sample, *, rollouts, solve_bonus=0., retain_graphs=False, size_penalty=0.):
    """Independent completed constructions under one unchanged closed state."""
    if not isfinite(size_penalty) or not 0 <= size_penalty < 1:
        raise ValueError("Size penalty must be finite and in [0, 1)")
    colors = model.network["colors"]
    with _closed(state, sample, colors), torch.set_grad_enabled(retain_graphs):
        domains = state.snapshot_domains()
        items, outcomes, signatures = [], [], []
        for _ in range(rollouts):
            candidate, logp, entropy, trace = model.rollout(domains, beta=0.)
            metrics = measure(model, state, candidate, objective="pruning", record_domains=True,
                              known_satisfiable=sample.get("known_satisfiable", False))
            count = 0 if candidate is None else len(candidate.core_keys)
            reward = metrics["reward"] * (1 - size_penalty * count / model.config["max_cores"])
            reward += solve_bonus * metrics["solved"]
            items.append(dict(trace=trace, old_logp=float(logp.detach()), entropy=float(entropy.detach()), reward=reward,
                              core_keys=() if candidate is None else candidate.core_keys,
                              open_colors=() if candidate is None else candidate.open_colors))
            if retain_graphs:
                items[-1].update(logp_tensor=logp, entropy_tensor=entropy)
            outcomes.append(metrics)
            signatures.append(None if candidate is None else (candidate.core_keys, candidate.open_colors))
    return dict(domains=domains, rollouts=items), outcomes, len(set(signatures))


def fit(data, *, output, initial=None, resume=None, trials=100_000, seed=0, device="cpu",
        groups=4, rollouts=4, lr=1e-4, normalization="group", root_fraction=.75,
        context_capacity=3, global_context=False, distance_features=False,
        objective_features=False, retain_graphs=False, learned_stopping=None,
        size_penalty=0., replay_weight=0., replay_samples=2, replay_capacity=256,
        compact_probes=64, compact_every=8, split_replay=False,
        max_cores=None, max_open_colors=None, max_work=None, max_size=None,
        epochs=1, clip_ratio=None, entropy_weight=.001, target_kl=.02, solve_bonus=0.,
        validation_every=4096, validation_attempts=0, validation_samples=8,
        validation_seconds=30., validation_limit=0, checkpoint_every=1024):
    """Train on independent state groups; validation never supplies transitions.

    Trials count *fresh* completed policy constructions, not rescoring passes.
    Contexts are bounded and come only from verified pruning by this policy.
    A selected sample receives its own immediate reward, never another sample's
    reward or credit for a teacher transition. No eager teacher search is run.
    """
    if (min(trials, groups, context_capacity, validation_every, checkpoint_every) < 1 or rollouts < 2
            or trials % (groups * rollouts) or not 0 <= root_fraction <= 1
            or normalization not in {"group", "rms", "none"}
            or not all(isfinite(value) and value >= 0 for value in (lr, entropy_weight, solve_bonus))
            or not isfinite(size_penalty) or not 0 <= size_penalty < 1
            or not isfinite(replay_weight) or replay_weight < 0
            or min(replay_samples, replay_capacity, compact_probes, compact_every) < 1
            or lr == 0 or epochs < 1 or epochs > 1 and clip_ratio is None):
        raise ValueError("Invalid training settings; trials must be divisible by groups * rollouts")
    if not data.get("train_indices") or not data.get("validation_indices"):
        raise ValueError("Provide explicit nonempty train_indices and validation_indices")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (replay_samples, replay_capacity, compact_probes, compact_every)):
        raise ValueError("Replay sizes and compact probe/refresh limits must be positive integers")
    if not isinstance(split_replay, bool) or split_replay and not replay_weight:
        raise ValueError("Split replay requires a boolean option and a positive replay weight")
    if not isinstance(validation_limit, int) or isinstance(validation_limit, bool) or validation_limit < 0:
        raise ValueError("Validation limit must be a nonnegative integer; 0 uses all validation puzzles")
    if retain_graphs and (epochs != 1 or clip_ratio is not None):
        raise ValueError("Retained graphs require one vanilla REINFORCE epoch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Python environment")
    if initial and resume:
        raise ValueError("Choose an initial policy or an exact training resume")
    output = Path(output)
    settings = {key: value for key, value in locals().copy().items()
                if key not in {"data", "output", "resume", "initial", "device", "trials"}}
    settings["branching"] = "MRV, maximum incident-core degree, variable name; fresh residual state"
    signature, sources = digest(data), source_hashes()
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if saved and (saved["data_digest"] != signature or saved["settings"] != settings):
        raise ValueError("Resume data/settings mismatch; use --initial for a different experiment")
    if saved and saved["source_hashes"] != sources:
        raise ValueError("Training source changed since resume; use --initial to start a new experiment")
    if saved and (saved["completed"] > trials or saved["completed"] % (groups * rollouts)):
        raise ValueError("Resume trials must extend a checkpoint at a full batch boundary")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(dict(version=1, data_digest=signature,
        settings=settings, trials=trials, device=device, initial=str(initial) if initial else None,
        initial_sha256=sha256(Path(initial).read_bytes()).hexdigest() if initial else None,
        resume=str(resume) if resume else None,
        resume_sha256=sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
        source_hashes=sources, torch_version=str(torch.__version__), python_version=platform.python_version(),
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None), indent=2) + "\n")
    base, state, pools, _ = training_states(data, progress_every=1000)
    roots = {key: item for key, pool in pools.items() for item in pool if item.get("root")}
    initial_domains, _ = instances(data)
    exposure = instance_hashes(data, [initial_domains[key] for key in roots])
    validation_hashes = instance_hashes(data, [initial_domains[key] for key in data["validation_indices"]])
    parent = saved if saved else torch.load(initial, map_location="cpu", weights_only=True) if initial else None
    if parent:
        inherited = parent.get("training", {}).get("train_instance_hashes")
        if inherited is None or set(inherited) & validation_hashes:
            raise ValueError("Parent lacks training fingerprints or has trained on validation instances")
        exposure.update(inherited)
    parent_path = Path(resume or initial) if resume or initial else None
    training = dict(train_instance_hashes=sorted(exposure), train_puzzles=sorted(roots),
        data_digest=signature, source_hashes=sources, parent=None if parent_path is None else
        dict(path=str(parent_path), sha256=sha256(parent_path.read_bytes()).hexdigest()))
    contexts = {key: [] for key in roots}
    rng, pending = random.Random(seed), []
    replay, replay_rng = {}, random.Random(seed + 2_000_000)
    if saved:
        model = Policy(base["network"], **saved["config"], **saved["actor_config"])
        model.reward_colors, model.beta = saved["reward_colors"], 0.
        model.to(device).train()
    else:
        model = initialize(base, seed=seed, initial=initial,
                           global_context=global_context, distance_features=distance_features,
                           objective_features=objective_features, learned_stopping=learned_stopping,
                           max_cores=max_cores,
                           max_open_colors=max_open_colors, max_work=max_work, max_size=max_size).to(device)
    if (replay_weight or size_penalty) and not model.actor_config.get("learned_stopping"):
        raise ValueError("Compact replay and size penalties require --learned-stopping")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    completed, batches, scale, best, history = 0, 0, .05, {}, []
    if saved:
        model.load_state_dict(saved["weights"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed, batches, scale, best = (saved[key] for key in ("completed", "batches", "scale", "best"))
        contexts, pending, history = saved["contexts"], saved["pending"], saved["history"]
        replay = saved.get("replay", {})
        if "replay_rng" in saved:
            replay_rng.setstate(saved["replay_rng"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] and device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    stop, previous_handlers = [], {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, lambda number, frame: stop.append(number))

    def checkpoint():
        atomic_save(output / "resume.pt", dict(version=1, data_digest=signature, settings=settings,
            weights=model.state_dict(), config=model.config, actor_config=model.actor_config,
            reward_colors=model.reward_colors, training=training, source_hashes=sources,
            optimizer=optimizer.state_dict(), completed=completed, batches=batches, scale=scale,
            best=best, history=history, contexts=contexts, pending=pending,
            replay=replay, replay_rng=replay_rng.getstate(),
            rng=rng.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else []))

    def snapshot(label):
        path = output / f"{label}.pt"
        _save(path, model, [], 0., 1., cost_metric="domain_entropy_pruning",
              actor_config=model.actor_config, fresh_trials=completed, training_recipe=(
                  "refine_compact_v1" if model.actor_config.get("learned_stopping") else "refine_v2"), training=training,
              settings=settings, seed=seed)
        return path

    def validate():
        from .refine_eval import evaluate
        path = snapshot(f"policy_{completed:09d}")
        indices = list(data["validation_indices"])
        if validation_limit:
            random.Random(100_000).shuffle(indices)
            indices = indices[:validation_limit]
        print(f"Validating {completed} fresh trials on CPU: {len(indices)} puzzles", flush=True)
        try:
            result = evaluate(model, data, indices=indices, attempts=validation_attempts, samples=validation_samples,
                seconds=validation_seconds, seed=seed + 100_000, progress=True, stop_requested=lambda: bool(stop))
        except InterruptedError:
            if not stop:
                raise
            print("Validation stopped; preserving the current training checkpoint", flush=True)
            return
        totals = result["totals"]
        result.update(fresh_trials=completed, checkpoint=path.name)
        (output / f"validation_{completed:09d}.json").write_text(json.dumps(result, indent=2) + "\n")
        dfs = totals["dfs_nodes"] if totals["dfs_nodes"] is not None else 10**18
        resolved = totals.get("root_resolved", totals["root_solved"])
        scores = dict(root=(-resolved, -totals["dfs_complete"], dfs),
                      dfs=(-totals["dfs_complete"], dfs, -resolved))
        for key, score in scores.items():
            if key not in best or score < tuple(best[key]["score"]):
                best[key] = dict(score=score, checkpoint=str(path.resolve()), trials=completed)
        (output / "best.json").write_text(json.dumps(best, indent=2) + "\n")
        print(f"validation at {completed} trials: resolved roots {resolved}/{totals['instances']} "
              f"({totals['root_solved']} SAT), "
              f"DFS {totals['dfs_nodes']} ({totals['dfs_complete']} complete), "
              f"sampled useful {totals['sampled_useful']}/{totals['sampled_trials']}", flush=True)

    try:
        print(f"Refined constructive RL on {device}: {normalization} advantages, "
              f"{groups} states x {rollouts} samples, {root_fraction:.0%} root sampling; no structural teacher search", flush=True)
        if model.actor_config.get("learned_stopping"):
            print(f"Learned stopping; relative size penalty {size_penalty:g}; compact replay weight {replay_weight:g}",
                  flush=True)
        if device.startswith("cuda"):
            print(f"GPU: {torch.cuda.get_device_name(device)}; PyTorch {torch.__version__}; "
                  f"CUDA {torch.version.cuda}; validation on CPU", flush=True)
        checkpoint()
        validate()
        checkpoint()
        began, last_report = perf_counter(), perf_counter()
        reported = Counter()
        while completed < trials and not stop:
            started, batch, additions = perf_counter(), [], []
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            counts, rewards, unique = Counter(), [], []
            for _ in range(min(groups, (trials - completed) // rollouts)):
                if not pending:
                    pending = list(roots)
                    rng.shuffle(pending)
                puzzle = pending.pop()
                root = not contexts[puzzle] or rng.random() < root_fraction
                sample = roots[puzzle] if root else rng.choice(contexts[puzzle])
                group, outcomes, distinct = collect(model, state, sample, rollouts=rollouts,
                    solve_bonus=solve_bonus, retain_graphs=retain_graphs, size_penalty=size_penalty)
                values = [item["reward"] for item in group["rollouts"]]
                for item, advantage in zip(group["rollouts"], advantages(values, normalization, scale)):
                    item["advantage"] = advantage
                batch.append(group)
                rewards.extend(values)
                unique.append(distinct)
                label = "root" if root else "context"
                counts[label + "_trials"] += rollouts
                counts[label + "_useful"] += sum(item["useful"] for item in outcomes)
                counts[label + "_solved"] += sum(item["solved"] for item in outcomes)
                counts["rejected"] += sum(item["rejected"] for item in outcomes)
                counts["work"] += sum(item["work"] for item in outcomes)
                counts["pruning_reward_sum"] += sum(item["reward"] for item in outcomes)
                counts["useful_cores"] += sum(len(item["core_keys"]) for item, outcome in
                                              zip(group["rollouts"], outcomes) if outcome["useful"])
                eligible = [item for item in outcomes if item["useful"] and not item["solved"]
                            and not item["contradictions"]]
                if eligible:
                    chosen = max(eligible, key=lambda item: item["reward"])
                    additions.append((puzzle, dict(domains=tuple(tuple(chosen["domains"][color]) for color in
                        model.network["colors"]), root=False, known_satisfiable=sample.get("known_satisfiable", False))))
            if replay_weight and batches % compact_every == 0:
                from .compact import compact_success
                from ..candidate_evaluation_tnreason import SummaryClusterCandidate
                # One bounded simplification per refresh. Prefer a fresh verified
                # success; a growth-only proposal can bootstrap stopping when all
                # early-stopped samples fail. It supplies supervised data only.
                choices = [(group, item) for group in batch for item in group["rollouts"]
                           if item["core_keys"] and item["reward"] > 0]
                if choices:
                    source, selected = max(choices, key=lambda pair: pair[1]["reward"])
                    candidate = SummaryClusterCandidate("compact_fresh", selected["core_keys"], selected["open_colors"])
                else:
                    source, candidate = batch[0], None
                source_sample = dict(domains=tuple(source["domains"][color] for color in model.network["colors"]))
                with _closed(state, source_sample, model.network["colors"]), torch.no_grad():
                    if candidate is None:
                        previous = model.actor_config["learned_stopping"]
                        try:
                            model.actor_config["learned_stopping"] = False
                            candidate, _, _, _ = model.rollout(state.domains, beta=0.)
                            counts["growth_teacher_rollouts"] += 1
                        finally:
                            model.actor_config["learned_stopping"] = previous
                    if candidate is not None:
                        compact = compact_success(model, state, candidate,
                            max_probes=compact_probes, split=split_replay)
                        counts["compact_attempts"] += 1
                        counts["compact_probes"] += compact.probes
                        counts["compact_rejected"] += compact.rejected
                        counts["compact_seconds"] += compact.cost_seconds
                        if compact.examples:
                            counts["compact_original_cores"] += compact.original_cores
                            counts["compact_final_cores"] += compact.final_cores
                        for example in compact.examples:
                            key = digest((example.domains, example.trace))
                            replay[key] = asdict(example)
                            counts["compact_examples"] += 1
                while len(replay) > replay_capacity:
                    replay.pop(next(iter(replay)))
            replay_batch = replay_rng.sample(list(replay.values()), min(replay_samples, len(replay))) if replay_weight else ()
            metrics = update(model, batch, optimizer, epochs=epochs, clip_ratio=clip_ratio,
                entropy_weight=entropy_weight, target_kl=target_kl, replay=replay_batch, replay_weight=replay_weight)
            for puzzle, sample in additions:
                if sample not in contexts[puzzle]:
                    contexts[puzzle].append(sample)
                    contexts[puzzle] = contexts[puzzle][-context_capacity:]
            scale = .95 * scale + .05 * sqrt(sum(value * value for value in rewards) / len(rewards))
            completed += len(rewards)
            batches += 1
            record = dict(batch=batches, fresh_trials=completed, reward=sum(rewards) / len(rewards),
                          unique=sum(unique) / len(unique), seconds=perf_counter() - started,
                          pruning_reward=counts["pruning_reward_sum"] / len(rewards),
                          replay_buffer=len(replay),
                          mean_useful_cores=counts["useful_cores"] / max(1, counts["root_useful"] + counts["context_useful"]),
                          sampled_entropy=sum(item["entropy"] for group in batch for item in group["rollouts"]) / len(rewards),
                          mean_cores=sum(sum(kind == "core" for kind, _ in item["trace"])
                              for group in batch for item in group["rollouts"]) / len(rewards),
                          mean_outputs=sum(sum(kind == "open" for kind, _ in item["trace"])
                              for group in batch for item in group["rollouts"]) / len(rewards),
                          cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
                          lr=optimizer.param_groups[0]["lr"], **counts, **metrics)
            history.append(record)
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            reported.update(counts)
            if batches % 16 == 0 or perf_counter() - last_report >= 45 or completed == trials:
                print(f"batch {batches}, trials {completed}/{trials}: window useful "
                      f"root {reported['root_useful']}/{reported['root_trials']}, "
                      f"context {reported['context_useful']}/{reported['context_trials']}; "
                      f"last batch reward {record['reward']:.4g}, active groups {metrics['active_groups']}/{len(batch)}, "
                      f"cores {record['mean_cores']:.1f} ({record['mean_useful_cores']:.1f} useful), "
                      f"replay {len(replay)}, gradient {metrics['grad_norm']:.3g}, elapsed {perf_counter() - began:.0f}s", flush=True)
                last_report = perf_counter()
                reported.clear()
            if completed % validation_every < len(rewards) or completed == trials:
                checkpoint()
                if stop:
                    break
                validate()
                checkpoint()
            elif completed % checkpoint_every < len(rewards):
                checkpoint()
        checkpoint()
        snapshot("final")
        return dict(trials=completed, batches=batches, best=best, interrupted=bool(stop))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory; never overwrite an earlier run")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--resume", type=Path, help="Exact optimizer/RNG state; write into a new output directory")
    parser.add_argument("--device", default="cpu")
    for key, default in dict(trials=100_000, seed=0, groups=4, rollouts=4, lr=1e-4,
            root_fraction=.75, context_capacity=3, epochs=1, entropy_weight=.001,
            target_kl=.02, solve_bonus=0., validation_every=4096, validation_attempts=0,
            validation_samples=8, validation_seconds=30., checkpoint_every=1024,
            validation_limit=0,
            size_penalty=0., replay_weight=0., replay_samples=2, replay_capacity=256,
            compact_probes=64, compact_every=8).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--normalization", choices=("group", "rms", "none"), default="group")
    parser.add_argument("--clip-ratio", type=float)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--distance-features", action="store_true")
    parser.add_argument("--objective-features", action="store_true")
    parser.add_argument("--retain-graphs", action="store_true", help="Avoid a second forward pass; needs more memory")
    parser.add_argument("--learned-stopping", action=argparse.BooleanOptionalAction, default=None,
                        help="Learn when to finish cores and outputs; otherwise inherit the parent")
    parser.add_argument("--split-replay", action="store_true", help="Try verified sequences of smaller useful clusters")
    for key in ("max_cores", "max_open_colors", "max_work", "max_size"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, help="Override this computational bound")
    args = vars(parser.parse_args())
    data = json.loads(args.pop("data").read_text())
    torch.set_num_threads(1)
    print(json.dumps(fit(data, **args)), flush=True)


if __name__ == "__main__":
    main()
