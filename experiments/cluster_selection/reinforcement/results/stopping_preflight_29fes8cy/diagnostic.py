"""No-learning audit of archived development constructions, preserving full pruning."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
import torch
from experiments.cluster_selection.candidate_evaluation_mac import evaluate_candidate
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection.reinforcement.actor import load
from experiments.cluster_selection.reinforcement.experiment import instances
from experiments.cluster_selection.reinforcement.train import _state

started = perf_counter()
torch.set_num_threads(1)
root = Path('experiments/cluster_selection/reinforcement/results')
archive = root / 'length_diagnostic_seed29_0rmst_im'
protocol = json.loads((archive/'protocol.json').read_text())
paths = {Path(name).name: Path(name) for name in protocol['source_sha256']}
data_path, checkpoint = paths['development.json'], paths['reinforce.pt']
for path in (data_path, checkpoint):
    assert sha256(path.read_bytes()).hexdigest() == protocol['source_sha256'][str(path)]
data = json.loads(data_path.read_text())
model = load(checkpoint).cpu().eval()
base = _state({**data, 'samples': data.get('samples') or [{}]})
initial, _ = instances(data)
limits = {key: model.config[key] for key in ('max_work', 'max_size')}
limits['compact_domains'] = model.config.get('compact_domains', False)
cases = [json.loads(path.read_text()) for path in sorted(archive.glob('case_*.json'))]
successes = []
for case in cases:
    full = next((item for item in case['trials'] if item['variant']=='same_outputs'
                 and item['cores']==case['full_cores'] and item.get('removed_all', 0)), None)
    if full:
        successes.append(case)
rows = []
for case in successes:
    state = base.constraint_network.create_mac_state(initial[case['puzzle']])
    assert state.propagate() and not state.is_solved()
    before, statistics, core_keys = state.snapshot_domains(), asdict(state.statistics), tuple(state._cores)
    candidate = SummaryClusterCandidate(**case['candidate'])
    full = evaluate_candidate(state, candidate, **limits)
    assert not full.contradiction
    anchor = model.network['colors'][next(idx for kind,idx in case['trace'] if kind=='open')]
    # Check connected construction and all base legal-action masks, but deliberately
    # not forced-growth decoding, which currently masks premature finish actions.
    model.candidate_trace(candidate, before, target_color=anchor)
    remaining = list(candidate.core_keys)
    evaluations, illegal, output_missing = 1, 0, 0
    for key in list(remaining):
        subset = [item for item in remaining if item != key]
        scope = {color for item in subset for color in state._cores[item].colors}
        if not set(candidate.open_colors) <= scope:
            output_missing += 1
            continue
        trial = SummaryClusterCandidate('stopping_preflight', tuple(subset), candidate.open_colors)
        try:
            model.candidate_trace(trial, before, target_color=anchor)
        except ValueError:
            illegal += 1
            continue
        result = evaluate_candidate(state, trial, **limits)
        evaluations += 1
        if not result.contradiction and result.domains == full.domains:
            remaining = subset
    shortened = SummaryClusterCandidate('stopping_preflight_final', tuple(remaining), candidate.open_colors)
    trace = model.candidate_trace(shortened, before, target_color=anchor)
    # A reachable replay must also satisfy the decoder masks, not only actions().
    keys, opened, phase = [], [], False
    sizes = [len(before[color]) for color in model.network['colors']]
    for action in trace:
        actions = model.actions(keys, opened, phase, sizes)
        allowed = model._continuations(actions, keys, opened, phase)
        if keys and not phase and ('finish', -1) in actions:
            allowed = sorted(set(allowed) | {actions.index(('finish', -1))})
        assert action in [actions[idx] for idx in allowed]
        kind, idx = action
        if kind == 'core': keys.append(idx)
        elif kind == 'open': opened.append(idx)
        elif kind == 'finish' and not phase: phase = True
        else: break
    result = evaluate_candidate(state, shortened, **limits)
    evaluations += 1
    assert result.domains == full.domains and not result.contradiction
    assert state.snapshot_domains() == before and asdict(state.statistics) == statistics
    assert tuple(state._cores) == core_keys and not state._added_core_keys and not state._queue
    assert state.decision_level == 0
    rows.append(dict(puzzle=case['puzzle'], draw=case['draw'], greedy=case['greedy'],
        original_cores=len(candidate.core_keys), shortened_cores=len(remaining),
        outputs=list(candidate.open_colors), full_all_domains_preserved=True, reachable_if_core_finish_enabled=True,
        all_domain_values_removed=sum(len(before[c])-len(full.domains[c]) for c in before),
        original_work=full.estimate.work, shortened_work=result.estimate.work,
        omitted_illegal_candidates=illegal, omitted_missing_output_candidates=output_missing,
        contraction_evaluations=evaluations, shortened_core_keys=remaining,
        legal_construction_trace=trace))
    print(f"puzzle {case['puzzle']}, draw {case['draw']}: {len(candidate.core_keys)} -> {len(remaining)} cores, identical full pruning", flush=True)
report = dict(protocol=dict(source=str(archive), checkpoint=str(checkpoint), data=str(data_path),
    source_hashes={str(path): sha256(path.read_bytes()).hexdigest() for path in
                   (data_path,checkpoint,Path(__file__),Path('experiments/cluster_selection/reinforcement/model.py'))},
    interpretation='Retrospective existing development validation. No training, no fresh model samples, no test-set input. Fixed anchor and all original outputs; connected legal core deletion. Full 1053-variable effects preserved. Greedy order, not a global minimum.',
    training_updates=0, fresh_policy_rollouts=0),
    summary=dict(archived_constructions=len(cases), archived_successes=len(successes),
        shortened=sum(row['shortened_cores']<row['original_cores'] for row in rows),
        reachable_if_core_finish_enabled=sum(row['reachable_if_core_finish_enabled'] for row in rows),
        original_cores=sum(row['original_cores'] for row in rows),
        shortened_cores=sum(row['shortened_cores'] for row in rows),
        contraction_evaluations=sum(row['contraction_evaluations'] for row in rows),
        seconds=perf_counter()-started), cases=rows)
out = Path(mkdtemp(prefix='stopping_preflight_', dir=root))
(out/'diagnostic.py').write_bytes(Path(__file__).read_bytes())
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Results:',out)
print(json.dumps(report['summary'],indent=2))
