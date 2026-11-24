# Cluster selection experiments

These experiments select and construct tensor clusters for additional pruning
after generalized arc consistency (GAC). Every application contracts the original
cores under the current domains. Temporary summaries are removed while verified
domain pruning is retained, including across backtracking.

Run module entry points from the repository root with the package installed or
the root on `PYTHONPATH`. Learning entry points additionally require PyTorch;
the regression suites require pytest.

For structural-first inference, `runtime.attach_hybrid(state, predictor,
heuristic_candidates=32, learned_candidates=32)` runs a structural batch followed
by learned construction on the resulting domains. `predictor` can be a
`ConstructionEnsemble` of compatible constructors. The two allowances add;
each cluster has its own contraction and GAC round, with temporary summaries
and retained pruning. Existing inference defaults are unchanged.

## Entry points

| Module | Purpose |
| --- | --- |
| `supervised.collect`, `supervised.train` | Collect independent candidate trials and train the supervised scorer. |
| `reinforcement.learn`, `reinforcement.qlearn` | Train a constructor by teacher imitation, REINFORCE, or Q-learning. See the [workflow documentation](reinforcement/README.md). |
| `reinforcement.experiment` | Compare construction policies with plain MAC and structural controls. |
| `library` | Collect successful heuristic applications from explicitly selected training puzzles into a fixed cluster library. |
| `library_evaluation` | Evaluate fixed libraries independently on recorded states with bounded contractions. |
| `mask_data`, `mask_train` | Replay successful applications into a mask corpus and train a model that predicts core and output masks. |

Each module supports `--help`. Small seed datasets are retained in [data](data/).
Generated runs, collected corpora, logs, and model checkpoints are local artifacts
and are ignored by Git. Historical paths in the reinforcement documentation
refer to that separate archive; reproducing those runs requires their recorded
inputs and checkpoints.

## Fixed libraries and mask training

Supply `DATA.json` with the raw network, puzzle domains, and explicit disjoint
`train_indices` and validation/test indices. Collection and corpus construction
use training rows only; reserved rows must not overlap their initial domains.
Each output directory below must be new. Collection and evaluation also support
`--resume` for their own incomplete runs.

```bash
python -m experiments.cluster_selection.library DATA.json \
  --output experiments/cluster_selection/results/collection \
  --puzzles 1000 --allowance 64 --workers 4 --record-failures

python -m experiments.cluster_selection.mask_data \
  experiments/cluster_selection/results/collection --data DATA.json \
  --output experiments/cluster_selection/results/mask-corpus --workers 4

python -m experiments.cluster_selection.mask_train \
  experiments/cluster_selection/results/mask-corpus/corpus.npz --data DATA.json \
  --output experiments/cluster_selection/results/mask-policy --epochs 30
```

The collection publishes `library.json` plus per-puzzle records. The mask corpus
replays recorded applications against the original network instead of reusing
cached tensor values. Training can validate on the supplied dataset's explicit
validation split; it does not evaluate the test split.

## Optional rule-inspired proposals

`rule_heuristic` adds overlap and small Hall subset proposals, inspired by Sudoku
locked candidates, pairs, triples and quads. It recognizes categorical/Boolean
channels from their actual tensor supports, including value permutations and
reversed Boolean polarity. Selection uses shared variables and relations;
every deduction comes from contracting the selected original cores under current
domains and running GAC. Unrecognized relations still receive cycle proposals.

```python
from experiments.cluster_selection.rule_heuristic import attach_heuristic

attach_heuristic(state, rule_inspired=True, max_candidates=64, max_steps=16384,
                 max_cores=64, max_open_colors=6, max_work=1_000_000,
                 max_size=100_000, compact_domains=True)
```

The option defaults to `False`, which delegates to the existing runtime policy.
The enabled policy regenerates proposals after each useful contraction and
caches only no-op inputs; summaries are temporary and pruning is reversible.
For a controlled comparison of proposal families under that same refreshed
schedule, attach `rule_heuristic.HeuristicPolicy(state, rule_inspired=enabled,
**limits)` as `state.on_closure` in both arms.

Enabled generation interleaves cycle and rule proposals within the requested
list size and splits traversal steps between them. Support recognition has its
own bounded preparation work. The result is a budgeted mixture, so a capped
list need not contain every legacy proposal. Core and tensor limits still
apply; the legacy ten-core default is too small for an eighteen-core triple.
Recreate the generator after changing tensor values or network structure.

## Regression checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  experiments/cluster_selection/tests experiments/constraint_networks/tests \
  demonstrations/sudoku/inferenceClusters/tests
```
