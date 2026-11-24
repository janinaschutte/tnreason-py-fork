# Constructive reinforcement learning, revised

The entry point is `python -m experiments.cluster_selection.reinforcement.refine`.
It reuses `ConstraintNetwork`, the existing tensor-support encoder, legal
construction actions, domain-conditioned contractions, GAC and rollback. The
supervised learners and the source used by the ongoing benchmark are unchanged.

## What is learned

One action sequence chooses an anchor, adds connected original cores, chooses
open variables and finishes a cluster. The growth decoder retains the ability
to construct large clusters: the current Sudoku limits are 64 cores, 6 outputs,
1,000,000 estimated operations and 100,000 intermediate tensor entries. These
are computational limits, not Sudoku rules. A completed cluster is evaluated
under the input domains; its summary is temporary and only its pruning survives.

The current `sampling_grow` decoder keeps adding an affordable core whenever
one is legal, then keeps adding legal outputs before finishing. It learns
which cores and outputs to choose; free early stopping is masked by this
decoder. The NumPy executor also limits the union of selected core scopes to
52 distinct variables, including variables with singleton domains. This
backend restriction is checked before domain compaction and is separate from
the core-count, work and tensor-size limits.

Several independent constructions are sampled under the *same* closed domains
before updating the policy. Every reward is measured by contraction and GAC,
then rolled back. A construction receives no reward from another construction.
This tests complete core-and-output constructions, including unsuccessful ones;
there is no fixed candidate library.

For domains D, let H(D) be the weighted sum of log domain sizes. Primary reward
variables have weight 1; other variables together receive comparable weight.
The default reward is the existing fraction of remaining entropy removed:

    r_i = [H(D) - H(GAC(D, cluster_i))] / H(D).

A contradiction has reward 1. No pruning has reward 0. A newly solved state
already has reward 1. `--solve-bonus` optionally adds an extra bonus *after*
the pruning reward is calculated; it defaults to zero.

The default leave-one-out advantage is normalized within an input state:

    A_i = [r_i - mean(r_j for j != i)] / std(r_1, ..., r_K).
    loss = -mean(A_i * sum_t log pi(a_it | D, a_i,<t)) - eta * entropy.

Identical rewards give zero advantage and no optimizer step for that group.
The sequence log probability is summed over its decisions. Dividing it by
cluster length would change the objective. Entropy is averaged over decisions.

Within-state normalization makes small verified pruning on a difficult state
matter in optimization. It changes the weighting of states; it is **not an
unbiased estimator of the old expected-reward objective**. In particular, one
positive sample and otherwise zeros have the same normalized advantages whether
the positive reward is small or 1. `--normalization rms` and `none` provide
controls that retain reward magnitudes across states. The RMS control uses a
running reward scale from previous batches: conditional on that scale, the
leave-one-out gradient estimates the expected pruning reward up to a common
factor. Group normalization changes that objective's state weighting. Neither
scaling has demonstrated a reliable Sudoku advantage in these pilots. The RMS
control is not an exact reproduction of the historical run.

## Data and optimization

- Supply explicit, disjoint `train_indices` and `validation_indices`. Initial
  domain fingerprints are checked. A parent checkpoint must carry its training
  fingerprints; inherited training exposure is excluded from validation.
- Roots are sampled by cycling through shuffled puzzles. `--root-fraction`
  controls how often the root is chosen when derived contexts exist. Otherwise
  the root is used. The actual root/context trial counts are logged.
- Derived contexts come only from verified, nonterminal pruning by fresh policy
  samples. Each puzzle has a bounded context pool. Selecting the best measured
  outcome supplies a curriculum state, not action credit for other samples.
- No teacher proposals, failed-value probes or mutations are generated. The
  historical pure-RL run still paid for these even with imitation weight zero.
- Adam updates accumulate gradients across several input states, with norm
  clipping. By default, sampling uses no gradient graph and re-scoring retains
  only one trace's graph at a time. `--retain-graphs` keeps the sampled graphs,
  avoiding a second forward pass at the cost of higher peak memory. It supports
  one vanilla update only. The paired generic CPU test gives identical gradients,
  weights and Adam moments; the Sudoku pilot also shows small floating-point
  differences between modes, so bitwise equality is not a universal guarantee.
  Collection and re-scoring use the same
  growth masks and no epsilon mixture. There is no dropout in the policy.
- One optimization epoch is ordinary on-policy REINFORCE. Repeated epochs require
  `--clip-ratio`; this uses a clipped **whole-sequence** importance ratio. The
  sampled KL guard can reject a later pass. It does not undo a preceding step
  or guarantee a bound on the policy's true KL divergence. Before the first
  update, sampled and current policies agree, so the guard is approximately
  zero. With the default single epoch it cannot limit that first update.

Optional `--global-context` and `--distance-features` add small residual heads:
global network/domain context for anchor selection, and generic selected-path
distances for core extensions. Their final layers start at zero, preserving a
parent's initial predictions. They are ablations, not assumed improvements.
The raw encoder and contraction semantics remain unchanged.

`--objective-features` adds the reward's log variable weights to the variable
embeddings through a zero-initialized linear map. This tells the actor which
pruning the objective values most, using 32 additional parameters at width 32.
It encodes no variable indices or problem-specific rules. A generic witness
shows why this matters: two isomorphic components receive identical action
scores without this input, although pruning them earns different rewards.
The feature is optional; the witness alone does not establish a Sudoku gain.

`--max-cores`, `--max-open-colors`, `--max-work` and `--max-size` can override
computational bounds. Changing a bound can change the parent's predictions
even before training, because the action masks and normalized features change.

## Validation and files

Validation reports independent one-step usefulness, actual root propagation,
and plain MAC search from the retained root domains. Root propagation and
remaining DFS have separate wall limits. A root solve counts as one DFS node.
Censored DFS totals are `null`, rather than being treated as complete results.
One-step sampling and model setup are outside the root timer.
`root_solved` counts satisfying assignments; `root_resolved` also counts proved
contradictions. A root contradiction needs no residual DFS and counts as one
completed search node. Sudoku's known satisfiable inputs must never contradict.
`baseline_resolved` records instances already resolved by ordinary GAC, before
cluster inference.

Residual search starts a fresh MAC state from the retained domains. It uses
minimum domain size, then maximum incident-core degree, then variable name to
choose a branching variable. It does not retain a pruning-age history from the
root episode. The full-MAC command below uses the same branching rule but keeps
cluster inference active throughout DFS.

The default `--validation-attempts 0` runs learned root propagation until its
time limit or an ordinary runtime stopping condition, without an aggregate
cluster/work allowance. Each individual cluster must still fit the policy's
contraction bounds. Proposal generation restarts after pruning, under the new
domains. A positive attempt count reproduces a fixed-attempt comparison.
The pilot explicitly uses 32. The structural-batch baseline and progressive
time-limited heuristic are different baselines and must be labelled separately;
the baseline helper rejects zero attempts instead of silently substituting a
different heuristic.

Each run writes into a **new directory**:

- `protocol.json`: data/source hashes, initialization hash and settings.
- `training.jsonl`: fresh trials, root/context usefulness, reward, diversity,
  construction sizes, entropy, gradients, clipping and actual optimizer steps.
- `policy_*.pt`, `validation_*.json`: immutable evaluated snapshots and results.
- `best.json`: separate pointers for best root solving and best completed DFS.
- `final.pt`: final policy, which need not be the best policy.
- `resume.pt`: rolling optimizer, sampling RNG, context pool and puzzle queue.

Console usefulness counts aggregate separately over root and context trials
since the preceding progress line. The JSONL retains one record per batch.

Checkpoints are written before a potentially long validation. SIGTERM/SIGINT
request a stop at a batch boundary or a validation sample/phase boundary.
A root or residual solve already in progress can continue to its own wall
limit before that boundary. The initial rolling checkpoint precedes even the
first validation. Abrupt worker loss can lose work since the last rolling
checkpoint. `--checkpoint-every` controls that interval.

Resume with the same data and settings, `--resume old_run/resume.pt`, a larger
`--trials` total and a **new** output directory. Do not also pass `--initial`.
Trial totals must be multiples of `groups * rollouts`. Source/settings changes
are rejected for exact continuation; use `--initial old_run/final.pt` for a new
experiment. CPU continuation is tested for bitwise equality of weights, Adam
moments, RNG and context selection. Changing hardware/software or CPU/CUDA can
change numerical results even when RNG state is restored.

Load policies with `reinforcement.actor.load`, especially when residual heads
are enabled, and pass the model object to `runtime.attach` or `refine_eval`.
The frozen legacy path loader does not understand extra residual-head weights.

## Evidence and limits

Pilot data and results are under
`results/refinement_20261123.hrs4eb3q`. The pilot uses 32 training puzzles and
16 randomly selected, GAC-stalled development-validation puzzles from the
existing 10,000/55 split. The repeatedly evaluated test100 set is not used for
recipe selection. The split manifest records original indices and hashes.

A separate generic capacity check uses equality constraints, one forbidden
pair, and irrelevant constraints. It checks all satisfying assignments before
accepting pruning. This can establish that the training machinery learns useful
constructions; it does not establish Sudoku generalization.

The completed small Sudoku pilots give mixed results. A final serial,
interleaved CPU comparison gives the following results at 30 seconds per root:
the untrained growth constructor solves 12/16 development puzzles and leaves
48 plain residual DFS nodes; the previous GPU replay model solves 11/16 and
leaves 48; the new global-context + RMS pilot solves the same 12 roots as the
untrained model and leaves 36 nodes. The neural methods take about 204--208
seconds in total and try almost identical numbers of proposals. Eleven of the
twelve saved nodes come from one puzzle; the paired interval includes zero.
This is a reproducible development-set DFS improvement, not an established
root-solving or generalization advantage. The untrained baseline uses random
network weights with the same strong growth masks, not arbitrary core subsets.
See the [serial comparison](results/refinement_20261123.hrs4eb3q/serial_time30_c8uq_f33/TABLES.txt)
and [audit](results/refinement_20261123.hrs4eb3q/AUDIT.md).

Fixed attempts can give a different conclusion: the same new RMS checkpoint
has only 6/16 roots and 149 residual nodes at 32 attempts. Actual full MAC,
which uses clusters again inside DFS, is another procedure: with 32 attempts per
closure and a 300-second whole-solve limit, the new RMS pilot takes 40 nodes
and solves 6/16 roots. The RF parent takes 32 nodes and the previous GPU replay
model 36; both solve 7 roots. The untrained control takes 45 nodes and solves
5 roots. All searches complete. This negative result stays
separate from the time-only root advantage. See the
[full-MAC comparison](results/refinement_20261123.hrs4eb3q/fullmac_final_3297pi_m/TABLE.txt)
for limits and timing provenance.

A stronger historical GPU success-replay checkpoint also matters when choosing
an initialization: its SHA starts `7d4e9e57`, and it reaches 33/55 roots with
147 residual nodes under the fixed-32 protocol, versus 30/55 and 157 for the
RF parent used in the first pilots. This is a different checkpoint from the
small CPU replay continuation used by the ongoing benchmark. Neither these
reused development comparisons nor the small pilots are fresh test evidence.
The 50% root/context continuation reaches 12/16 roots and 47 residual nodes at
30 seconds, versus the untrained constructor's 12/16 and 48; this one-seed,
one-node difference needs replication. For perspective, the 51-family
Sudoku-specific expert reaches 14/16 roots and 32 residual nodes, or 52/55 and
74, under its larger cluster bounds. It is a separate reference, not part of
the generic learning algorithm.

## Run and deploy

The shared-filesystem HTCondor template is `refine_gpu.sub`, using Bernoulli's
`alphareason` Python. It submits **six experimental continuations**: RMS and
group normalization, each with seeds 0, 1 and 2, from the same stronger GPU
success-replay parent (`7d4e9e57...`). Each gets 262,144 fresh constructions,
four states with four rollouts per update, Adam at `1e-4`, 50% root sampling
when contexts exist, a zero-initialized global-context head, and no extra solve bonus. The 50% root
fraction is an experiment based on one small continuation; it is not an
established optimum. Group normalization is a competing objective weighting,
not an assumed improvement over RMS. Both arms use the same architecture;
the serial pilot does not isolate the head's causal contribution. To test the
old architecture separately, submit with `global_flag=`. Neither arm uses
retained graphs.

After making the new files available on Bernoulli:

```bash
cd /Home/optimier/schuette/tnreasonproj/tnreason-py
condor_submit experiments/cluster_selection/reinforcement/refine_gpu.sub
```

Every submission gets a new cluster-ID directory, with a separate run per
normalization and seed.
The shell wrapper locks each run, creates a new segment, and resumes its latest
rolling checkpoint after an eviction. It never silently resets Adam or switches
to weights-only continuation. Do not change source/settings in an active run.
The supplied initialization is a previous **trained** checkpoint, not a fresh
warm-up. Omit `--initial` in a direct `refine` invocation for a scratch run.
The template requests up to 120 seconds for graceful shutdown and removal.
HTCondor's [machine and administrator policies](https://htcondor.readthedocs.io/en/24.0/man-pages/condor_submit.html)
can shorten or omit this grace period, so the rolling checkpoint remains
necessary.

Validation runs on CPU and can leave the GPU idle. The root and residual phases
each have their own time limit; 55 instances at 30 seconds can therefore take
up to 55 minutes, plus setup and optional one-step trials. Initial GAC preparation
also uses CPU. Per-puzzle validation progress and training GPU peak memory are
logged. GPU speed and retained-graph peak memory have not been measured locally;
`--retain-graphs` remains opt-in.
Wall-clock validation depends on the worker's CPU. Compare selected checkpoints
and the untrained reference on the same machine before interpreting differences
between runs on different workers.

Use the new adapter-aware evaluation command for saved policies:

```bash
python -m experiments.cluster_selection.reinforcement.refine_eval DATA.json \
  --checkpoint CHECKPOINT.pt --output NEW_ROOT_RESULTS.json \
  --attempts 0 --seconds 30

python -m experiments.cluster_selection.reinforcement.refine_eval DATA.json \
  --checkpoint CHECKPOINT.pt --output NEW_MAC_RESULTS.json \
  --full-mac --attempts 32 --seconds 300 --nodes 10000
```

The first command measures root propagation, then plain residual DFS. The second
starts from the original input domains and uses learned clusters throughout DFS,
with one whole-solve time limit. These node totals describe different algorithms.
`--indices` can select explicit evaluation instances; otherwise the dataset's
`validation_indices` are used. Checkpoint training fingerprints are required and
checked against those instances. Reports include source/data/checkpoint hashes,
configuration, timing scope and completed-search counts. Existing output files
are refused. Independent sampled usefulness is optional (`--samples 8`) and is
excluded from root timing; it is not available in `--full-mac` mode.

For embedding in an existing solver:

```python
from experiments.cluster_selection import runtime
from experiments.cluster_selection.reinforcement.actor import load
runtime.attach(state, load("CHECKPOINT.pt"))
```

Pass the loaded model object: the frozen legacy pathname loader cannot load the
new optional heads. Temporary summary removal and backtracking still belong to
the existing tested runtime.
