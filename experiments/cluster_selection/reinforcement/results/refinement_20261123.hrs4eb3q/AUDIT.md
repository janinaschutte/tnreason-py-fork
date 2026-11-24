# Constructive reinforcement learning: audit and controlled pilot

Status: 2026-11-23. This is a development diagnosis, not a new held-out result.
The code has testable improvements; the completed small pilots do not establish a robust improvement in Sudoku solving.

## Data and evaluation scope

- The existing source contains 10,000 training puzzles and 55 validation puzzles; 47 validation roots remain unresolved after ordinary GAC.
- The pilot uses 32 training roots sampled across GAC-closed entropy quartiles and 16 shuffled unresolved development-validation roots.
- Selection seeds, source hashes, indices and normalized puzzle hashes are recorded in the compact [split_summary.json](split_summary.json).
- The held-out 100 benchmark is not read to select these changes. These 16 development puzzles are reused for diagnostics and must not become a new claimed test set.
- Training states can include verified useful transitions, but a new state is not an independent puzzle. Root/context frequencies therefore need explicit reporting.

## What the old learner actually trained

The reference is the scratch, zero-warm-up, zero-imitation `sampling_grow` REINFORCE run:
[configuration](../reinforce_sampling_grow_gpu_20261106.q3mc4c0g/run_plan.json),
[training output](../reinforce_sampling_grow_gpu_20261106.q3mc4c0g/training.7906.0.out),
and [best checkpoint](../reinforce_sampling_grow_gpu_20261106.q3mc4c0g/job_7906.0/policy.best.pt).
Its encoder has width 32, two message-passing rounds and 37,985 parameters.
The untrained baseline keeps the same construction masks, growth decoder, contraction limits and GAC, using freshly initialized neural weights with seed0.
The represented Sudoku CSP has 1,053 variables and 2,916 original tensor cores:
81 primary nine-valued variables, 243 auxiliary nine-valued variables and 729 binary auxiliaries.

Four independently completed clusters from the same domains provide a leave-one-out baseline:

\[
L=-\frac1K\sum_i\frac{R_i-b_{-i}}s\sum_t\log\pi_\theta(a_{it}\mid D,C_{it})-\eta\bar H,
\qquad b_{-i}=\frac{\sum_{j\ne i}R_j}{K-1}.
\]

The original score-function gradient is valid for immediate cluster reward: action log probabilities are summed, and replaying actual growth masks matches sampling.
A constant epsilon mixture during training versus greedy inference is a distribution difference, not evidence of an incorrect gradient.
Adam used constant learning rate 0.0003, gradient clipping 1 and entropy weight 0.001.
There were no logged gradient norms, sampled policy drift or construction-diversity diagnostics.

The best checkpoint at chunk 52 (2.6 passes, 57,632 updates) reduced the historical 55-puzzle validation total from 135 to 105 DFS nodes.
Training to chunk 200 (10 passes, 216,648 updates) ended at 149 nodes; additional roots solved fell from 22 at best to 12 at final.
Longer training demonstrably failed to retain the earlier result; this alone does not identify an optimizer or representation cause.
Dynamic state pools grew from 10,000 roots to 48,485 states at best and 90,520 at final, so live training averages also changed their state distribution.

At chunk 52 the log records 83,977 teacher contractions versus 4,420 fresh policy contractions, about 19 teacher evaluations per policy trial.
Imitation had coefficient 0, but teacher fallback still supplied 326 state advances across 1,105 fresh states.
Eager teacher evaluation is removable overhead for a teacher-free learner, but removing it also changes the curriculum.
The new trainer explicitly makes that change and gives no credit for a teacher transition.
Old weights-only continuation reset Adam, RNG, reward scale, caches and state pools; it was not exact training resumption.

Earlier continuations gave only modest, one-seed evidence: unchanged learning rate best 105/final 112 nodes; lower rate best 104/final 107; success replay best 99/final 105.
Their [protocol](../learning_diagnostics_20261113.67i11ilh/continuations/protocol.json) and saved checkpoints define that comparison.
They support testing retention mechanisms, not assuming replay or another optimizer will solve the problem.

“Success replay” refers to two different continuations of the trained RF parent, not one checkpoint.
The portfolio and ongoing1800s benchmark use the small CPU continuation:86 additional updates on16 roots, epoch6, SHA083879d…; the later GPU best is epoch4/4,670 additional updates/2,000 puzzle visits, SHA7d4e9e….
The GPU checkpoint was selected at33 roots and99 historical full-MAC nodes on55 development puzzles. Its audited current-protocol results appear below; historical full-MAC nodes and current separate plain-residual nodes measure different procedures. [Checkpoint bookkeeping](ongoing_replay_metadata.json) and [full metadata](prior_checkpoints.json) distinguish them; the existing benchmark remains unchanged and its test outcomes are not read here.

## Architecture and construction evidence

The old input omits explicit membership of the reward's primary-variable set.
An independently reproduced generic symmetric CSP gives identical anchor logits when those roles are swapped, despite different rewards:
[objective_visibility_audit.json](objective_visibility_audit.json).
This is a concrete missing input, not proof that it explains Sudoku performance: Sudoku variable dimensions already distinguish some roles.
New optional zero-initialized heads expose objective role, global anchor context and construction distances while preserving old checkpoint behavior initially.
The completed objective-role and global-context pilots are reported below; generic input distinguishability does not establish a Sudoku benefit.
The compact [architecture receipt](compactarchitecture_receipt.json) records the feature-equivalence checks, role witness, head ablation and expert-proof capacity checks.

The [length diagnostic](../length_diagnostic_seed29_0rmst_im/REPORT.txt) found 15 useful full constructions among 64 samples.
Fourteen of those 15 had a shorter prefix giving the same pruning, but usefulness typically first appeared at 53–63 cores.
Retaining only the anchor was useful in 2/64 cases versus 15/64 with the original outputs.
This supports measuring stopping and output choice; it does not justify restricting clusters to be small.
Those retrospective prefixes already knew the final outputs and are not deployable early-stop predictions.

The [expert witness audit](expert_capacity__z5uzgje/REPORT.txt) examines 138 saved useful applications across all 16 development puzzles, without retraining on expert examples.
Their median size is 8 cores, median output count 1, and conditioned work ranges 244–37,683; 78 summaries retain only auxiliary variables.
After removing factors identically true under their recorded domains, 119/138 witnesses fit the learner's 64-core/6-output/1M-work/100K-entry limits and admit legal free-stop traces.
The other 19 exceed core count or expanded output size; larger expert work limits do not explain most of its success.
None of these 119 exact compact sets can be emitted by the current forced-growth sampler because stop remains masked while additions are possible.
Affordable supersets can still imply the same pruning, so this is a stopping restriction, not proof that the useful deduction is unreachable.
The expert visits its own sequential states; witness affordability is not a matched solving comparison.
Filtering tautological cores is a rejected root-action fix: [eligible_tautologies.json](eligible_tautologies.json) finds none among9,560 eligible cores across16 closed validation roots; the expert's removed factors were already excluded by the constructor's active-variable restriction.

## What the new training code changes

- [refine.py](../../refine.py) batches independent state groups before an optimizer step, uses on-policy samples without epsilon contamination, and records root/context exposures.
- [updates.py](../../updates.py) keeps actual trace masks and summed likelihoods, checks finite losses/gradients, logs clipping, and skips groups with identical rewards.
- Leave-one-out centering is retained. Prior RMS scaling is a control; within-group standard-deviation scaling is an explicitly experimental optimization heuristic.
- Multiple optimization epochs require clipped importance ratios; excessive sampled approximate KL discards the pending step. Default pilots use one vanilla epoch.
- This KL is measured before a step; it is zero for the first on-policy pass and does not measure post-step drift in the one-epoch pilots.
- Optional retained rollout graphs avoid a second neural forward for one vanilla update. Small CPU fixtures match rescored updates exactly; [two actual Sudoku traces](sudoku_gradient_47udu0q4/summary.json) give numerically close gradients and anchor probabilities, not bitwise-equal updates. GPU memory/throughput remains unmeasured.
- Exact resume includes optimizer, random states, reward scale and contexts, with data/configuration/source checks. New output directories and source archives preserve earlier runs.
- [test_refine_updates.py](../../../tests/test_refine_updates.py) checks gradients, clipping signs, masks, finite guards, cached graphs and zero groups. Existing supervised code and old runtime remain unchanged.

The small [forward profile](batch_profile_1klaogkl/summary.json) attributes about 86% of collection time to neural rollout calls and about 10% to candidate measurement.
It is a shared-CPU profile with four zero-reward samples and no optimizer step, not a solver timing benchmark or evidence of GPU utilization.
Static encoding reuse alone addresses a small fraction; many small partial-graph scorers remain a batching opportunity.
A fixed generic six-core capacity check improves sampled useful constructions16/128→127/128 after512 fresh training trials, while preserving all four globally valid assignments ([audit](generic_capacity_5o09mv3r/audit.json)). This deliberately fits one small network; free variables prevent root solving. It verifies that the learning machinery can acquire useful constructions, not Sudoku generalization.

## Completed 2,048-trial pilot

Each trained arm starts from the same frozen best RF parent, samples 32 fixed training roots, uses four states × four trials per batch, and stops after 2,048 fresh contractions.
Learning rate is 0.0001, validation has 32 application attempts and a 30s root ceiling, and residual plain MAC measures remaining DFS under identical branching.
The final checkpoint is prescribed by trial count; no best-checkpoint search selects these numbers.
All 16 puzzles, including the largest node outliers, remain in the criterion.
Inputs, pairs, intervals and common-seed endpoint measurements are in [analysis_t5p_vb88](analysis_t5p_vb88/REPORT.txt).

| Method | Root solves | Residual DFS nodes | Root wall seconds |
|---|---:|---:|---:|
| Plain MAC |0/16|242|0.74|
| Structural heuristic, bounded32 |2/16|228|1.86|
| Untrained constructor |5/16|67|173.19|
| Frozen trained RF parent |7/16|58|161.32|
| New RMS, seed0 |7/16|60|142.22|
| New group normalization, seed0 |6/16|68|151.00|
| New global context + group, seed0 |8/16|51|142.33|
| New group normalization, seed1 |7/16|62|135.26|
| New objective role + RMS, seed0 |8/16|56|141.67|
| New objective role + group, seed0 |5/16|59|137.95|
| Global + group,75% root exposure requested |8/16|62|144.80|
| Global + group,50% root exposure requested |6/16|61|157.26|
| New global context + RMS, seed0 |6/16|149|90.33|
| Sudoku-specific expert,51 families |14/16|32|28.94|

Neural times are fresh serial CPU3 measurements; plain/heuristic/expert times are reused earlier observations, with different host contention.
The two later context arms use audited training-end measurements on CPUs1/2, not that serial timing pass.
Model loading, setup, sampled trials and residual DFS are excluded. No fixed-cap root hit its ceiling; every residual search completed.
The expert has larger cluster limits and problem-specific rules; it is a reference, not a matched generic arm.

Global context improves two puzzles and worsens two: mean DFS delta −0.4375, paired 95% bootstrap interval [−3.5625,2.1875].
One puzzle accounts for −19 nodes; the other 15 sum to +12. This is encouraging root coverage, not a robust aggregate gain.
RMS/group0/group1 totals change by +2/+10/+4 nodes; every corresponding DFS interval crosses 0.
Group normalization therefore remains provisional. Bootstrap resamples paired puzzles 10,000 times and does not cover training-seed uncertainty.
All common single-cluster comparisons use seed100000: parent49/128 useful; RMS38, group0 40, global44, group1 44.
Objective+RMS changes mean DFS by −0.125 (95% interval [−3.1891,2.5]), with three root wins and two root losses; objective+group loses two roots and gains none.
Their matching sampled usefulness changes 49→41/128 and 49→45/128 respectively; objective+RMS also reduces mean immediate reward (paired 95% interval [−0.0979,−0.0182]).
These short, one-seed role-feature runs do not demonstrate a solving improvement, despite the generic missing-input witness.
The original differently seeded45/128 baseline is not used for these paired comparisons.
The context arms realize81.25%/52.54% root exposure because initial visits must use roots; useful transitions supply the other states.
Neither improves fixed32 DFS over the root-only global model. Their four-state groups use retained graphs: actual Sudoku gradients are numerically close to the earlier recomputation path, not bitwise identical.
The75%/50% pair itself shares the same source version and graph mode; all initial policy weights, domains and reward targets match.
The matching global+RMS control ends at149 nodes and six roots, with43/128 sampled-useful trials versus49 initially; its prescribed final endpoint is reported even though initialization remains its best checkpoint. A global head is therefore not an independently demonstrated fix.
A [post-hoc global-head ablation](global_parts_tpph3uj1/paired_comparison.json) preserves all eight root solves when that head is zeroed, while residual DFS rises51→61; transplanting only the learned head onto the old parent exactly preserves its58-node result. Root coverage came from base-weight updates in this pilot; the head's observed gain needs co-adaptation and is concentrated in one puzzle.

## All 55 development puzzles, up to32 attempts

These include 8 puzzles already solved by GAC. Root propagation retains pruning, then a separate plain MAC search measures residual nodes; this is not full MAC with learned clusters at every search node.
The [parent/global audit](full55_q13qt7s1/audit.json), [role audit](full55_objective_rms_seed0_2s1qqgfn/audit.json), [random receipts](full55_random_216gsslg/untrained_after.json) and [reference audit](full55_references_watyom61/audit.json) passed.

| Method | Root solves | Residual DFS | Root seconds | Residual seconds |
|---|---:|---:|---:|---:|
| Plain MAC |8/55|479|4.74|6.07|
| Structural heuristic |16/55|357|5.73|2.46|
| Untrained constructor |22/55|249|438.68|3.51|
| Frozen trained parent |30/55|157|418.23|2.78|
| New global context + group |33/55|145|412.66|2.71|
| New objective role + RMS |30/55|172|380.65|2.68|
| Earlier GPU success replay, best |33/55|147|331.46|2.12|
| Sudoku expert,51 families |52/55|74|85.53|1.47|

Global context wins 5 roots and loses 2 versus parent; the 95% paired interval for root-fraction change is [−0.0364,0.1455], and for total DFS change [−64,31].
The role model wins 5 roots and loses 5: sampled reward improves on 55, while DFS worsens by 15. Single-cluster reward and solving quality need separate reporting.
All residuals completed. Shared-host times are indicative; the existing 55 puzzles also selected the old parent checkpoint and are not a fresh test.

## 30 seconds per root on16 development puzzles

The final control interleaves all three neural models puzzle by puzzle on one pinned CPU, with a shuffled puzzle order and rotated method order. Each root has30s and no application-attempt cap. All models use the same optimized domain-feature implementation and adapter-aware checkpoint loader; domain features and encoder outputs match the original implementation on48 saved states. [Serial audit](serial_time30_c8uq_f33/summary.json), [all protocols](consolidated_jt6jc5l3/TABLES.txt).

| Method | Root solves | Residual DFS | Root seconds | Residual seconds |
|---|---:|---:|---:|---:|
| Plain MAC |0/16|242|0.74|1.43|
| Progressive heuristic |6/16|194|310.16|2.23|
| Untrained constructor |12/16|48|207.43|0.38|
| Earlier GPU success replay, best |11/16|48|203.14|0.39|
| New global context + RMS |12/16|36|204.25|0.34|
| Sudoku expert,51 families |14/16|32|28.94|0.30|

Global+RMS matches the untrained constructor's exact12 solved roots. It lowers residual DFS48→36 through two puzzles;11 of those12 saved nodes come from source puzzle20 (24→13), and the paired95% interval for mean DFS change is[−2.1875,0]. There is no additional root solve versus random and no demonstrated training-seed robustness.
Opportunities are now similar:1,093 proposals/1,054 contractions for RMS versus1,095/1,062 for random. Mean cluster sizes are62.60 versus62.85 cores; mean outputs are5.15 versus5.91. RMS makes109 useful applications versus94, with16.73M estimated work versus8.08M. The residual improvement survives this common run and cannot be attributed solely to earlier host contention.
The poor fixed32 RMS result reverses under repeated proposals for30s; this is a budget-dependent outcome, not a reason to discard its negative fixed-cap result. Other earlier time-only arms gave11 roots/48 nodes for global+group and role+RMS, and12/47 for the50%-root group arm; their separate-core timings are not the serial control.
All48 assignments passed original-table/given/reference checks and source/input/checkpoint hashes remained unchanged. Deadline hits are expected budget censoring; all residual searches completed. Plain/heuristic/expert rows reuse earlier measurements and the expert has larger problem-specific limits. External shared-host activity remains uncontrolled.

Actual full MAC, which applies clusters again inside DFS, is a separate protocol: the [final comparison](fullmac_final_3297pi_m/TABLE.txt) uses32 attempts per closure and300s per whole solve. Plain/heuristic/untrained/RF-parent/global-group/GPU-replay/global-RMS use242/187/45/32/36/36/40 nodes respectively, with0/2/5/7/8/7/6 root solves; all16 searches complete for every method. All112 assignments pass independent original-constraint/given/input checks. The expert's14 roots/32 nodes remain a separate root+plain reference.
Global+RMS improves on the45-node untrained full-MAC baseline but worsens the32-node RF parent and36-node GPU replay. Its174.84s whole-solve time is indicative; historical parent/global times combine completed30s cases with censored cases retried at300s. The30s root-only improvement is therefore not an established improvement to the fixed-cap full-search solver.

## Reward alignment and next decision

Current immediate reward is weighted entropy reduction, \(r_t=[H(D_t)-H(D_{t+1})]/H(D_t)\).
Every positive reward certifies current pruning, but the changing denominator means these rewards are not an additive fixed potential for a whole root episode.
A root-solve bonus can encourage completion; group standardization can erase its magnitude when one rollout succeeds and the others all score 0.
For example, the normalized advantages are unchanged if that lone positive reward changes from 0.001 to 2. RMS or unscaled controls are necessary when testing the bonus.

A future episodic option could use fixed denominator \(H(D_0)\), return-to-go and a first-solve bonus across several verified applications.
Shared on-policy prefixes could compare continuations, but this is a conditional curriculum; full-trace importance ratios require extra care beyond one vanilla update.
Neither option is implemented or validated by these pilot numbers.
The next full retrain should be a controlled experiment from the preserved stronger GPU replay checkpoint: paired RMS and group normalization over three seeds, both with the same zero-initialized global head, retaining the parent baseline. This matches the strongest time-budget pilot without claiming that the head independently caused the gain; no-head operation remains available. The50%-root curriculum is experimental. Fixed32 and time-only evaluation answer different resource questions, and immediate reward does not measure whole-root completion.
