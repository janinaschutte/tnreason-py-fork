# tnreason for Sudoku

We here demonstrate the usage of `tnreason` for solving Sudoku puzzles using a tensor network approach.

## Background

Sudoku is an instance of a Constraint Satisfaction Problem, which are in more generality representable by boolean tensor networks.

## Animated cluster reasoning

The [README animation](../../images/sudoku_cluster_reasoning.gif) shows a recorded
generic learned-constructor run: 23 givens, 34 assigned cells after ordinary GAC,
then six useful cluster applications that finish the puzzle without DFS.
Fourteen contractions that removed no values are omitted from the animation.
Each displayed transition is the result of contraction **and** GAC; a summary
may first prune auxiliary variables before propagation reaches the visible cells.

Blue pencil marks and unit outlines highlight the selected links for the focus
digit. A highlighted row, column or box supplies context for those links;
it does not mean that every tensor in that unit was selected. Purple marks show
all retained variables, independently of the focus digit. When several retained
position variables belong to the same unit, their digits are grouped in the
sidebar. Red marks show eliminations, and green numbers show inferred assignments.
Temporary summaries are discarded after use, retaining their pruning.

Regenerate the animation from the committed compact trace:

```bash
python -m demonstrations.sudoku.cluster_animation \
  demonstrations/sudoku/examples/cluster_reasoning.json \
  --verify --output /tmp/sudoku-clusters.gif
```

Rendering requires Pillow. `--verify` reconstructs the original `ConstraintNetwork`
and reapplies every recorded cluster under the current domains. It checks all
1,053 variable domains, the displayed tensor links, contraction costs, unchanged
original cores, summary removal and the final assignment. No archived checkpoint
or dataset is needed. The trace records the original evaluation and checkpoint
hashes; it illustrates one successful run rather than an average solving result.
Use `--preview /tmp/sudoku-clusters.png` to export a still frame. Existing exports
are preserved unless `--force` is supplied.

## Files 

- `representation.py` spezifies the Sudoku rules as categorical constraints.
- `visualization.py` visualized the Sudoku puzzle as arrays.
- `solution.py` prepares a solution based on an Expectation-Propagation algorithm.

## Examples

Examples can be found in `examples/sudoku/examples/`:
- `n=2.py` a simple example of a (2x2)x(2x2) Sudoku puzzle, which is solved by single atom-categorical message passing.
- `n=3.py` a more complex example of a (3x3)x(3x3) Sudoku puzzle

We further apply the algorithms on the [Sudoku-Bench Dataset](https://huggingface.co/datasets/SakanaAI/Sudoku-Bench) of Sakana AI in `examples/sudoku/sudoku_bench`.

## Optional Sudoku clusters for MAC

Generic MAC and the benchmark MAC adapters do not automatically select Sudoku
clusters. The helpers below are separate, explicit experiment utilities. Use
`network.solve_mac(domains=domains)` for generic MAC without these helpers.

For a `ConstraintNetwork` using the standard categorical Sudoku encoding:

```python
from demonstrations.sudoku.inferenceClusters.mac import prepare_sudoku_state

state = prepare_sudoku_state(network, domains=domains, num=3, pairs=True, locked=True)
result = network.solve_mac(state=state)
```

Use `num=2` for 4x4 Sudoku; omit `domains` when givens are already cores.
With `pairs=True`, preprocessing selects naked/hidden pairs after root propagation
and contracts their eight original links into four-variable Boolean tensors.
With `locked=True`, affected row–box and column–box intersections are checked
after each GAC closure, including during search. A two-variable Boolean summary
is added only when pointing or claiming can remove candidates. Blank Sudoku
adds none. Contracted tensors are cached for reuse, keyed by their actual
Boolean supports and local tensor topology. Pairs default on in this
helper; locked candidates default off. Either can be enabled independently.
The propagated domains and contraction caches are reused for search; only the
new cores and affected constraints are queued. Summaries do not include the
selection domains. Branch additions are removed on backtracking, while their
cached tensors can be reused in later branches. Original cores remain unchanged;
generic MAC has no Sudoku rules.
Pairs first appearing later in search are not selected automatically.
Already solved or contradictory roots do not need extra summaries.

`inferenceClusters/CODEGEN_PROMPT.txt` contains a code-generation task for
extending the standalone cluster library while keeping MAC and its benchmark
adapters independent of Sudoku heuristics.

## Standalone rule library

The expanded [inference cluster library](inferenceClusters/README.md) describes
finite rule patterns and checks their consequences by exact Boolean contraction
of the original tensors. Select rule families explicitly through its registry;
importing the library does not enable Sudoku heuristics in generic MAC.

The comparison experiment accepts a Nikoli100 parquet dataset and writes a new
local result directory:

```bash
python -m experiments.sudoku_rule_clusters \
  --dataset /path/to/nikoli100.parquet \
  --families locked_candidates basic_fish --limit 10
```

Reading parquet requires a pandas parquet engine such as PyArrow. See
`python -m experiments.sudoku_rule_clusters --help` for scan, contraction, and
search budgets.
