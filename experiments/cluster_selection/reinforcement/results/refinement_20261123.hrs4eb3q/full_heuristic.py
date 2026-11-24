"""Full55 structural-batch reference; no learned or test-data selection."""
from pathlib import Path
import json
import sys
from tempfile import mkdtemp
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[4]))
from full_validation import DATA, PARENT, digest, write
from experiments.cluster_selection.reinforcement import refine, refine_eval
from experiments.cluster_selection.reinforcement.actor import load
import torch

torch.set_num_threads(1)
output = Path(mkdtemp(prefix="full55_heuristic32_", dir=BASE))
data = json.loads(DATA.read_text())
model = load(PARENT)
protected = {str(path): digest(path) for path in (DATA, PARENT, Path(__file__), BASE / "full_validation.py")}
sources = refine.source_hashes()
settings = dict(mode="heuristic", attempts=32, seconds=30., nodes=10000, seed=100000)
write(output / "protocol.json", dict(data=str(DATA), checkpoint=str(PARENT), hashes=protected,
    sources=sources, settings=settings, scope="55 existing development puzzles; structural batch32, then plain residual MAC"))
result = refine_eval.evaluate_baseline(model, data, **settings)
assert sources == refine.source_hashes() and all(digest(Path(path)) == wanted for path, wanted in protected.items())
reference = json.loads((BASE / "full55_references_watyom61/plain.json").read_text())
assignments = {row["index"]: row["residual"]["assignment"] for row in reference["rows"]}
assert all(all(value in row["root"]["domains"][color] for color, value in assignments[row["index"]].items()) for row in result["rows"])
write(output / "result.json", result)
write(output / "audit.json", dict(immutable_inputs=True, valid_outputs=True, retained_references=55))
print(output, result["totals"], flush=True)
