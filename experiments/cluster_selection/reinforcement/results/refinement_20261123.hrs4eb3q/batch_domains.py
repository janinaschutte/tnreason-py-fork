"""Prototype one indexed domain-mask write, preserving the existing features."""
from collections.abc import Mapping
from hashlib import sha256
import inspect
import json
from numbers import Integral
from pathlib import Path
from statistics import median
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import torch

from experiments.cluster_selection.encoder import Encoder
from experiments.cluster_selection.reinforcement.actor import load

BASE = Path(__file__).resolve().parent


def domain_features(self, domains):
    if isinstance(domains, Mapping):
        domains = [domains[color] for color in self.network["colors"]]
    if len(domains) != len(self.dimensions):
        raise ValueError("Expected every variable's domain in network order")
    rows, columns = [], []
    for idx, values in enumerate(domains):
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               or not 0 <= value < self.network["dimensions"][idx] for value in values):
            raise ValueError("Domain value outside variable dimension")
        values = list(values)
        rows.extend([idx] * len(values))
        columns.extend(values)
    mask = self.dimensions.new_zeros(len(domains), self.max_dim)
    if columns:
        mask[torch.tensor(rows, dtype=torch.long, device=mask.device),
             torch.tensor(columns, dtype=torch.long, device=mask.device)] = 1
    scalars = torch.stack((self.dimensions / self.max_dim, mask.sum(1) / self.dimensions,
                           (self.dimensions == 2).to(mask.dtype), self.degree.log1p()), dim=1)
    return torch.cat((mask, self.valid.to(mask.dtype), scalars), dim=1)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(mkdtemp(prefix="domain_features_", dir=BASE))
    data_path, states_path = BASE / "pilot_data.json", BASE / "pilot_states.json"
    data, cached = json.loads(data_path.read_text()), json.loads(states_path.read_text())
    assert cached["pilot_data_sha256"] == sha256(data_path.read_bytes()).hexdigest()
    states = [row["domains"] for row in cached["states"]]
    assert len(states) == 48
    parent = BASE.parent / "learning_diagnostics_20261113.67i11ilh/heldout/reinforce.pt"
    model = load(parent)
    features_equal = encoded_equal = 0
    for domains in states:
        assert torch.equal(Encoder.domain_features(model, domains), domain_features(model, domains))
        features_equal += 1
        with torch.no_grad():
            original = model.encode(domains)
            model.domain_features = lambda values: domain_features(model, values)
            try:
                changed = model.encode(domains)
            finally:
                del model.domain_features
            assert len(original) == len(changed) and all(torch.equal(a, b) for a, b in zip(original, changed))
        encoded_equal += 1
    generic = Encoder(dict(colors=["a", "b"], dimensions=[2, 3],
        cores=[dict(key="ab", colors=[0, 1], shape=[2, 3], values=[True]*6)]), width=4, rounds=1)
    valid = [lambda: [[], []], lambda: [[0], [2]], lambda: [[1, 0, 1], [2, 0, 2]],
        lambda: dict(b={0, 2}, a={0, 1}), lambda: [[np.int64(1)], [np.int32(2)]],
        lambda: [(value for value in (0, 1)), (value for value in (0, 2))]]
    invalid = [lambda: [[0]], lambda: dict(a=[0]), lambda: [[True], [0]], lambda: [[-1], [0]],
        lambda: [[2], [0]], lambda: [[0], [3]], lambda: [[1.], [0]], lambda: [[np.bool_(True)], [0]],
        lambda: [["1"], [0]], lambda: [None, [0]]]
    for make in valid:
        assert torch.equal(Encoder.domain_features(generic, make()), domain_features(generic, make()))
    for make in invalid:
        errors = []
        for function in (Encoder.domain_features, domain_features):
            try:
                function(generic, make())
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                raise AssertionError("Invalid domain was accepted")
        assert errors[0] == errors[1]
    times = {"original": [], "batched": []}
    for _ in range(5):
        for domains in states:
            for name, function in (("original", Encoder.domain_features), ("batched", domain_features)):
                began = perf_counter()
                function(model, domains)
                times[name].append(perf_counter() - began)
    profiles = {}
    for name, function in (("original", Encoder.domain_features), ("batched", domain_features)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            function(model, states[0])
        profiles[name] = {row.key: row.count for row in profile.key_averages()
                         if row.key in ("aten::index_put_", "aten::_index_put_impl_", "aten::empty", "aten::to", "aten::_to_copy")}
    report = dict(features_bitwise_equal=features_equal, full_encoded_tuples_bitwise_equal=encoded_equal,
        generic_valid_cases=len(valid), generic_identical_error_cases=len(invalid), variables=len(data["network"]["colors"]),
        median_cpu_seconds={name: median(values) for name, values in times.items()},
        CPU_profiler_operator_counts=profiles, cpu_feature_speedup=median(times["original"])/median(times["batched"]),
        calls_per_method=len(times["original"]), torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
        scope="CPU0 one thread, cached48development states; no actual GPU speed measurement or solver performance claim",
        input_sha256={str(path): sha256(path.read_bytes()).hexdigest() for path in (data_path, states_path, parent, Path(__file__))})
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    Path("/tmp/domain_features_override.txt").write_text(inspect.getsource(domain_features))
    print(output, json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
