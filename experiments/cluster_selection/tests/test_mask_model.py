import numpy as np
import pytest
import torch

from tnreason.engine.workload_to_numpy import NumpyCore
from experiments.constraint_networks.constraint_networks import ConstraintNetwork
from experiments.cluster_selection.data import network_data
from experiments.cluster_selection.mask_model import MaskPolicy
from experiments.cluster_selection.candidate_evaluation_tnreason import SummaryClusterCandidate
from experiments.cluster_selection import runtime


def example():
    network = ConstraintNetwork({
        "ab": NumpyCore(values=np.eye(2, dtype=bool), colors=["a", "b"]),
        "bc": NumpyCore(values=np.eye(2, dtype=bool), colors=["b", "c"]),
        "ac": NumpyCore(values=np.array([[1, 1], [1, 0]], dtype=bool), colors=["a", "c"]),
        "cd": NumpyCore(values=np.eye(2, dtype=bool), colors=["c", "d"]),
        "free": NumpyCore(values=np.ones(2, dtype=bool), colors=["q"]),
    }, maintain_indices=True)
    raw = network_data(network)
    return network, raw


def test_batched_encoding_equals_existing_encoder_and_single_state():
    torch.manual_seed(1)
    _, raw = example()
    model = MaskPolicy(raw, width=8, rounds=2, anchor_rounds=2)
    domains = [[0, 1]] * len(raw["colors"])
    domains[-1] = [1]
    allowed = model.domain_features(domains)[:, :model.max_dim].bool()
    actual = model.encode_allowed(allowed[None])
    expected = model.encode(domains)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a[0], e)
    batch = model(torch.stack((allowed, allowed)), anchors=torch.tensor([0, 0]))
    single = model(allowed[None], anchors=torch.tensor([0]))
    for key in batch:
        torch.testing.assert_close(batch[key][0], single[key][0])
    assert torch.isneginf(batch["anchor_logits"][:, -1]).all()


def test_anchor_can_be_pruned_outside_selected_cluster_and_outputs():
    torch.manual_seed(3)
    _, raw = example()
    model = MaskPolicy(raw, width=8, rounds=1, anchor_rounds=2)
    allowed = torch.ones(1, len(raw["colors"]), 2, dtype=torch.bool)
    cores = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    opened = torch.zeros(1, len(raw["colors"]), dtype=torch.bool)
    opened[0, raw["colors"].index("a")] = True
    pruned = torch.zeros_like(opened)
    anchor = raw["colors"].index("d")
    pruned[0, anchor] = True
    loss, metrics = model.loss(allowed, cores, opened, pruned, anchors=[anchor])
    assert loss.isfinite() and metrics["anchor_loss"].isfinite()
    loss.backward()
    assert model.anchor_marker.grad is not None and model.anchor_marker.grad.abs().sum() > 0
    assert model.anchor_head[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="must have been pruned"):
        model.loss(allowed, cores, opened, pruned, anchors=[raw["colors"].index("a")])


def test_learned_masks_can_reproduce_a_useful_cluster_without_proposals():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    network, raw = example()
    model = MaskPolicy(raw, width=16, rounds=2, anchor_rounds=2, max_cores=5, max_open_colors=2)
    allowed = torch.ones(1, len(raw["colors"]), 2, dtype=torch.bool)
    cores = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    opened = torch.zeros(1, len(raw["colors"]), dtype=torch.bool)
    opened[0, raw["colors"].index("a")] = True
    pruned = torch.ones_like(opened)
    pruned[0, raw["colors"].index("q")] = False
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(120):
        optimizer.zero_grad()
        loss, _ = model.loss(allowed, cores, opened, pruned)
        loss.backward()
        optimizer.step()
    candidate = model.eval().propose({color: (0, 1) for color in raw["colors"]})
    assert candidate is not None and set(candidate.core_keys) == {"ab", "bc", "ac"}
    assert candidate.open_colors == ("a",)
    state = network.create_mac_state()
    original = state.snapshot_domains()
    state.push_level()
    runtime.attach(state, model, max_candidates=1)
    assert state.propagate()
    assert all(state.domains[color] == (0,) for color in ("a", "b", "c", "d"))
    assert state.domains["q"] == (0, 1)
    assert state.on_closure.statistics["selected"] == 1
    assert not state._temporary_core_keys and set(state._cores) == set(network.coresDict)
    state.pop_level()
    assert state.snapshot_domains() == original


def test_mask_checkpoint_roundtrip_and_invalid_domains(tmp_path):
    _, raw = example()
    model = MaskPolicy(raw, width=8, rounds=1)
    path = tmp_path / "mask.pt"
    torch.save(dict(version=1, kind="mask", network=raw, config=model.config,
                    reward_colors=raw["colors"], weights=model.state_dict()), path)
    restored = runtime.load(path)
    assert isinstance(restored, MaskPolicy) and restored.config == model.config
    for key in model.state_dict():
        torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key])
    with pytest.raises(ValueError, match="nonempty domains"):
        model(torch.zeros(1, len(raw["colors"]), 2, dtype=torch.bool))


def test_ranked_anchors_share_encoding_and_match_individual_predictions(monkeypatch):
    torch.manual_seed(11)
    _, raw = example()
    model = MaskPolicy(raw, width=8, rounds=2, anchor_rounds=2,
                       max_cores=5, max_open_colors=2, anchor_candidates=3).eval()
    # Keep decoding nonempty so this checks actual candidates as well as logits.
    with torch.no_grad():
        model.count_head[-1].weight.zero_()
        model.count_head[-1].bias.fill_(-10)
        model.count_head[-1].bias[3] = 10
        model.count_head[-1].bias[model.config["max_cores"] + 2] = 10
    domains = {color: (0, 1) for color in raw["colors"]}
    domains["q"] = (1,)
    allowed = model._allowed(domains)
    with torch.no_grad():
        scores = model(allowed)["anchor_logits"][0]
        anchors = scores.argsort(descending=True, stable=True)[:3]
        expected = [model(allowed, anchors=[int(anchor)]) for anchor in anchors]
    encode, condition = model.encode_allowed, model._condition
    encoded, predictions = [], []

    def record_encode(mask):
        encoded.append(mask.shape[0])
        return encode(mask)

    def record_condition(variables, cores, chosen):
        prediction = condition(variables, cores, chosen)
        predictions.append(prediction)
        return prediction

    monkeypatch.setattr(model, "encode_allowed", record_encode)
    monkeypatch.setattr(model, "_condition", record_condition)
    candidates = model.proposals(domains)
    assert encoded == [1] and len(predictions) == 1
    actual = predictions[0]
    assert actual["anchors"].tolist() == anchors.tolist()
    assert raw["colors"].index("q") not in actual["anchors"].tolist()
    for index, single in enumerate(expected):
        for key, values in actual.items():
            torch.testing.assert_close(values[index], single[key][0])
    expected_candidates = list(dict.fromkeys(model._candidate(single) for single in expected))
    assert candidates == expected_candidates
    model.config["anchor_candidates"] = 1
    assert model.proposals(domains) == [model.propose(domains)]
    assert model.proposals({color: (0,) for color in raw["colors"]}) == []


def _rank_anchors(model, monkeypatch, ordering):
    def scores(variables, cores, allowed):
        result = torch.full(allowed.shape[:2], -torch.inf, device=variables.device)
        for rank, color in enumerate(ordering):
            index = model.network["colors"].index(color)
            result[:, index] = len(ordering) - rank
        return result.masked_fill(allowed.sum(-1) <= 1, -torch.inf)
    monkeypatch.setattr(model, "_anchor_scores", scores)


def test_ranked_anchors_deduplicate_masks_and_respect_decode_budgets(monkeypatch):
    _, raw = example()
    model = MaskPolicy(raw, width=8, rounds=1, anchor_candidates=4)
    _rank_anchors(model, monkeypatch, ["d", "c", "b", "a", "q"])
    noop = SummaryClusterCandidate("mask", ("free",), ("q",))
    useful = SummaryClusterCandidate("mask", ("ab", "bc", "ac"), ("a",))
    decoded = []

    def candidate(prediction, row=0):
        anchor = raw["colors"][int(prediction["anchors"][row])]
        decoded.append(anchor)
        return noop if anchor in ("d", "c") else useful

    monkeypatch.setattr(model, "_candidate", candidate)
    domains = {color: (0, 1) for color in raw["colors"]}
    assert model.proposals(domains) == [noop, useful]
    assert decoded == ["d", "c", "b", "a"]
    decoded.clear()
    assert model.proposals(domains, max_candidates=2) == [noop]
    assert decoded == ["d", "c"]
    decoded.clear()
    assert model.proposals(domains, max_steps=1) == [noop]
    assert decoded == ["d"]
    with pytest.raises(ValueError, match="positive"):
        MaskPolicy(raw, anchor_candidates=0)
    with pytest.raises(ValueError, match="positive"):
        model.proposals(domains, max_candidates=0)


def test_second_anchor_prunes_after_first_noop_only_when_opted_in(monkeypatch):
    network, raw = example()
    model = MaskPolicy(raw, width=8, rounds=1, anchor_candidates=2)
    _rank_anchors(model, monkeypatch, ["q", "d", "c", "b", "a"])
    noop = SummaryClusterCandidate("mask", ("free",), ("q",))
    useful = SummaryClusterCandidate("mask", ("ab", "bc", "ac"), ("a",))
    decoded = []

    def candidate(prediction, row=0):
        anchor = raw["colors"][int(prediction["anchors"][row])]
        decoded.append(anchor)
        return noop if anchor == "q" else useful

    monkeypatch.setattr(model, "_candidate", candidate)
    strict = runtime.attach(network.create_mac_state(), model,
                            max_candidates=64, stop_on_noop=True)
    assert strict.propagate()
    assert all(domain == (0, 1) for domain in strict.domains.values())
    assert strict.on_closure.statistics["selected"] == 1
    assert strict.on_closure.statistics["useful"] == 0

    capped = runtime.attach(network.create_mac_state(), model,
                            max_candidates=1, stop_on_noop=False)
    assert capped.propagate()
    assert all(domain == (0, 1) for domain in capped.domains.values())
    assert capped.on_closure.statistics["selected"] == 1

    decoded.clear()
    state = runtime.attach(network.create_mac_state(), model,
                           max_candidates=64, stop_on_noop=False)
    original = state.snapshot_domains()
    state.push_level()
    assert state.propagate()
    assert all(state.domains[color] == (0,) for color in ("a", "b", "c", "d"))
    assert state.domains["q"] == (0, 1)
    assert state.on_closure.statistics["selected"] == 2
    assert state.on_closure.statistics["useful"] == 1
    # The final q prediction is freshly decoded after the useful application.
    assert decoded == ["q", "d", "q"]
    assert not state._temporary_core_keys and set(state._cores) == set(network.coresDict)
    state.pop_level()
    assert state.snapshot_domains() == original
