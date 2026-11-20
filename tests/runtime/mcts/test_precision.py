"""FP64 search arithmetic around low-precision neural-network boundaries."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from models.policy.policy_oracle import cfg_of as policy_config
from models.preference.preference_oracle import cfg_of as preference_config

from mcpref.models.policy.network import Policy
from mcpref.models.preference.network import Preference
from mcpref.runtime.mcts.search import decide
from mcpref.runtime.mcts.tree import LatentHistory, Tree, advance, quantize
from mcpref.runtime.sampling.codecs import _decode_tree, _encode_tree
from runtime.mcts.helpers import FakePolicy, FakePreference, config


def real_history(dtype):
    return LatentHistory(
        torch.tensor([[[256.0]]], dtype=dtype),
        torch.tensor([[[0.0, 0.0, 1.0]]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_tree_and_rebased_root_use_fp64_without_mutating_real_history(dtype):
    real = real_history(dtype)
    tree = Tree(real)
    advance(tree, 1, real, torch.ones(1, dtype=torch.float64), 0.01)
    assert tree.base.Y.dtype == tree.base.C.dtype == torch.float64
    assert tree.root.z.dtype == tree.root.x.dtype == torch.float64
    assert real.Y.dtype == real.C.dtype == dtype
    assert tree.root.z.item() == 256.0


def test_quantization_promotes_before_scaled_division():
    # 1 / .01 rounds to 100; BF16's rounded .01 would reconstruct 1.0009765625.
    result = quantize(torch.ones(1, dtype=torch.bfloat16), torch.ones(1), 0.01)
    assert result.dtype == torch.float64
    assert result.item() == 1.0


def test_quantization_resolves_bf16_input_above_half_step():
    # BF16(.025) is .0250244140625, strictly above half of the .05 cell.
    result = quantize(torch.tensor([0.025], dtype=torch.bfloat16), torch.ones(1), 0.05)
    assert result.item() == 0.05


def test_bf16_outputs_are_promoted_before_search_arithmetic_and_round_trip():
    tree = Tree(real_history(torch.bfloat16))
    policy = FakePolicy(logits=[0.0, -0.125], shifts=[[0.125], [0.25]])
    preference = FakePreference(logits=(0.00390625, 0.125))
    for model in (policy, preference):
        model.embed = SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16))
    cfg = config(simulations=3, depth=2, width=2, epsilon=0.125)
    decide(
        tree,
        np.ones(2, dtype=bool),
        policy,
        preference,
        torch.ones(1),
        cfg,
        np.random.default_rng(3),
        float("inf"),
    )
    first = tree.root.children[0]
    assert first.x.item() == first.z.item() == 256.125
    assert first.children[0].x.item() == 256.25
    expected_prior = 1 / (1 + math.exp(-0.125))
    expected_value = (1 / (1 + math.exp(-0.125)) + cfg.min_p) * (
        2 / (1 + math.exp(-0.00390625)) - 1
    )
    assert first.prior == pytest.approx(expected_prior, rel=1e-14)
    assert first.value == pytest.approx(expected_value, rel=1e-14)
    assert first.mean == pytest.approx(
        expected_value * (cfg.discount + cfg.discount**2) / 2, rel=1e-14
    )
    for Y, C in policy.readout_calls:
        assert Y.dtype == C.dtype == torch.bfloat16
    for Ya, Yb in preference.calls:
        assert Ya.dtype == Yb.dtype == torch.bfloat16
    for saved in (tree, _decode_tree(_encode_tree(tree), torch.device("cpu"))):
        child = saved.root.children[0]
        assert child.z.dtype == child.x.dtype == child.shift.dtype == torch.float64
        assert child.x.item() == 256.125
        assert child.children[0].x.item() == 256.25


def test_legacy_tree_restore_promotes_all_nodes_and_base():
    tree = Tree(real_history(torch.bfloat16))
    saved = _encode_tree(tree)
    saved["root"].update(
        z=torch.tensor([256.0], dtype=torch.bfloat16),
        x=torch.tensor([256.0], dtype=torch.bfloat16),
        shift=torch.tensor([0.125], dtype=torch.bfloat16),
    )
    saved["base"] = {key: value.bfloat16() for key, value in saved["base"].items()}
    saved["root"]["children"] = [dict(saved["root"], children=[], action=0)]
    restored = _decode_tree(saved, torch.device("cpu"))
    assert restored.base.Y.dtype == restored.base.C.dtype == torch.float64
    assert restored.root.z.dtype == restored.root.x.dtype == torch.float64
    assert restored.root.shift.dtype == torch.float64
    assert restored.root.children[0].x.dtype == torch.float64
    assert restored.root.children[0].parent is restored.root


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_real_networks_keep_their_dtype_and_use_fp64_tree(device, dtype):
    policy = Policy(policy_config(), 1, ["forward", "jump"]).to(device, dtype).eval()
    preference = Preference(preference_config(), 1).to(device, dtype).eval()
    real = real_history(dtype)
    real = LatentHistory(real.Y.to(device), real.C.to(device))
    tree = Tree(real)
    decision = decide(
        tree,
        np.ones(2, dtype=bool),
        policy,
        preference,
        torch.ones(1, dtype=torch.float64, device=device),
        config(simulations=4, depth=2, width=2),
        np.random.default_rng(3),
        float("inf"),
    )
    assert decision.probabilities.dtype == np.float64
    assert decision.probabilities.sum() == pytest.approx(1.0)
    assert all(p.dtype == dtype and p.grad is None for p in policy.parameters())
    assert all(p.dtype == dtype and p.grad is None for p in preference.parameters())
    for child in tree.root.children:
        assert child.z.dtype == child.x.dtype == child.shift.dtype == torch.float64
        assert torch.isfinite(child.z).all() and math.isfinite(child.mean)
    chosen = next(
        child for child in tree.root.children if child.action == decision.action
    )
    restored = _decode_tree(_encode_tree(tree), torch.device(device))
    advance(
        restored,
        chosen.action,
        LatentHistory(chosen.x[None, None], tree.base.C),
        torch.ones(1, device=device, dtype=torch.float64),
        0.1,
    )
    torch.testing.assert_close(restored.root.x, chosen.x, rtol=0, atol=0)
