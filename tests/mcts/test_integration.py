"""Several complete decisions against the independent simulation oracle."""

import numpy as np
import pytest
import torch

from mcpref.mcts.search import decide
from mcpref.mcts.tree import Tree

from .helpers import FakePolicy, FakePreference, config, latent_history
from .search_oracle import RefTree, ref_decide


def networks():
    base = torch.tensor(
        [[0.03, -0.02], [-0.01, 0.04], [0.02, 0.01]], dtype=torch.float64
    )

    def logits(R):
        return torch.stack(
            (R[..., 0], R[..., 1], 0.2 * (R[..., 0] - R[..., 1])), dim=-1
        )

    def preference(Ya, Yb):
        return Ya[0, -1, 0] - Yb[0, -1, 0], 0.3 * (Ya[0, -1, 1] + Yb[0, -1, 1])

    policy = FakePolicy(logits_fn=logits, shift_fn=lambda R, a: base[a] + 0.01 * R)
    return policy, FakePreference(logits_fn=preference)


def assert_tree(actual, expected):
    torch.testing.assert_close(actual.z, expected.z)
    torch.testing.assert_close(actual.x, expected.x)
    assert actual.action == expected.action
    if expected.prior is None:
        assert actual.prior is None
    else:
        assert actual.prior == pytest.approx(expected.prior, abs=1e-12)
    assert actual.n == expected.n
    assert actual.mean == pytest.approx(expected.mean, abs=1e-12)
    if expected.value is None:
        assert actual.value is None
    else:
        assert actual.value == pytest.approx(expected.value, abs=1e-12)
    if expected.shift is None:
        assert actual.shift is None
    else:
        torch.testing.assert_close(actual.shift, expected.shift)
    assert len(actual.children) == len(expected.children)
    for child, reference in zip(actual.children, expected.children, strict=True):
        assert_tree(child, reference)


def test_several_decisions_match_independent_simulation_oracle():
    real = latent_history([[0.1, -0.2], [0.3, 0.4]], actions=3)
    actual = Tree(real)
    expected = RefTree(real)
    mask = np.array([True, False, True])
    alpha = torch.tensor([1.0, 2.0], dtype=torch.float64)
    cfg = config(simulations=4, depth=2, width=2, c_puct=0.7, temperature=1.5)
    rng = np.random.default_rng(91)
    expected_rng = np.random.default_rng(91)
    for _ in range(3):
        policy, preference = networks()
        ref_policy, ref_preference = networks()
        result = decide(actual, mask, policy, preference, alpha, cfg, rng, float("inf"))
        action, probabilities = ref_decide(
            expected, mask, ref_policy, ref_preference, alpha, cfg, expected_rng
        )
        assert result.action == action
        np.testing.assert_allclose(
            result.probabilities, probabilities, atol=1e-12, rtol=0
        )
        assert_tree(actual.root, expected.root)
        assert len(policy.readout_calls) == len(ref_policy.readout_calls)
        assert len(preference.calls) == len(ref_preference.calls)
