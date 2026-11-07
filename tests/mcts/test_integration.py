"""Several complete decisions against the independent simulation oracle."""

import numpy as np
import pytest
import torch

from mcpref.mcts.search import decide
from mcpref.mcts.tree import Tree

from .helpers import (
    FakePolicy,
    FakePreference,
    UnexpectedCalls,
    config,
    latent_history,
    node,
)
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


def subtree_state(current):
    return (
        current.z.tolist(),
        current.x.tolist(),
        current.action,
        current.prior,
        None if current.shift is None else current.shift.tolist(),
        current.n,
        current.mean,
        current.value,
        [subtree_state(child) for child in current.children],
    )


def decide_existing(tree, mask):
    return decide(
        tree,
        np.array(mask),
        UnexpectedCalls(),
        UnexpectedCalls(),
        torch.ones(1, dtype=torch.float64),
        config(width=2),
        np.random.default_rng(0),
        float("inf"),
    )


def test_reused_root_forbids_action_and_preserves_its_subtree():
    tree = Tree(latent_history([[0.0]]))
    forbidden = node(
        [1.0], x=[1.1], action=0, prior=0.25, shift=[1.0], parent=tree.root,
        n=7, mean=-0.2, value=0.4,
    )  # fmt: skip
    descendant = node(
        [2.0], x=[2.1], action=1, prior=0.5, shift=[1.0], parent=forbidden,
        n=5, mean=0.1, value=0.3,
    )  # fmt: skip
    forbidden.children = [descendant]
    allowed = node(
        [3.0], action=1, prior=0.75, shift=[3.0], parent=tree.root, n=2, value=0.0
    )
    tree.root.children = [forbidden, allowed]
    before = subtree_state(forbidden)
    result = decide_existing(tree, [False, True])
    assert result.action == 1
    np.testing.assert_array_equal(result.probabilities, [0.0, 1.0])
    assert (forbidden.n, allowed.n) == (7, 3)
    assert forbidden.parent is tree.root
    assert forbidden.children[0] is descendant
    assert subtree_state(forbidden) == before


def test_alternating_masks_reenable_children_without_resetting_statistics():
    tree = Tree(latent_history([[0.0]]))
    children = [
        node([1.0], action=0, prior=0.5, shift=[1.0], parent=tree.root, n=2, value=0.0),
        node([2.0], action=1, prior=0.5, shift=[2.0], parent=tree.root, n=5, value=0.0),
    ]
    tree.root.children = children
    cases = [
        ([True, False], 0, (3, 5)),
        ([False, True], 1, (3, 6)),
        ([True, False], 0, (4, 6)),
    ]
    for mask, action, counts in cases:
        forbidden = children[1 - action]
        before = subtree_state(forbidden)
        result = decide_existing(tree, mask)
        assert result.action == action
        expected = np.zeros(2)
        expected[action] = 1.0
        np.testing.assert_array_equal(result.probabilities, expected)
        assert tuple(child.n for child in children) == counts
        assert subtree_state(forbidden) == before
        assert all(
            actual is original
            for actual, original in zip(tree.root.children, children, strict=True)
        )


def test_several_decisions_match_independent_simulation_oracle():
    real = latent_history([[0.1, -0.2], [0.3, 0.4]], actions=3)
    actual = Tree(real)
    expected = RefTree(real)
    masks = (
        np.array([True, False, True]),
        np.array([False, True, False]),
        np.array([True, False, True]),
    )
    alpha = torch.tensor([1.0, 2.0], dtype=torch.float64)
    cfg = config(simulations=4, depth=2, width=2, c_puct=0.7, temperature=1.5)
    rng = np.random.default_rng(91)
    expected_rng = np.random.default_rng(91)
    for mask in masks:
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
