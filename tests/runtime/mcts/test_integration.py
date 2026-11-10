"""Several complete decisions against the independent simulation oracle."""

import numpy as np
import pytest
import torch

from mcpref.runtime.mcts import search
from mcpref.runtime.mcts.search import decide
from mcpref.runtime.mcts.tree import Tree
from runtime.mcts.helpers import (
    FakePolicy,
    FakePreference,
    UnexpectedCalls,
    config,
    latent_history,
    node,
)
from runtime.mcts.search_oracle import RefNode, RefTree, ref_decide, ref_history


def networks(history_pi=1000, history_P=1000):
    base = torch.tensor(
        [[0.03, -0.02], [-0.01, 0.04], [0.02, 0.01]], dtype=torch.float64
    )

    def logits(R):
        return torch.stack(
            (R[..., 0], R[..., 1], 0.2 * (R[..., 0] - R[..., 1])), dim=-1
        )

    def preference(Ya, Yb):
        return Ya[0, -1, 0] - Yb[0, -1, 0], 0.3 * (Ya[0, -1, 1] + Yb[0, -1, 1])

    policy = FakePolicy(
        logits_fn=logits, shift_fn=lambda R, a: base[a] + 0.01 * R, history=history_pi
    )
    return policy, FakePreference(logits_fn=preference, history=history_P)


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


def test_long_model_windows_match_independent_full_history_simulation():
    real = latent_history([[0.1 * i, -0.2 * i] for i in range(11)], actions=3)
    actual, expected = Tree(real), RefTree(real)
    actual_parent, expected_parent = actual.root, expected.root
    for i in range(7):
        z = torch.tensor([0.3 + i, -0.4 - i], dtype=torch.float64)
        x = torch.tensor([0.31 + i, -0.39 - i], dtype=torch.float64)
        actual_child = node(z, x=x, action=i % 3, prior=1.0, parent=actual_parent)
        expected_child = RefNode(
            z.clone(), x.clone(), action=i % 3, prior=1.0, parent=expected_parent
        )
        actual_parent.children = [actual_child]
        expected_parent.children = [expected_child]
        actual_parent, expected_parent = actual_child, expected_child
    actual_leaf = actual_parent
    full_before = ref_history(actual, actual_leaf)
    base_before = (real.Y.clone(), real.C.clone())
    policy, preference = networks(history_pi=3, history_P=2)
    ref_policy, ref_preference = networks(history_pi=3, history_P=2)
    cfg = config(simulations=6, depth=10, width=2, c_puct=0.7)
    rng, expected_rng = np.random.default_rng(37), np.random.default_rng(37)

    result = decide(
        actual,
        np.array([True, False, True]),
        policy,
        preference,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        cfg,
        rng,
        float("inf"),
    )
    action, probabilities = ref_decide(
        expected,
        np.array([True, False, True]),
        ref_policy,
        ref_preference,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        cfg,
        expected_rng,
    )

    assert result.action == action
    np.testing.assert_allclose(result.probabilities, probabilities, atol=0, rtol=0)
    assert_tree(actual.root, expected.root)
    for calls, reference_calls in (
        (policy.readout_calls, ref_policy.readout_calls),
        (preference.calls, ref_preference.calls),
    ):
        for inputs, reference_inputs in zip(calls, reference_calls, strict=True):
            for value, reference in zip(inputs, reference_inputs, strict=True):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for value, reference in zip(
        policy.logit_calls, ref_policy.logit_calls, strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    full_after = ref_history(actual, actual_leaf)
    for value, reference in zip(full_after, full_before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert actual.base is real
    torch.testing.assert_close(real.Y, base_before[0], rtol=0, atol=0)
    torch.testing.assert_close(real.C, base_before[1], rtol=0, atol=0)
    assert rng.random() == expected_rng.random()


def test_leaf_evaluation_and_expansion_share_one_model_history(monkeypatch):
    tree = Tree(latent_history([[float(i), -float(i)] for i in range(12)], actions=3))
    parent = tree.root
    for i in range(6):
        child = node([i + 20.0, i - 20.0], action=i % 3, prior=1.0, parent=parent)
        parent.children = [child]
        parent = child
    policy, preference = networks(history_pi=3, history_P=2)
    original = search.model_history
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr("mcpref.runtime.mcts.search.model_history", counted)
    decide(
        tree,
        np.ones(3, dtype=bool),
        policy,
        preference,
        torch.ones(2, dtype=torch.float64),
        config(depth=8, width=2),
        np.random.default_rng(0),
        float("inf"),
    )

    assert calls == 1
    assert len(policy.readout_calls) == len(preference.calls) == 1
