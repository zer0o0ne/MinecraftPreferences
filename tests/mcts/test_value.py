"""Leaf value and caching against specs/mcts.md (2)."""

import math

import numpy as np
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


def test_leaf_value_compares_working_history_with_parent():
    tree = Tree(latent_history([[1.0], [1.5]]))
    child = node([2.0], x=[99.0], action=2, prior=1.0, parent=tree.root)
    tree.root.children = [child]
    preference = FakePreference((math.log(3.0), math.log(3.0)))
    decide(
        tree,
        np.array([True, True, True, True]),
        UnexpectedCalls(),
        preference,
        torch.ones(1, dtype=torch.float64),
        config(min_p=0.25),
        np.random.default_rng(0),
    )
    # sigmoid(log 3) = 3/4 for both outputs, hence V = (3/4 + 1/4)(2*3/4 - 1) = 1/2.
    assert child.value == 0.5
    assert child.n == 1
    assert child.mean == 0.45
    assert len(preference.calls) == 1
    child_history, parent_history = preference.calls[0]
    torch.testing.assert_close(
        child_history[0, -1], torch.tensor([2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(parent_history, tree.base.Y)
    assert child_history.shape[1] == tree.base.Y.shape[1] + 1
    assert preference.grad_modes == [False]


def test_cached_leaf_value_is_reused_without_network_calls():
    tree = Tree(latent_history([[0.0]]))
    child = node([1.0], action=0, prior=1.0, parent=tree.root)
    tree.root.children = [child]
    preference = FakePreference((0.0, math.log(3.0)))
    for _ in range(2):
        decide(
            tree,
            np.ones(4, dtype=bool),
            UnexpectedCalls(),
            preference,
            torch.ones(1, dtype=torch.float64),
            config(simulations=2),
            np.random.default_rng(0),
        )
        assert len(preference.calls) == 1
    assert child.n == 4
    assert child.mean == child.value


def test_root_value_field_is_ignored_and_root_is_not_backed_up():
    tree = Tree(latent_history([[0.0]]))
    tree.root.value = 123.0
    policy = FakePolicy(logits=[0.0, 0.0, 0.0, 0.0], shifts=torch.zeros(4, 1))
    preference = UnexpectedCalls()
    decide(
        tree,
        np.ones(4, dtype=bool),
        policy,
        preference,
        torch.ones(1, dtype=torch.float64),
        config(),
        np.random.default_rng(0),
    )
    assert preference.calls == 0
    assert tree.root.value == 123.0
    assert (tree.root.n, tree.root.mean) == (0, 0.0)
    assert all(mode is False for mode in policy.grad_modes)
