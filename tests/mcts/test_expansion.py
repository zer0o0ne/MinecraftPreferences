"""Expansion against specs/mcts.md (3): masking, order, priors, states and depth."""

import math

import numpy as np
import torch

from mcpref.mcts.search import decide
from mcpref.mcts.tree import Tree

from .helpers import (
    FakePolicy,
    UnexpectedCalls,
    assert_values,
    config,
    latent_history,
    node,
)


def run_root(policy, mask, cfg):
    tree = Tree(latent_history([[0.0, 0.0]]))
    decide(
        tree,
        np.array(mask),
        policy,
        UnexpectedCalls(),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        cfg,
        np.random.default_rng(0),
        float("inf"),
    )
    return tree


def softmax_oracle(logits):
    weights = [math.exp(value - max(logits)) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def test_top_l_root_mask_tie_order_and_unnormalised_priors():
    logits = [3.0, 100.0, 2.0, 2.0]
    policy = FakePolicy(logits=logits, shifts=torch.zeros(4, 2))
    tree = run_root(policy, [True, False, True, True], config(width=2))
    assert [child.action for child in tree.root.children] == [0, 2]
    probabilities = softmax_oracle(logits)
    np.testing.assert_allclose(
        [child.prior for child in tree.root.children],
        [probabilities[0], probabilities[2]],
        rtol=1e-14,
        atol=0,
    )
    assert sum(child.prior for child in tree.root.children) < 1.0
    assert (
        len(policy.readout_calls)
        == len(policy.logit_calls)
        == len(policy.shift_calls)
        == 1
    )
    R, actions = policy.shift_calls[0]
    assert R.shape == (2, 2)
    assert_values(actions, torch.tensor([0, 2]))


def test_expands_every_allowed_action_when_fewer_than_width():
    policy = FakePolicy(logits=[0.0, 1.0, 0.0, 3.0], shifts=torch.zeros(4, 2))
    tree = run_root(policy, [False, True, False, True], config(width=4))
    assert [child.action for child in tree.root.children] == [3, 1]


def test_expansion_uses_quantised_working_and_unrounded_exact_states():
    shifts = torch.tensor(
        [[0.25, 0.125], [0.0, 0.0], [0.75, 0.375], [0.0, 0.0]],
        dtype=torch.float64,
    )
    policy = FakePolicy(logits=[3.0, 100.0, 2.0, 2.0], shifts=shifts)
    tree = run_root(policy, [True, False, True, True], config(width=2, epsilon=0.5))
    first, second = tree.root.children
    assert_values(first.shift, torch.zeros(2))
    assert_values(first.z, torch.zeros(2))
    assert_values(first.x, torch.tensor([0.25, 0.125]))
    # u/epsilon = 1.5 in both coordinates rounds to 2; alpha=(1,2) gives q=(1,.5).
    assert_values(second.shift, torch.tensor([1.0, 0.5]))
    assert_values(second.z, torch.tensor([1.0, 0.5]))
    assert_values(second.x, torch.tensor([0.75, 0.375]))


def test_imagined_expansion_ignores_root_mask():
    tree = Tree(latent_history([[0.0, 0.0]]))
    child = node([1.0, 1.0], action=0, prior=1.0, parent=tree.root, value=0.5)
    tree.root.children = [child]
    policy = FakePolicy(logits=[0.0, 1.0, 2.0, 3.0], shifts=torch.zeros(4, 2))
    decide(
        tree,
        np.array([True, False, False, False]),
        policy,
        UnexpectedCalls(),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        config(depth=2),
        np.random.default_rng(0),
        float("inf"),
    )
    assert [grandchild.action for grandchild in child.children] == [3, 2, 1, 0]
    Y, C = policy.readout_calls[0]
    assert_values(Y[0, -1], child.z)
    assert C.shape[1] == tree.base.C.shape[1] + 1


def test_depth_limit_evaluates_but_never_expands():
    tree = Tree(latent_history([[0.0, 0.0]]))
    child = node([1.0, 1.0], action=0, prior=1.0, parent=tree.root, value=0.5)
    tree.root.children = [child]
    policy = UnexpectedCalls()
    decide(
        tree,
        np.ones(4, dtype=bool),
        policy,
        UnexpectedCalls(),
        torch.ones(2, dtype=torch.float64),
        config(depth=1),
        np.random.default_rng(0),
        float("inf"),
    )
    assert policy.calls == 0
    assert child.children == []
