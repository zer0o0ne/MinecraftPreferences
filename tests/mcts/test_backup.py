"""Hand-computed PUCT selection and discounted backup, specs/mcts.md (4), (5)."""

import math

import numpy as np
import torch

from mcpref.mcts.search import decide
from mcpref.mcts.tree import Tree

from .helpers import FakePreference, UnexpectedCalls, config, latent_history, node


def choose(tree, children, cfg):
    tree.root.children = children
    preference = FakePreference((0.0, math.log(3.0)))
    decide(
        tree,
        np.ones(4, dtype=bool),
        UnexpectedCalls(),
        preference,
        torch.ones(1, dtype=torch.float64),
        cfg,
        np.random.default_rng(0),
    )
    return preference


def test_puct_uses_all_child_counts_and_zero_q_for_unvisited():
    tree = Tree(latent_history([[0.0]]))
    children = [
        node([1.0], action=0, prior=0.2, parent=tree.root, n=2, mean=0.1),
        node([2.0], action=1, prior=0.5, parent=tree.root, n=0),
        node([3.0], action=2, prior=0.3, parent=tree.root, n=1, mean=0.2),
    ]
    preference = choose(tree, children, config(c_puct=1.0))
    # Scores are .1+.2*sqrt(3)/3, .5*sqrt(3), .2+.3*sqrt(3)/2; child 1 wins.
    assert children[1].n == 1
    assert [child.n for child in children] == [2, 1, 1]
    assert preference.calls[0][0][0, -1].item() == 2.0


def test_puct_tie_goes_to_earliest_expansion_order():
    tree = Tree(latent_history([[0.0]]))
    children = [
        node([1.0], action=2, prior=0.5, parent=tree.root, n=1, mean=0.25),
        node([2.0], action=1, prior=0.5, parent=tree.root, n=1, mean=0.25),
    ]
    preference = choose(tree, children, config(c_puct=2.0))
    assert [child.n for child in children] == [2, 1]
    assert preference.calls[0][0][0, -1].item() == 1.0


def test_backup_discounted_running_means_along_whole_path():
    tree = Tree(latent_history([[0.0]]))
    first = node([1.0], action=0, prior=1.0, parent=tree.root, n=1, mean=0.2)
    leaf = node([2.0], x=[20.0], action=1, prior=1.0, parent=first, n=2, mean=0.1)
    first.children = [leaf]
    tree.root.children = [first]
    preference = FakePreference((math.log(3.0), math.log(3.0)))
    decide(
        tree,
        np.ones(4, dtype=bool),
        UnexpectedCalls(),
        preference,
        torch.ones(1, dtype=torch.float64),
        config(depth=2, discount=0.5, min_p=0.25),
        np.random.default_rng(0),
    )
    # V=1/2 at depth 2: the first edge gets .5**2*V=.125, the second .5*V=.25.
    assert (first.n, leaf.n) == (2, 3)
    assert first.mean == 0.2 + (0.125 - 0.2) / 2
    assert leaf.mean == 0.1 + (0.25 - 0.1) / 3
    assert tree.root.n == 0
