"""Visit-count decision distribution and caller RNG against specs/mcts.md (6)."""

import dataclasses

import numpy as np
import torch

from mcpref.mcts.search import Decision, decide
from mcpref.mcts.tree import Tree

from .helpers import FakePolicy, UnexpectedCalls, config, latent_history, node


def test_temperature_full_probability_vector_seeded_draw_and_rng_advance():
    assert [field.name for field in dataclasses.fields(Decision)] == [
        "action",
        "probabilities",
    ]
    tree = Tree(latent_history([[0.0]]))
    first = node([1.0], action=1, prior=0.5, parent=tree.root, n=1, mean=0.0, value=0.0)
    second = node(
        [2.0], action=3, prior=0.5, parent=tree.root, n=3, mean=1.0, value=1.0
    )
    tree.root.children = [first, second]
    rng = np.random.default_rng(17)
    oracle_rng = np.random.default_rng(17)
    result = decide(
        tree,
        np.ones(4, dtype=bool),
        UnexpectedCalls(),
        UnexpectedCalls(),
        torch.ones(1, dtype=torch.float64),
        config(c_puct=0.0, temperature=2.0),
        rng,
    )
    # c=0 selects mean=1, so counts become (1,4); square-root weights are (1,2).
    expected_p = np.array([1.0 / 3.0, 2.0 / 3.0])
    expected_index = oracle_rng.choice(2, p=expected_p, replace=True, size=None)
    assert result.action == [1, 3][expected_index]
    assert result.probabilities.shape == (4,)
    assert result.probabilities.dtype == np.float64
    np.testing.assert_allclose(result.probabilities, [0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0])
    assert rng.random() == oracle_rng.random()


def test_all_zero_visit_weights_fall_back_to_uniform_children():
    tree = Tree(latent_history([[0.0, 0.0]]))
    policy = FakePolicy(logits=[0.0] * 4, shifts=torch.zeros(4, 2))
    rng = np.random.default_rng(23)
    oracle_rng = np.random.default_rng(23)
    result = decide(
        tree,
        np.ones(4, dtype=bool),
        policy,
        UnexpectedCalls(),
        torch.ones(2, dtype=torch.float64),
        config(),
        rng,
    )
    assert [child.n for child in tree.root.children] == [0, 0, 0, 0]
    np.testing.assert_allclose(result.probabilities, np.full(4, 0.25))
    expected = oracle_rng.choice(4, p=np.full(4, 0.25), replace=True, size=None)
    assert result.action == expected
    assert rng.random() == oracle_rng.random()
