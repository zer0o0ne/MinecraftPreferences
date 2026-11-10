"""Visit-count decision distribution and caller RNG against specs/mcts.md (6)."""

import dataclasses

import numpy as np
import torch

from mcpref.runtime.mcts.search import Decision, decide
from mcpref.runtime.mcts.tree import Tree
from runtime.mcts.helpers import (
    FakePolicy,
    UnexpectedCalls,
    config,
    latent_history,
    node,
)


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
        float("inf"),
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
        float("inf"),
    )
    assert [child.n for child in tree.root.children] == [0, 0, 0, 0]
    np.testing.assert_allclose(result.probabilities, np.full(4, 0.25))
    expected = oracle_rng.choice(4, p=np.full(4, 0.25), replace=True, size=None)
    assert result.action == expected
    assert rng.random() == oracle_rng.random()


class ClockPreference:
    history = 1000

    def __init__(self, clock, elapsed=0.0):
        self.clock, self.elapsed, self.calls = clock, elapsed, 0

    def __call__(self, Ya, Yb):
        self.calls += 1
        self.clock[0] += self.elapsed
        return torch.zeros(1), torch.zeros(1)


def decision_tree():
    tree = Tree(latent_history([[0.0]]))
    tree.root.children = [node([1.0], action=0, prior=1.0, parent=tree.root)]
    return tree


def test_expired_deadline_still_completes_minimum_simulations(monkeypatch):
    monkeypatch.setattr("mcpref.runtime.mcts.search.time.perf_counter", lambda: 20.0)
    tree = decision_tree()
    preference = ClockPreference([20.0])
    decide(
        tree,
        np.ones(4, dtype=bool),
        UnexpectedCalls(),
        preference,
        torch.ones(1, dtype=torch.float64),
        config(simulations=5, min_n_sampling=3, depth=1),
        np.random.default_rng(0),
        10.0,
    )
    assert preference.calls == 1
    assert tree.root.children[0].n == 3


def test_deadline_checked_before_simulation_and_running_one_finishes(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "mcpref.runtime.mcts.search.time.perf_counter", lambda: clock[0]
    )
    tree = decision_tree()
    preference = ClockPreference(clock, elapsed=20.0)
    decide(
        tree,
        np.ones(4, dtype=bool),
        UnexpectedCalls(),
        preference,
        torch.ones(1, dtype=torch.float64),
        config(simulations=5, min_n_sampling=1, depth=1),
        np.random.default_rng(0),
        10.0,
    )
    assert preference.calls == 1
    assert tree.root.children[0].n == 1
    assert tree.root.children[0].mean == 0.0


def test_decide_requires_deadline_as_last_argument():
    import inspect

    parameters = list(inspect.signature(decide).parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "tree",
        "mask",
        "pi",
        "P",
        "alpha",
        "cfg",
        "rng",
        "deadline",
    ]
    assert parameters[-1].default is inspect.Parameter.empty
