"""Cached change predictions describe the actual MCTS draw without new readouts."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diagnostics.core.world import config as diagnostics_config
from diagnostics.core.world import context

from mcpref.diagnostics.producers.mcts import change_metrics
from mcpref.runtime.mcts.search import decide, expand
from mcpref.runtime.mcts.tree import Tree, advance
from mcpref.runtime.sampling.codecs import _decode_tree, _encode_tree
from mcpref.runtime.sampling.runner import Sampler
from mcpref.runtime.sampling.shards import load_step
from runtime.mcts.helpers import FakePolicy, FakePreference, config, latent_history


def metrics(root, action):
    record = SimpleNamespace(tree={"root": root}, action=action)
    ctx = context("trajectory", "mcts", steps=(1,), clusters=("episode",))
    return change_metrics(record, ctx, diagnostics_config())


@pytest.mark.parametrize("action,expected", [(0, 1), (1, 0)])
def test_metric_uses_actual_draw_not_policy_or_visit_argmax(action, expected):
    result, populations = metrics({"change_probability": 0.5, "action": 0}, action)
    value = result["mcts.repeat_when_change_predicted"]
    assert (value["numerator"], value["n"], value["value"]) == (expected, 1, expected)
    assert populations["mcts.repeat_when_change_predicted"] == ["episode"]


@pytest.mark.parametrize(
    "root,reason",
    [
        ({"action": 0}, "change_head_unavailable"),
        ({"action": None, "change_probability": 0.8}, "no_previous_action"),
        ({"action": 0, "change_probability": 0.2}, None),
    ],
)
def test_missing_or_negative_condition_is_not_a_zero_repeat(root, reason):
    result, populations = metrics(root, 0)
    value = result["mcts.repeat_when_change_predicted"]
    assert value["value"] is None and value["n"] == 0 and value["reason"] == reason
    assert populations["mcts.repeat_when_change_predicted"] == []


@pytest.mark.parametrize("actual", [0, 1])
def test_cached_head_survives_reroot_even_via_another_action(actual):
    history = latent_history([[0.0, 0.0]], actions=2)
    tree = Tree(history)
    pi = FakePolicy(logits=[2.0, 0.0], shifts=[[0.0, 0.0], [1.0, 0.0]])
    calls = []

    def change(readout):
        calls.append(readout.clone())
        return readout.new_tensor([0.8])

    pi.change = change
    cfg, mask, alpha = config(width=2), np.ones(2, dtype=bool), torch.ones(2)
    with torch.no_grad():
        expand(tree, tree.root, mask, pi, alpha, cfg, history)
        child = tree.root.children[0]
        expand(tree, child, mask, pi, alpha, cfg, history)
    assert len(pi.readout_calls) == len(pi.logit_calls) == len(calls) == 2
    for head, logits in zip(calls, pi.logit_calls):
        torch.testing.assert_close(head, logits)
    factual = latent_history([[0.0, 0.0]], [[int(actual == 0), int(actual == 1), 0]])
    advance(tree, actual, factual, alpha, cfg.epsilon)
    assert tree.root is child and child.action == actual
    snapshot = _encode_tree(tree)
    restored = _decode_tree(snapshot, torch.device("cpu"))
    assert restored.root.change_probability == pytest.approx(0.8)
    result, _ = metrics(snapshot["root"], actual)
    assert result["mcts.repeat_when_change_predicted"]["value"] == 1
    del snapshot["root"]["change_probability"]
    assert _decode_tree(snapshot, torch.device("cpu")).root.change_probability is None


def test_change_head_does_not_change_fixed_budget_draws_or_rng():
    results = []
    for probability in (0.1, 0.9):
        tree = Tree(latent_history([[0.0, 0.0]], actions=2))
        pi = FakePolicy(logits=[2.0, 1.0], shifts=[[0.0, 0.0], [1.0, 0.0]])
        pi.change = lambda R, p=probability: R.new_full(R.shape[:-1], p)
        rng = np.random.default_rng(42)
        decision = decide(
            tree,
            np.ones(2, dtype=bool),
            pi,
            FakePreference(),
            torch.ones(2),
            config(width=2, simulations=5),
            rng,
            float("inf"),
        )
        results.append((decision, rng.bit_generator.state, len(pi.readout_calls)))
    assert results[0][0].action == results[1][0].action
    np.testing.assert_array_equal(
        results[0][0].probabilities, results[1][0].probabilities
    )
    assert results[0][1:] == results[1][1:]


def test_sampling_shard_boundary_keeps_previous_action_and_cached_head(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    sampler.sample()
    records = [load_step(site.out, i) for i in range(4)]
    assert records[0].tree["root"]["action"] is None
    for before, after in pairwise(records):
        assert after.tree["root"]["action"] == before.action
        assert after.tree["root"]["change_probability"] == 0.5
