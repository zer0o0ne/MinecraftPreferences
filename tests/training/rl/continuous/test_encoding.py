"""Mixed-index dispatch preserves independent preference gradients and fixed advantages."""

from types import SimpleNamespace

import pytest
import torch

from mcpref.training.rl.learning.encoding import micro_histories
from mcpref.training.rl.learning.history import eval_encoder
from mcpref.training.rl.learning.retrain import advantage_histories
from mcpref.training.rl.pairs.types import StateRef
from training.rl.continuous.world import World


def preference(a, b):
    return {
        "kind": "preference",
        "a": {"episode": f"trajectory#{a // 12:06d}", "index": a},
        "b": {"episode": f"trajectory#{b // 12:06d}", "index": b},
        "labels": {"p": 0.8, "inc": 0},
    }


def encoded(world, items, fixed=False):
    gradient, rewards = micro_histories(
        items,
        world.phi,
        None,
        world.sampling,
        world.length,
        3,
        SimpleNamespace(horizon=4),
        fixed,
        continuous=world.cfg,
    )
    return list(gradient), list(rewards)


def test_both_preference_endpoints_have_their_own_local_phi_gradient(
    monkeypatch, tmp_path
):
    world = World(monkeypatch, tmp_path)
    histories, rewards = encoded(world, [preference(3, 14)])
    assert len(histories) == 2 and rewards == []
    assert world.anchor_reads == [0, 12]
    assert [call.index for call in world.phi.calls if call.grad] == [2, 3, 13, 14]
    for history, anchor, endpoint, first in zip(histories, [0, 12], [3, 14], [2, 13]):
        torch.testing.assert_close(history.Y, world.expected_y(anchor, endpoint))
        derivative, inputs = torch.autograd.grad(
            history.Y.sum(), (world.phi.head[0].weight, world.phi.inputs)
        )
        dtheta, dx = world.gradients(anchor, first, [(endpoint, 1.0)])
        torch.testing.assert_close(
            derivative, torch.tensor([[dtheta]], dtype=torch.float64)
        )
        torch.testing.assert_close(inputs, dx)
        assert derivative.abs().sum() > 0
    world.unchanged()


@pytest.mark.parametrize("fixed", [False, True])
def test_mixed_order_eligibility_and_fixed_advantages_control_only_reward_branches(
    monkeypatch, tmp_path, fixed
):
    world = World(monkeypatch, tmp_path)
    items = [
        preference(3, 14),
        {"kind": "pending", "index": 11},
        {"kind": "fresh", "index": 4},
        {"kind": "fresh", "index": 11},
    ]
    histories, rewards = encoded(world, items, fixed)
    assert world.anchor_reads == [0, 12, 8, 0, 8]
    assert len(histories) == 7
    for history, (anchor, endpoint) in zip(
        histories, [(0, 3), (12, 14), (8, 11), (0, 4), (0, 5), (8, 11), (8, 12)]
    ):
        torch.testing.assert_close(history.Y, world.expected_y(anchor, endpoint))
        assert history.Y.requires_grad
    expected = (
        [] if fixed else [(8, t) for t in range(12, 16)] + [(0, t) for t in range(6, 9)]
    )
    assert len(rewards) == len(expected)
    for history, (anchor, endpoint) in zip(rewards, expected):
        torch.testing.assert_close(history.Y, world.expected_y(anchor, endpoint))
        assert not history.Y.requires_grad and history.Y.grad_fn is None
    assert [call.index for call in world.phi.calls if call.grad] == [
        2,
        3,
        13,
        14,
        10,
        11,
        3,
        4,
        5,
        10,
        11,
        12,
    ]
    assert items[0]["labels"] == {"p": 0.8, "inc": 0}
    world.unchanged()


def test_eval_encoder_uses_endpoint_anchors_in_order_and_restores_mode(
    monkeypatch, tmp_path
):
    world = World(monkeypatch, tmp_path)
    world.phi.train()
    encode = eval_encoder(
        world.phi, None, world.sampling, world.length, 3, continuous=world.cfg
    )
    refs = [StateRef("trajectory#000001", 14), StateRef("trajectory#000000", 3)]
    histories = encode(refs)
    assert world.anchor_reads == [12, 0]
    assert world.phi.training and torch.is_grad_enabled()
    assert not any(call.training or call.grad for call in world.phi.calls)
    for history, (anchor, endpoint) in zip(histories, [(12, 14), (0, 3)]):
        torch.testing.assert_close(history, world.expected_y(anchor, endpoint))
        assert not history.requires_grad
    world.unchanged()


@pytest.mark.parametrize("kind,t,anchor", [("fresh", 7, 4), ("pending", 11, 8)])
def test_frozen_advantage_inputs_keep_the_same_branch_through_all_futures(
    monkeypatch, tmp_path, kind, t, anchor
):
    world = World(monkeypatch, tmp_path)
    context = SimpleNamespace(perception=None, states=world.length, replay=world.cfg)
    modules = {"phi": world.phi, "pi": SimpleNamespace(names=["a", "b", "c"])}
    with torch.no_grad():
        origin, futures = advantage_histories(
            {"kind": kind, "index": t},
            modules,
            world.sampling,
            context,
            SimpleNamespace(horizon=4),
        )
    assert world.anchor_reads == [anchor]
    assert len(futures) == 4
    assert not any(call.grad for call in world.phi.calls)
    torch.testing.assert_close(origin.Y, world.expected_y(anchor, t))
    for history, endpoint in zip(futures, range(t + 1, t + 5)):
        torch.testing.assert_close(history, world.expected_y(anchor, endpoint))
        assert not history.requires_grad
    world.unchanged()
