"""Conditional values, truncated derivatives and branch boundaries of continuous RL."""

import pytest
import torch

from training.rl.continuous.world import World


@pytest.mark.parametrize(
    "t,length,k,anchor,first,fresh,horizon",
    [
        (10, 3, 2, 8, 9, False, 3),
        (10, 5, 2, 4, 9, False, 3),
        (10, 2, 5, 4, 6, False, 3),
        (0, 4, 4, 0, 0, True, 1),
        (4, 3, 1, 0, 4, True, 4),
        (11, 3, 2, 8, 10, True, 4),
        (11, 2, 4, 8, 8, True, 4),
    ],
)
def test_values_actions_and_local_gradients_match_closed_form(
    monkeypatch, tmp_path, t, length, k, anchor, first, fresh, horizon
):
    world = World(monkeypatch, tmp_path, k=k, length=length)
    differentiable, rewards = world.branch(t, fresh=fresh, horizon=horizon)
    endpoints = [t, t + 1] if fresh else [t]
    assert len(differentiable) == len(endpoints)
    future_indices = list(range(t + (2 if fresh else 1), t + horizon + 1))
    assert len(rewards) == len(future_indices)
    for history, endpoint in zip(differentiable + rewards, endpoints + future_indices):
        torch.testing.assert_close(history.Y, world.expected_y(anchor, endpoint))
        assert torch.equal(history.C, world.expected_c(endpoint))
        assert history.Y.dtype == history.C.dtype == torch.float64
    assert all(not h.Y.requires_grad and h.Y.grad_fn is None for h in rewards)
    assert world.anchor_reads == [anchor]
    assert world.observation_reads == list(
        range(anchor, t + max(horizon, int(fresh)) + 1)
    )
    grad_indices = list(range(first, t + int(fresh) + 1))
    assert [call.index for call in world.phi.calls if call.grad] == grad_indices
    assert all(call.saved == 0 for call in world.phi.calls if not call.grad)
    assert all(call.origin == world.origin for call in world.phi.calls)
    assert not next(
        call for call in world.phi.calls if call.index == first
    ).incoming_grad
    weights = [(endpoint, float(i + 1)) for i, endpoint in enumerate(endpoints)]
    sum(
        weight * history.Y.sum()
        for history, (_, weight) in zip(differentiable, weights)
    ).backward()
    dtheta, dx = world.gradients(anchor, first, weights)
    torch.testing.assert_close(
        world.phi.head[0].weight.grad, torch.tensor([[dtheta]], dtype=torch.float64)
    )
    torch.testing.assert_close(world.phi.inputs.grad, dx)
    world.unchanged()


def test_successor_and_rewards_never_switch_anchor_at_grid_or_cycle_boundary(
    monkeypatch, tmp_path
):
    world = World(monkeypatch, tmp_path)
    differentiable, rewards = world.branch(11, fresh=True, horizon=4)
    assert world.anchor_reads == [8]
    assert world.observation_reads == list(range(8, 16))
    assert [call.index for call in world.phi.calls if call.grad] == [10, 11, 12]
    for history, endpoint in zip(differentiable + rewards, range(11, 16)):
        torch.testing.assert_close(history.Y, world.expected_y(8, endpoint))
    assert all(not history.Y.requires_grad for history in rewards)
    assert not next(call for call in world.phi.calls if call.index == 13).incoming_grad
    world.unchanged()


def test_successor_is_item_conditioned_not_independently_reencoded(
    monkeypatch, tmp_path
):
    world = World(monkeypatch, tmp_path)
    earlier, _ = world.branch(13, fresh=True)
    independent, _ = world.branch(14)
    assert world.anchor_reads == [8, 12]
    torch.testing.assert_close(earlier[1].Y, world.expected_y(8, 14))
    torch.testing.assert_close(independent[0].Y, world.expected_y(12, 14))
    assert not torch.equal(earlier[1].Y, independent[0].Y)
    world.unchanged()


@pytest.mark.parametrize("frozen", [False, True])
def test_no_grad_and_frozen_phi_do_not_force_graph_or_change_module_mode(
    monkeypatch, tmp_path, frozen
):
    world = World(monkeypatch, tmp_path)
    world.phi.train()
    if frozen:
        world.phi.requires_grad_(False)
        encoded, rewards = world.branch(10, fresh=True, horizon=4)
    else:
        with torch.no_grad():
            encoded, rewards = world.branch(10, fresh=True, horizon=4)
        assert not any(call.grad for call in world.phi.calls)
    assert world.phi.training
    assert all(not history.Y.requires_grad for history in encoded + rewards)
    assert all(call.saved == 0 for call in world.phi.calls)
    world.unchanged()
