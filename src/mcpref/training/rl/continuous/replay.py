"""One item-conditioned recurrent trajectory; specs/rl-continuous.md."""

import torch

from mcpref.config import REPO
from mcpref.models.perception.continuous import observation_step
from mcpref.models.perception.continuous.state import detach_state
from mcpref.runtime.mcts.tree import LatentHistory
from mcpref.runtime.sampling.anchors import read_anchor
from mcpref.runtime.sampling.history import append
from mcpref.runtime.sampling.shards import read_observation


def branch(
    phi, sampling, states, action_count, continuous, t, *, fresh=False, horizon=0
):
    """Return origin/successor graphs and detached reward-only continuation."""
    g = max(0, t - continuous.gradient_history + 1)
    r = max(0, t - max(states, continuous.gradient_history) + 1)
    a = r // continuous.replay_stride * continuous.replay_stride
    out = REPO / sampling.out_dir
    device = next(phi.parameters()).device
    state = read_anchor(out, a, sampling.steps, phi, device, continuous.replay_stride)
    if state is not None:
        state = detach_state(state)
    grad = torch.is_grad_enabled()
    history, encoded, rewards = None, [], []
    last_grad = t + int(fresh)
    for index in range(a, t + max(int(fresh), horizon) + 1):
        if index == last_grad + 1:
            state = detach_state(state)
            history = LatentHistory(history.Y.detach(), history.C.detach())
        observation, action = read_observation(out, index)
        with torch.set_grad_enabled(grad and g <= index <= last_grad):
            z, state = observation_step(phi, observation.frame, observation.line, state)
            history = append(history, z, action, action_count, states)
        if t <= index <= last_grad:
            encoded.append(history)
        elif index > last_grad:
            rewards.append(history)
    return encoded, rewards
