"""Real latent/action histories and result-first preferences of the spec."""

import torch
from torch import Tensor

from mcpref.env.state import EnvState
from mcpref.mcts.tree import LatentHistory
from mcpref.perception.stream import PhiStream
from mcpref.preferences.network import Preference


def append(
    history: LatentHistory | None,
    z: Tensor,
    action: int | None,
    actions: int,
    states: int,
) -> LatentHistory:
    """Notation: append z and its preceding-action row, then retain the last L rows."""
    if history is None:
        Y = z.reshape(1, 1, -1)
    else:
        Y = torch.cat((history.Y, z.reshape(1, 1, -1)), dim=1)
    C = Y.new_zeros((1, 1, actions + 1))
    C[0, 0, -1 if action is None else action] = 1
    if history is not None:
        C = torch.cat((history.C, C), dim=1)
    return LatentHistory(Y[:, -states:], C[:, -states:])


def encode(
    stream: PhiStream,
    history: LatentHistory | None,
    raw: list[EnvState],
    preceding: list[int | None],
    actions: int,
    states: int,
) -> LatentHistory:
    """(3): encode every not-yet-encoded raw state and append its real history row."""
    for observation, action in zip(raw, preceding):
        z = stream.step(observation.frame, observation.line)
        history = append(history, z, action, actions, states)
    return history


def preference_pair(result: tuple[Tensor, Tensor]) -> dict[str, float]:
    """(6): the two logits, without a sigmoid."""
    succ, inc = result
    return {"succ": succ.item(), "inc": inc.item()}


def preferences(
    P: Preference,
    current: LatentHistory,
    origin: LatentHistory,
    start: LatentHistory,
) -> tuple[dict[str, float], dict[str, float]]:
    """(6): P(Y_t+1,Y_t), then P(Y_t+1,Y_tau), under no gradients."""
    with torch.no_grad():
        close = preference_pair(P(current.Y, origin.Y))
        trajectory = preference_pair(P(current.Y, start.Y))
    return close, trajectory
