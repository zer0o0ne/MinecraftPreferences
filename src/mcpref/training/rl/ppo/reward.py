"""(1), (2) of specs/ppo.md: the H-step preference advantage."""

from collections.abc import Sequence

import torch
from torch import Tensor

from mcpref.models.preference.network import Preference


def reward(
    P: Preference,
    origins: Sequence[Tensor],
    futures: Sequence[Sequence[Tensor]],
    horizon: int,
    discount: float,
    min_p: float,
) -> Tensor:
    """(1), (2): each origin's complete, discounted preference advantage, shape (m,)."""
    with torch.no_grad():
        groups = {}
        for row, (origin, consequences) in enumerate(zip(origins, futures)):
            for k, future in enumerate(consequences, start=1):
                key = (future.shape[1], origin.shape[1])
                groups.setdefault(key, []).append((row, k, future, origin))
        values = {}
        for group in groups.values():
            consequences = torch.cat([pair[2] for pair in group], dim=0)
            origins_batch = torch.cat([pair[3] for pair in group], dim=0)
            s, u = P(consequences, origins_batch)
            group_values = (torch.sigmoid(u) + min_p) * (2 * torch.sigmoid(s) - 1)
            for (row, k, _, _), value in zip(group, group_values):
                values[row, k] = discount**k * value
        advantages = [
            sum(values[row, k] for k in range(1, len(consequences) + 1))
            for row, consequences in enumerate(futures)
        ]
        return torch.stack(advantages, dim=0)
