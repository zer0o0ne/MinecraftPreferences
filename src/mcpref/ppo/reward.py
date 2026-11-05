"""(1), (2) of specs/ppo.md: the H-step preference advantage."""

from collections.abc import Sequence

import torch
from torch import Tensor

from mcpref.preferences.network import Preference


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
        advantages = []
        for origin, consequences in zip(origins, futures):
            values = []
            for k, future in enumerate(consequences, start=1):
                s, u = P(future, origin)
                value = (torch.sigmoid(u[0]) + min_p) * (2 * torch.sigmoid(s[0]) - 1)
                values.append(discount**k * value)
            advantages.append(sum(values))
        return torch.stack(advantages, dim=0)
