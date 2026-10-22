"""(9) of specs/policy.md: the P5 loss of each clip, the cross-entropy with the next
actions' distribution plus the shift error in squared d_alpha."""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mcpref.gui.mask import Masks
from mcpref.policy.config import PolicyConfig
from mcpref.policy.targets import targets
from mcpref.pretrain.clip import Clip


def cross_entropy(logits: Tensor, q: Tensor) -> Tensor:
    """(9): CE(lambda_s, q_s) = -sum_a q_s(a) log softmax(lambda_s)_a on the rows
    (m (L - 2), |A|), shape (m, L - 2); 0 where q_s = 0."""
    ce = F.cross_entropy(
        logits.flatten(0, 1),
        q.flatten(0, 1),
        weight=None,
        reduction="none",
        label_smoothing=0.0,
    )
    return ce.unflatten(0, q.shape[:2])


class PolicyLoss(torch.nn.Module):
    """(9): l^pi of each clip; w_Delta, the masks and d_alpha as plain attributes."""

    def __init__(
        self, cfg: PolicyConfig, masks: Masks, metric: Callable[[Tensor], Tensor]
    ) -> None:
        super().__init__()
        self.shift_weight, self.masks, self.metric = cfg.shift_weight, masks, metric

    def forward(
        self, out: tuple[Tensor, Tensor], Z: Tensor, clips: list[Clip]
    ) -> Tensor:
        """(9): the mean over s = 0..L-3 of CE(lambda_s, q_s) + w_Delta chi_s
        d_alpha(shift_s - sg(z_{s+1} - z_s))^2, shape (m,)."""
        logits, shift = out
        L = logits.shape[1]
        q, chi = zip(*(targets(clip, self.masks) for clip in clips))
        q = torch.tensor(np.stack(q), dtype=logits.dtype, device=logits.device)
        chi = torch.tensor(np.stack(chi), device=logits.device)
        ce = cross_entropy(logits[:, : L - 2], q)
        error = shift[:, : L - 2] - (Z[:, 1 : L - 1] - Z[:, : L - 2]).detach()
        d2 = self.metric(error).square()
        return (ce + self.shift_weight * torch.where(chi, d2, 0.0)).sum(dim=1) / (L - 2)

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(11): l^pi declares no group."""
        return {}
