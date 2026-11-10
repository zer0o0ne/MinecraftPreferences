"""(9) of specs/policy.md: the P5 loss of each clip, the cross-entropy with the next
actions' distribution plus the shift error in squared d_alpha."""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mcpref.gui.mask import Masks
from mcpref.policy.config import PolicyConfig
from mcpref.policy.targets import targets, weights
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

    def terms(
        self, out: tuple[Tensor, ...], Z: Tensor, clips: list[Clip]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """(3) of specs/pretrain-diagnostics.md: terms, valid rows and observations."""
        logits, shift, *shared = out
        L = logits.shape[1]
        if shared:
            current = shared[0]
        else:
            index = {a: i for i, a in enumerate(self.masks.names)}
            current = np.stack([weights(clip, index) for clip in clips])
            current = torch.tensor(current, dtype=logits.dtype, device=logits.device)
        current = current[:, : L - 2]
        q, chi = zip(*(targets(clip, self.masks) for clip in clips))
        q = torch.tensor(np.stack(q), dtype=logits.dtype, device=logits.device)
        chi = torch.tensor(np.stack(chi), device=logits.device)
        ce = cross_entropy(logits[:, : L - 2], q)
        factual = (Z[:, 1 : L - 1] - Z[:, : L - 2]).detach()
        d2 = self.metric(shift[:, : L - 2] - factual).square()
        factual_d2 = self.metric(factual).square()
        shift_term = self.shift_weight * torch.where(chi, d2, 0.0)
        zero_shift = self.shift_weight * torch.where(chi, factual_d2, 0.0)
        terms = {
            "action": ce.sum(dim=1) / (L - 2),
            "shift": shift_term.sum(dim=1) / (L - 2),
            "total": (ce + shift_term).sum(dim=1) / (L - 2),
        }
        valid = {"action_valid": q.sum(dim=-1) > 0, "shift_valid": chi}
        observations = {
            "q": q.detach(),
            "current": current.detach(),
            "zero_shift": (zero_shift.sum(dim=1) / (L - 2)).detach(),
        }
        return terms, valid, observations

    def forward(self, out: tuple[Tensor, ...], Z: Tensor, clips: list[Clip]) -> Tensor:
        """(9): the original per-state addition and reduction, shape (m,)."""
        return self.terms(out, Z, clips)[0]["total"]

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(11): l^pi declares no group."""
        return {}
