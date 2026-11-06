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
        self, out: tuple[Tensor, Tensor], Z: Tensor, clips: list[Clip]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """(3) of specs/pretrain-diagnostics.md: per-clip terms and valid rows."""
        logits, shift = out
        L = logits.shape[1]
        q, chi = zip(*(targets(clip, self.masks) for clip in clips))
        q = torch.tensor(np.stack(q), dtype=logits.dtype, device=logits.device)
        chi = torch.tensor(np.stack(chi), device=logits.device)
        index = {a: i for i, a in enumerate(self.masks.names)}
        current = np.stack([weights(clip, index)[: L - 2] for clip in clips])
        current = torch.tensor(current, dtype=logits.dtype, device=logits.device)
        ce = cross_entropy(logits[:, : L - 2], q)
        error = shift[:, : L - 2] - (Z[:, 1 : L - 1] - Z[:, : L - 2]).detach()
        d2 = self.metric(error).square()
        shift_term = self.shift_weight * torch.where(chi, d2, 0.0)
        terms = {
            "action": ce.sum(dim=1) / (L - 2),
            "shift": shift_term.sum(dim=1) / (L - 2),
            "total": (ce + shift_term).sum(dim=1) / (L - 2),
        }
        valid = {"action_valid": q.sum(dim=-1) > 0, "shift_valid": chi}
        self._diagnostic_targets = (q.detach(), current.detach())
        return terms, valid

    def diagnostic_targets(self) -> tuple[Tensor, Tensor]:
        value = self._diagnostic_targets
        self._diagnostic_targets = None
        return value

    def zero_shift(self, Z: Tensor, chi: Tensor) -> Tensor:
        """(16): the shift term with every predicted shift replaced by zero."""
        L = Z.shape[1]
        error = -(Z[:, 1 : L - 1] - Z[:, : L - 2]).detach()
        d2 = self.metric(error).square()
        return (self.shift_weight * torch.where(chi, d2, 0.0)).sum(dim=1) / (L - 2)

    def forward(
        self, out: tuple[Tensor, Tensor], Z: Tensor, clips: list[Clip]
    ) -> Tensor:
        """(9): the original per-state addition and reduction, shape (m,)."""
        total = self.terms(out, Z, clips)[0]["total"]
        self._diagnostic_targets = None
        return total

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(11): l^pi declares no group."""
        return {}
