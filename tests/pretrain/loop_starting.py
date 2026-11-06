"""Fakes of loop_modules.py declaring the starting `groups` of configs/pretrain.yaml (the seven of
specs/perception.md, the two of specs/policy.md, the two of specs/preference.md), for the tests
that put a block's real modules inside `pretrain` (specs/pretrain.md Interface)."""

import torch

from .loop_modules import (
    ACTIONS,
    F64,
    D,
    Fake,
    L,
    Phi,
    PhiLoss,
    Pi,
    PLoss,
    Pref,
    fixed,
    pi_terms,
)


class PhiStart(Phi):
    """The fake phi plus e_1 + .. + e_4; the six groups of phi of specs/perception.md."""

    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = (fixed(20 + k, D) for k in range(4))

    def forward(self, clips: list) -> torch.Tensor:
        return super().forward(clips) + self.e1 + self.e2 + self.e3 + self.e4

    def groups(self) -> dict:
        return {"phi_vision": [self.A], "phi_vision_no_decay": [self.c],
                "phi_llm": [self.e1], "phi_llm_no_decay": [self.e2],
                "phi_head": [self.e3], "phi_head_no_decay": [self.e4]}  # fmt: skip


class PhiLossStart(PhiLoss):
    def groups(self) -> dict:
        return {"phi_dual": [self.lam]}


class PiStart(Pi):
    """The fake pi with the groups of specs/policy.md (11)."""

    def groups(self) -> dict:
        return {"pi": [self.Wa, self.V], "pi_no_decay": [self.ba]}


class PiLossStart(Fake):
    """pi_terms of loop_modules.py, without parameters."""

    def diagnostic_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._diagnostic

    def terms(self, out: tuple, Z: torch.Tensor, clips: list) -> tuple[dict, dict]:
        self.note("pi_loss", n=len(clips))
        logits, shift = out
        a = torch.tensor([c.info["a"] for c in clips])
        action = -torch.log_softmax(logits, -1).gather(-1, a[..., None])[..., 0].sum(1)
        shift_term = ((shift[:, :-1] - (Z[:, 1:] - Z[:, :-1])) ** 2).sum((1, 2))
        valid = torch.ones(len(clips), L - 2, dtype=torch.bool)
        q = torch.nn.functional.one_hot(a[:, : L - 2], ACTIONS).to(F64)
        self._diagnostic = q, q.clone()
        return {
            "action": action,
            "shift": shift_term,
            "total": pi_terms(logits, shift, Z, clips),
        }, {"action_valid": valid, "shift_valid": valid}

    def forward(self, out: tuple, Z: torch.Tensor, clips: list) -> torch.Tensor:
        return self.terms(out, Z, clips)[0]["total"]

    def zero_shift(self, Z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return ((Z[:, 1:] - Z[:, :-1]) ** 2).sum((1, 2))

    def groups(self) -> dict:
        return {}


class PStart(Pref):
    """The fake P with the groups of specs/preference.md (12)."""

    def groups(self) -> dict:
        return {"P": [self.u, self.v], "P_no_decay": [self.beta]}


def starting(**real: torch.nn.Module) -> dict:
    """The six modules in module order: the fakes above, replaced by `real` ones."""
    mods = {"phi": PhiStart(), "phi_loss": PhiLossStart(), "pi": PiStart(),
            "pi_loss": PiLossStart(), "P": PStart(), "P_loss": PLoss()}  # fmt: skip
    return mods | real
