"""(1), (7), (8), (11), (12) of specs/preference.md: the preference network P."""

import torch
from torch import Tensor

from mcpref.preferences.config import PreferenceConfig
from mcpref.preferences.layers import LayerS, LayerX
from mcpref.transformer.layers import decay_groups, rms_norm

KINDS = {"S": LayerS, "X": LayerX}
"""Notation Lambda: the layer class of each kind."""


class Preference(torch.nn.Module):
    """(1), (7), (8): two latent histories -> the logits (s, t) of (p_succ, p_inc)."""

    def __init__(self, cfg: PreferenceConfig, latent: int) -> None:
        """(11): built on the CPU from `seed`, no global generator's state changed."""
        super().__init__()
        self.history = cfg.history
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg.seed)
            self.embed = torch.nn.Linear(latent, cfg.width, bias=False)
            self.layers = torch.nn.ModuleList(
                KINDS[kind](
                    cfg.width, cfg.hidden, cfg.heads, cfg.rope_base, cfg.norm_eps
                )
                for kind in cfg.layers
            )
            self.norm = rms_norm(cfg.width, cfg.norm_eps)
            self.pref = torch.nn.Linear(cfg.width, 1, bias=False)
            self.inc = torch.nn.Linear(cfg.width, 1, bias=True)

    def tokens(self, Y: Tensor) -> tuple[Tensor, Tensor]:
        """(1): H^0 of the n = min(L_P, l) last rows of Y, their positions j - (n - 1)."""
        n = min(self.history, Y.shape[1])
        return self.embed(Y[:, -n:]), torch.arange(1 - n, 1, device=Y.device)

    def forward(self, Z_a: Tensor, Z_b: Tensor) -> tuple[Tensor, Tensor]:
        """(7), (8): s = (r_a - r_b) w_succ, t = (r_a + r_b) w_inc + b_inc, shape (m,)."""
        A, pos_a = self.tokens(Z_a)
        B, pos_b = self.tokens(Z_b)
        for layer in self.layers:
            A, B = layer(A, B, pos_a, pos_b)
        r_a, r_b = self.norm(A[:, -1]), self.norm(B[:, -1])
        return self.pref(r_a - r_b)[:, 0], self.inc(r_a + r_b)[:, 0]

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): P - the weights of the Linears, P_no_decay - the gains and b_inc, each
        in named_parameters() order."""
        return decay_groups(self, "P")
