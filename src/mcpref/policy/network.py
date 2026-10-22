"""(2)-(7), (10), (11) of specs/policy.md: the policy pi over a latent history and the
actions executed in it."""

import numpy as np
import torch
from torch import Tensor

from mcpref.policy.config import PolicyConfig
from mcpref.policy.targets import next_actions, weights
from mcpref.pretrain.clip import Clip
from mcpref.transformer.layers import Layer, decay_groups, mlp, rms_norm


class LayerC(Layer):
    """(3): the pre-norm layer, each row attending causally to the rows j' <= j."""

    def forward(self, H: Tensor, pos: Tensor) -> Tensor:
        H_hat = self.attn_norm(H)
        return self.update(H, H_hat, H_hat, pos, pos, causal=True)


class Dynamics(torch.nn.Module):
    """(6): gamma_Delta, U_1, U_3, U_2 of the shift MLP on x = [r; e_a] of size 2d."""

    def __init__(self, width: int, hidden: int, latent: int, eps: float) -> None:
        super().__init__()
        self.norm = rms_norm(2 * width, eps)
        self.w1 = torch.nn.Linear(2 * width, hidden, bias=False)
        self.w3 = torch.nn.Linear(2 * width, hidden, bias=False)
        self.w2 = torch.nn.Linear(hidden, latent, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """(6): (SiLU(x' U_1) * x' U_3) U_2, x' = N_gamma_Delta(x)."""
        return mlp(self, self.norm(x))


class Policy(torch.nn.Module):
    """(2)-(7): a history (Y, C) -> its readouts; their logits over A and the shift of
    a given action."""

    def __init__(self, cfg: PolicyConfig, latent: int, names: tuple[str, ...]) -> None:
        """(10): built on the CPU from `seed`, no global generator's state changed."""
        super().__init__()
        self.history, self.action_dropout = cfg.history, cfg.action_dropout
        self.names, self.index = names, {a: i for i, a in enumerate(names)}
        d, n_a = cfg.width, len(names)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg.seed)
            self.embed = torch.nn.Linear(latent, d, bias=False)
            self.executed = torch.nn.Linear(n_a + 1, d, bias=False)
            self.layers = torch.nn.ModuleList(
                LayerC(d, cfg.hidden, cfg.heads, cfg.rope_base, cfg.norm_eps)
                for _ in range(cfg.layers)
            )
            self.norm = rms_norm(d, cfg.norm_eps)
            self.head = torch.nn.Linear(d, n_a, bias=False)
            self.actions = torch.nn.Embedding(
                num_embeddings=n_a,
                embedding_dim=d,
                padding_idx=None,
                max_norm=None,
                scale_grad_by_freq=False,
                sparse=False,
            )
            self.dynamics = Dynamics(d, cfg.shift_hidden, latent, cfg.norm_eps)

    def dropped(self, Y: Tensor, C: Tensor) -> Tensor:
        """(2): in training mode, c_j replaced by the one-hot of no action where
        b = rand((m, n)) < p_none; C unchanged in eval mode."""
        if not self.training:
            return C
        u = torch.rand(C.shape[:2], device=Y.device, dtype=torch.float32)
        none = C.new_zeros(C.shape[-1])
        none[-1] = 1.0
        return torch.where(u[..., None] < self.action_dropout, none, C)

    def readout(self, Y: Tensor, C: Tensor) -> Tensor:
        """(2)-(4): r_j of the n = min(L_pi, l) last rows, positions j - (n - 1), shape
        (m, n, d)."""
        n = min(self.history, Y.shape[1])
        Y, C = Y[:, -n:], C[:, -n:]
        H = self.embed(Y) + self.executed(self.dropped(Y, C))
        pos = torch.arange(1 - n, 1, device=Y.device)
        for layer in self.layers:
            H = layer(H, pos)
        return self.norm(H)

    def logits(self, R: Tensor) -> Tensor:
        """(5): lambda = r W_lambda, (..., d) -> (..., |A|)."""
        return self.head(R)

    def shift(self, R: Tensor, a: Tensor) -> Tensor:
        """(6): the shift in z of action a[...] at readout R[...], shape (..., D)."""
        return self.dynamics(torch.cat([R, self.actions(a)], dim=-1))

    def forward(self, Z: Tensor, clips: list[Clip]) -> tuple[Tensor, Tensor]:
        """(7): the logits (m, L, |A|) and the shifts of the next actions (m, L, D), C
        by (1)."""
        C = np.stack([weights(clip, self.index) for clip in clips])
        a = np.stack([next_actions(clip, self.index) for clip in clips])
        R = self.readout(Z, torch.tensor(C, dtype=Z.dtype, device=Z.device))
        return self.logits(R), self.shift(R, torch.tensor(a, device=Z.device))

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(11): pi - the weights of the Linears and E, pi_no_decay - the gains, each in
        named_parameters() order."""
        return decay_groups(self, "pi")
