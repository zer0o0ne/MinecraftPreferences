"""(2)-(6) of specs/preference.md: RoPE, attention, the SwiGLU MLP and the layers S, X."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from mcpref.preferences.config import PreferenceConfig


def rope(x: Tensor, positions: Tensor, base: float) -> Tensor:
    """(2): the pairs (i, i + d_h/2) of the last axis of x (..., n, d_h) rotated by
    positions_j * theta_i, theta_i = base^(-2i/d_h), in the dtype of x."""
    d_h = x.shape[-1]
    i = torch.arange(d_h // 2, dtype=x.dtype, device=x.device)
    theta = base ** (-2 * i / d_h)
    angle = positions.to(x.dtype)[:, None] * theta
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., : d_h // 2], x[..., d_h // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def split_heads(x: Tensor, heads: int) -> Tensor:
    """(3): (m, n, d) -> (m, h, n, d_h), head eta the columns (eta-1) d_h .. eta d_h - 1."""
    return x.unflatten(-1, (heads, -1)).transpose(-3, -2)


def attention(
    layer: torch.nn.Module,
    Q: Tensor,
    K: Tensor,
    pos_q: Tensor,
    pos_k: Tensor,
    heads: int,
    base: float,
) -> Tensor:
    """(3): Attn(Q, K) with the layer's q, k, v, o; no mask, no causality."""
    q = rope(split_heads(layer.q(Q), heads), pos_q, base)
    k = rope(split_heads(layer.k(K), heads), pos_k, base)
    v = split_heads(layer.v(K), heads)
    o = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1 / math.sqrt(q.shape[-1]),
        enable_gqa=False,
    )
    return layer.o(o.transpose(-3, -2).flatten(-2))


def mlp(layer: torch.nn.Module, x: Tensor) -> Tensor:
    """(4): F(x) = (SiLU(x W_1) * x W_3) W_2 with the layer's w1, w3, w2."""
    return layer.w2(F.silu(layer.w1(x)) * layer.w3(x))


def rms_norm(cfg: PreferenceConfig) -> torch.nn.RMSNorm:
    """Notation N_gamma: RMSNorm over the last axis of size d, gain initialised to 1."""
    return torch.nn.RMSNorm(cfg.width, eps=cfg.norm_eps, elementwise_affine=True)


class Layer(torch.nn.Module):
    """(5), (6): gamma_1, W_q, W_k, W_v, W_o, gamma_2, W_1, W_3, W_2 of one layer."""

    def __init__(self, cfg: PreferenceConfig) -> None:
        super().__init__()
        d, f = cfg.width, cfg.hidden
        self.heads, self.rope_base = cfg.heads, cfg.rope_base
        self.attn_norm = rms_norm(cfg)
        self.q = torch.nn.Linear(d, d, bias=False)
        self.k = torch.nn.Linear(d, d, bias=False)
        self.v = torch.nn.Linear(d, d, bias=False)
        self.o = torch.nn.Linear(d, d, bias=False)
        self.mlp_norm = rms_norm(cfg)
        self.w1 = torch.nn.Linear(d, f, bias=False)
        self.w3 = torch.nn.Linear(d, f, bias=False)
        self.w2 = torch.nn.Linear(f, d, bias=False)

    def update(
        self, H: Tensor, H_hat: Tensor, G_hat: Tensor, pos_h: Tensor, pos_g: Tensor
    ) -> Tensor:
        """(5), (6) for one argument H, its queries H_hat attending to the keys G_hat:
        H + Attn(H_hat, G_hat), then plus F(N_gamma2(.))."""
        H = H + attention(self, H_hat, G_hat, pos_h, pos_g, self.heads, self.rope_base)
        return H + mlp(self, self.mlp_norm(H))


class LayerS(Layer):
    """(5): each argument attends to itself, the same parameters for both."""

    def forward(
        self, A: Tensor, B: Tensor, pos_a: Tensor, pos_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        A_hat, B_hat = self.attn_norm(A), self.attn_norm(B)
        return (
            self.update(A, A_hat, A_hat, pos_a, pos_a),
            self.update(B, B_hat, B_hat, pos_b, pos_b),
        )


class LayerX(Layer):
    """(6): each argument attends to the other from the layer's inputs, one set of
    parameters for both directions."""

    def forward(
        self, A: Tensor, B: Tensor, pos_a: Tensor, pos_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        A_hat, B_hat = self.attn_norm(A), self.attn_norm(B)
        return (
            self.update(A, A_hat, B_hat, pos_a, pos_b),
            self.update(B, B_hat, A_hat, pos_b, pos_a),
        )
