"""Transcription of specs/preference.md (2)-(6) in float64, nothing from src/: RMSNorm, RoPE,
the attention (3) (`causal`: the sum over $j' \\le j$, specs/policy.md (3)), the MLP (4) and
the residual steps of a layer, on the weights of a module with the attributes of `Layer`
(Interface; a matrix acts on rows, so $W$ = `Linear.weight.T`); a naive loop of (3) over
pairs, heads and rows. Moved from tests/preferences/preference_oracle.py."""

import math

import torch
from torch import nn

ATTRIBUTES = ("attn_norm", "q", "k", "v", "o", "mlp_norm", "w1", "w3", "w2")  # Layer
LINEARS = ("q", "k", "v", "o", "w1", "w3", "w2")
GAINS = ("attn_norm", "mlp_norm")


def norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """N_gamma: RMSNorm over the last axis."""
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * gain


def positions(n: int) -> torch.Tensor:
    """varpi_j = j - (n - 1), j = 0..n-1."""
    return torch.arange(n, dtype=torch.float64) - (n - 1)


def rope(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """(2) on the last axis of x (..., n, d_h); pos (n,)."""
    half = x.shape[-1] // 2
    theta = base ** (-2 * torch.arange(half, dtype=x.dtype) / x.shape[-1])
    angle = pos.to(x.dtype)[:, None] * theta
    a, b = x[..., :half], x[..., half:]
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def rope_one(x: list[float], pos: float, base: float) -> list[float]:
    """(2) on one vector, element by element."""
    half, out = len(x) // 2, list(x)
    for i in range(half):
        angle = pos * base ** (-2 * i / len(x))
        out[i] = x[i] * math.cos(angle) - x[i + half] * math.sin(angle)
        out[i + half] = x[i + half] * math.cos(angle) + x[i] * math.sin(angle)
    return out


def weights(layer: nn.Module) -> dict:
    """W_q .. W_2 (acting on rows) and the gains of a layer."""
    w = {k: getattr(layer, k).weight.T for k in LINEARS}
    return w | {"g1": layer.attn_norm.weight, "g2": layer.mlp_norm.weight}


def attn(W: dict, Q, K, pos_q, pos_k, h: int, base: float, causal: bool = False):
    """(3), the heads as blocks of d_h columns; `causal`: keys j' <= j only."""
    m, n_q, d = Q.shape
    split = lambda x: x.reshape(m, -1, h, d // h).transpose(1, 2)
    q = rope(split(Q @ W["q"]), pos_q, base)
    k = rope(split(K @ W["k"]), pos_k, base)
    v = split(K @ W["v"])
    s = q @ k.transpose(-1, -2) / math.sqrt(d // h)
    if causal:
        keep = torch.ones(n_q, K.shape[1], dtype=torch.bool).tril()
        s = s.masked_fill(~keep, -math.inf)
    w = torch.softmax(s, -1)
    return (w @ v).transpose(1, 2).reshape(m, n_q, d) @ W["o"]


def attn_loop(W: dict, Q, K, pos_q, pos_k, h: int, base: float, causal: bool = False):
    """(3) by loops over pairs r, heads e and rows j, j' in Python floats; `causal`: the
    sum over j' = 0..j."""
    (m, n_q, d), n_k, dh = Q.shape, K.shape[1], Q.shape[2] // h
    qs, ks, vs = (X.tolist() for X in (Q @ W["q"], K @ W["k"], K @ W["v"]))
    out = torch.zeros(m, n_q, d, dtype=torch.float64)
    for r in range(m):
        for e in range(h):
            c = slice(e * dh, (e + 1) * dh)
            kap = [rope_one(ks[r][j][c], float(pos_k[j]), base) for j in range(n_k)]
            for j in range(n_q):
                keys = range(min(j + 1, n_k)) if causal else range(n_k)
                qj = rope_one(qs[r][j][c], float(pos_q[j]), base)
                z = [
                    sum(a * b for a, b in zip(qj, kap[jj])) / math.sqrt(dh)
                    for jj in keys
                ]
                top = max(z)
                p = [math.exp(x - top) for x in z]
                for i in range(dh):
                    o = sum(pj * vs[r][jj][c][i] for pj, jj in zip(p, keys)) / sum(p)
                    out[r, j, e * dh + i] = o
    return out @ W["o"]


def mlp(W: dict, x: torch.Tensor) -> torch.Tensor:
    """(4), SiLU(z) = z sigmoid(z)."""
    a = x @ W["w1"]
    return (a * torch.sigmoid(a) * (x @ W["w3"])) @ W["w2"]


def update(W: dict, H, H_hat, G_hat, pos_h, pos_g, h: int, base: float, eps: float,
           causal: bool) -> torch.Tensor:  # fmt: skip
    """The residual steps of (5), (6): H + Attn(H_hat, G_hat), then + F(N_gamma2(.))."""
    H = H + attn(W, H_hat, G_hat, pos_h, pos_g, h, base, causal)
    return H + mlp(W, norm(H, W["g2"], eps))


def randomize_gains(net: nn.Module) -> None:
    """Every gain (and b_inc when present) drawn away from its initial value."""
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("norm.weight") or name == "inc.bias":
                p.uniform_(0.5, 1.5)
