"""Transcription of specs/preference.md (1)-(8), (10), (12) in float64, nothing from src/: the
math on the weights of the modules (read through the attribute names of Interface; a matrix
acts on rows, so $W$ = `Linear.weight.T`); a naive loop of (3) over pairs, heads and rows."""

import math

import torch
from torch import nn

from mcpref.preferences.config import PreferenceConfig

LINEARS = ("q", "k", "v", "o", "w1", "w3", "w2")  # per layer, Interface
GAINS = ("attn_norm", "mlp_norm")


def cfg_of(**changes) -> PreferenceConfig:
    """A tiny config: d_h = 4 (two RoPE frequencies), a small base, a large eps."""
    base = {"history": 3, "width": 8, "heads": 2, "hidden": 12, "layers": ("S", "X"),
            "rope_base": 7.0, "norm_eps": 0.05, "seed": 11, "latency_ms": 3.0,
            "latency_fail_ms": 10.0, "latency_calls": 5}  # fmt: skip
    return PreferenceConfig(**(base | changes))


def norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """N_gamma of Notation."""
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


def attn(W: dict, Q, K, pos_q, pos_k, h: int, base: float) -> torch.Tensor:
    """(3), the heads as blocks of d_h columns."""
    m, n_q, d = Q.shape
    split = lambda x: x.reshape(m, -1, h, d // h).transpose(1, 2)
    q = rope(split(Q @ W["q"]), pos_q, base)
    k = rope(split(K @ W["k"]), pos_k, base)
    v = split(K @ W["v"])
    w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // h), -1)
    return (w @ v).transpose(1, 2).reshape(m, n_q, d) @ W["o"]


def attn_loop(W: dict, Q, K, pos_q, pos_k, h: int, base: float) -> torch.Tensor:
    """(3) by loops over pairs r, heads e and rows j, j' in Python floats."""
    (m, n_q, d), n_k, dh = Q.shape, K.shape[1], Q.shape[2] // h
    qs, ks, vs = (X.tolist() for X in (Q @ W["q"], K @ W["k"], K @ W["v"]))
    out = torch.zeros(m, n_q, d, dtype=torch.float64)
    for r in range(m):
        for e in range(h):
            c = slice(e * dh, (e + 1) * dh)
            kap = [rope_one(ks[r][j][c], float(pos_k[j]), base) for j in range(n_k)]
            for j in range(n_q):
                qj = rope_one(qs[r][j][c], float(pos_q[j]), base)
                z = [sum(a * b for a, b in zip(qj, kj)) / math.sqrt(dh) for kj in kap]
                top = max(z)
                p = [math.exp(x - top) for x in z]
                for i in range(dh):
                    o = sum(p[jj] * vs[r][jj][c][i] for jj in range(n_k)) / sum(p)
                    out[r, j, e * dh + i] = o
    return out @ W["o"]


def mlp(W: dict, x: torch.Tensor) -> torch.Tensor:
    """(4), SiLU(z) = z sigmoid(z)."""
    a = x @ W["w1"]
    return (a * torch.sigmoid(a) * (x @ W["w3"])) @ W["w2"]


def layer_s(W: dict, H, pos, cfg: PreferenceConfig) -> torch.Tensor:
    """(5) on one argument."""
    x = norm(H, W["g1"], cfg.norm_eps)
    H = H + attn(W, x, x, pos, pos, cfg.heads, cfg.rope_base)
    return H + mlp(W, norm(H, W["g2"], cfg.norm_eps))


def layer_x(W: dict, A, B, pos_a, pos_b, cfg: PreferenceConfig) -> tuple:
    """(6) from the layer's inputs."""
    a, b = norm(A, W["g1"], cfg.norm_eps), norm(B, W["g1"], cfg.norm_eps)
    A1 = A + attn(W, a, b, pos_a, pos_b, cfg.heads, cfg.rope_base)
    B1 = B + attn(W, b, a, pos_b, pos_a, cfg.heads, cfg.rope_base)
    A2 = A1 + mlp(W, norm(A1, W["g2"], cfg.norm_eps))
    return A2, B1 + mlp(W, norm(B1, W["g2"], cfg.norm_eps))


def stack(net: nn.Module, cfg: PreferenceConfig, Ya, Yb, pos=positions) -> tuple:
    """(1), (7): the readouts r_a, r_b; `pos(n)` the positions of n used rows."""
    n_a, n_b = min(cfg.history, Ya.shape[1]), min(cfg.history, Yb.shape[1])
    A = Ya[:, Ya.shape[1] - n_a :] @ net.embed.weight.T
    B = Yb[:, Yb.shape[1] - n_b :] @ net.embed.weight.T
    pa, pb = pos(n_a), pos(n_b)
    for kind, layer in zip(cfg.layers, net.layers):
        W = weights(layer)
        if kind == "S":
            A, B = layer_s(W, A, pa, cfg), layer_s(W, B, pb, cfg)
        else:
            A, B = layer_x(W, A, B, pa, pb, cfg)
    g = net.norm.weight
    return norm(A[:, n_a - 1], g, cfg.norm_eps), norm(B[:, n_b - 1], g, cfg.norm_eps)


def preference(net: nn.Module, cfg: PreferenceConfig, Ya, Yb, pos=positions) -> tuple:
    """(8): the logits (s, t)."""
    ra, rb = stack(net, cfg, Ya, Yb, pos)
    s = (ra - rb) @ net.pref.weight[0]
    return s, (ra + rb) @ net.inc.weight[0] + net.inc.bias[0]


def bce(x: torch.Tensor, y) -> torch.Tensor:
    """BCE of Notation on logits x, log sigma(x) = -log(1 + e^{-x})."""
    log_sig = lambda z: -torch.log1p(torch.exp(-z))
    return -(y * log_sig(x) + (1 - y) * log_sig(-x))


def loss(out: tuple, labels: list[dict]) -> torch.Tensor:
    """(10) per pair: 2 BCE(t, inc) + 2 [inc = 0] BCE(s, p); for inc = 1 the term of s
    times 0 on the target 0.5 (its gradient 0, not absent)."""
    s, t = out
    terms = []
    for r, y in enumerate(labels):
        keep = 1.0 if y["inc"] == 0 else 0.0
        target = y["p"] if y["inc"] == 0 else 0.5
        terms.append(2 * bce(t[r], float(y["inc"])) + 2 * keep * bce(s[r], target))
    return torch.stack(terms)


def decayed_names(cfg: PreferenceConfig) -> tuple[list[str], list[str]]:
    """(12): the names of the parameters of `P` and `P_no_decay`, in Interface order."""
    decay, rest = ["embed.weight"], []
    for i in range(len(cfg.layers)):
        decay += [f"layers.{i}.{k}.weight" for k in LINEARS]
        rest += [f"layers.{i}.{k}.weight" for k in GAINS]
    return decay + ["pref.weight", "inc.weight"], rest + ["norm.weight", "inc.bias"]


def randomize_gains(net: nn.Module) -> None:
    """Every gain (and b_inc when present) drawn away from its initial value."""
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("norm.weight") or name == "inc.bias":
                p.uniform_(0.5, 1.5)
