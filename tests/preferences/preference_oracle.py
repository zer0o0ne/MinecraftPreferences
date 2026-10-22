"""Transcription of specs/preference.md (1), (5)-(8), (10), (12) in float64, nothing from src/:
the math on the weights of the modules (read through the attribute names of Interface; a
matrix acts on rows, so $W$ = `Linear.weight.T`); (2)-(4) and a layer's residual steps from
tests/transformer/layers_oracle.py."""

import torch
from torch import nn
from transformer.layers_oracle import (  # noqa: F401  (re-exported to the tests of P)
    GAINS,
    LINEARS,
    norm,
    positions,
    randomize_gains,
    update,
    weights,
)

from mcpref.preferences.config import PreferenceConfig


def cfg_of(**changes) -> PreferenceConfig:
    """A tiny config: d_h = 4 (two RoPE frequencies), a small base, a large eps."""
    base = {"history": 3, "width": 8, "heads": 2, "hidden": 12, "layers": ("S", "X"),
            "rope_base": 7.0, "norm_eps": 0.05, "seed": 11, "latency_ms": 3.0,
            "latency_fail_ms": 10.0, "latency_calls": 5}  # fmt: skip
    return PreferenceConfig(**(base | changes))


def layer_s(W: dict, H, pos, cfg: PreferenceConfig) -> torch.Tensor:
    """(5) on one argument."""
    x = norm(H, W["g1"], cfg.norm_eps)
    return update(W, H, x, x, pos, pos, cfg.heads, cfg.rope_base, cfg.norm_eps, False)


def layer_x(W: dict, A, B, pos_a, pos_b, cfg: PreferenceConfig) -> tuple:
    """(6) from the layer's inputs."""
    a, b = norm(A, W["g1"], cfg.norm_eps), norm(B, W["g1"], cfg.norm_eps)
    h, base, eps = cfg.heads, cfg.rope_base, cfg.norm_eps
    return (update(W, A, a, b, pos_a, pos_b, h, base, eps, False),
            update(W, B, b, a, pos_b, pos_a, h, base, eps, False))  # fmt: skip


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
