"""Transcription of specs/policy.md (1)-(9) in float64, nothing from src/: the input weights
(1), the shifted actions (7), the targets (8) of world states (the world mask of
specs/action-mask.md (1) transcribed), the readout (2)-(4) with the causal attention of
tests/transformer/layers_oracle.py, the logits (5), the shift (6), the loss (9); on the weights
of a `Policy` read through the attribute names of Interface ($W$ = `Linear.weight.T`)."""

from collections import Counter

import numpy as np
import torch
from torch import nn
from transformer.layers_oracle import attn, attn_loop, mlp, norm, positions, weights

from mcpref.policy.config import PolicyConfig

PITCH_LIMIT, N_WORLD = (
    90.0,
    34,
)  # configs/action-mask.yaml; A starts with 34 world actions


def cfg_of(**changes) -> PolicyConfig:
    """A tiny config: d_h = 4, a small base, a large eps, dropout 0.25."""
    base = {"history": 4, "width": 8, "heads": 2, "hidden": 12, "layers": 2,
            "shift_hidden": 6, "rope_base": 7.0, "norm_eps": 0.05, "action_dropout": 0.25,
            "shift_weight": 1.5, "seed": 11, "latency_ms": 3.0, "latency_fail_ms": 10.0,
            "latency_calls": 5}  # fmt: skip
    return PolicyConfig(**(base | changes))


def input_weights(clip, names: tuple) -> np.ndarray:
    """(1): C of a clip, (L, |A| + 1): counts over n_j; the last row the one-hot of none."""
    info, L = clip.info, len(clip.info["states"])
    C = np.zeros((L, len(names) + 1))
    for j in range(L - 1):
        for a, count in Counter(info["actions"][j]).items():
            C[j, names.index(a)] = count / info["n"][j]
    C[L - 1, len(names)] = 1.0
    return C


def next_actions(clip, names: tuple) -> list[int]:
    """(7): iota of the action of A_{s+1} when s <= L - 3 and n_{s+1} = 1, else 0."""
    info, L = clip.info, len(clip.info["states"])
    one = lambda s: s <= L - 3 and info["n"][s + 1] == 1
    return [names.index(info["actions"][s + 1][0]) if one(s) else 0 for s in range(L)]


def allowed(name: str, pitch: float, hotbar: int, names: tuple) -> bool:
    """specs/action-mask.md (1) at a world state."""
    return (names.index(name) < N_WORLD and name != f"hotbar_{hotbar + 1}"
            and not (name == "turn_up" and pitch <= -PITCH_LIMIT)
            and not (name == "turn_down" and pitch >= PITCH_LIMIT))  # fmt: skip


def targets(clip, names: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(8) of a clip of world states: q (L - 2, |A|), chi (L - 2,)."""
    info, L = clip.info, len(clip.info["states"])
    assert not any(info["gui"])
    q, chi = np.zeros((L - 2, len(names))), np.zeros(L - 2, dtype=bool)
    for s in range(L - 2):
        h = clip.lines[info["states"][s + 1] - info["states"][0]]["hotbar"]
        T = [
            a
            for a in info["actions"][s + 1]
            if allowed(a, info["pitch"][s + 1], h, names)
        ]
        for a, count in Counter(T).items():
            q[s, names.index(a)] = count / len(T)
        chi[s] = info["n"][s] == 1 and info["n"][s + 1] == 1 and len(T) == 1
    return q, chi


def readout(
    net: nn.Module, cfg: PolicyConfig, Y, C, loop: bool = False
) -> torch.Tensor:
    """(2)-(4) on (Y, C) as given (no dropout): the used rows, their positions, the causal
    layers C, the final norm; `loop`: (3) by the naive loop."""
    ell = Y.shape[1]
    n = min(cfg.history, ell)
    H = Y[:, ell - n :] @ net.embed.weight.T + C[:, ell - n :] @ net.executed.weight.T
    pos, eps = positions(n), cfg.norm_eps
    for layer in net.layers:
        W = weights(layer)
        x = norm(H, W["g1"], eps)
        H = H + (attn_loop if loop else attn)(W, x, x, pos, pos, cfg.heads, cfg.rope_base, True)  # fmt: skip
        H = H + mlp(W, norm(H, W["g2"], eps))
    return norm(H, net.norm.weight, eps)


def draw(m: int, n: int, p: float) -> torch.Tensor:
    """(2): b = rand((m, n), float32) < p from the global CPU generator."""
    return torch.rand((m, n), dtype=torch.float32) < p


def dropped(C: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(2): the rows j of the used rows (the last b.shape[1]) with b_j: the one-hot of none."""
    C = C.clone()
    used = C[:, C.shape[1] - b.shape[1] :]
    none = torch.zeros(C.shape[-1], dtype=C.dtype)
    none[-1] = 1.0
    used[b] = none
    return C


def logits(net: nn.Module, R: torch.Tensor) -> torch.Tensor:
    """(5)."""
    return R @ net.head.weight.T


def shift(net: nn.Module, cfg: PolicyConfig, R: torch.Tensor, a: torch.Tensor):
    """(6): x = N([r; e_a]), (SiLU(x U_1) * x U_3) U_2."""
    dyn = net.dynamics
    x = norm(torch.cat([R, net.actions.weight[a]], -1), dyn.norm.weight, cfg.norm_eps)
    u = x @ dyn.w1.weight.T
    return (u * torch.sigmoid(u) * (x @ dyn.w3.weight.T)) @ dyn.w2.weight.T


def forward(net, cfg, names, Z, clips, b=None) -> tuple:
    """(7): C by (1) (rows dropped where b), a-> by (7)."""
    C = torch.tensor(np.stack([input_weights(c, names) for c in clips]), dtype=Z.dtype)
    R = readout(net, cfg, Z, C if b is None else dropped(C, b))
    a = torch.tensor([next_actions(c, names) for c in clips])
    return logits(net, R), shift(net, cfg, R, a)


def loss(out: tuple, Z, clips, names: tuple, w: float, metric) -> torch.Tensor:
    """(9) per clip, term by term: the CE of each state with a next interval, plus w times
    the squared metric of the shift error (target detached), the term removed by
    `torch.where` where not chi (so every shift row stays in the graph, gradient 0)."""
    lam, sh = out
    values = []
    for i, clip in enumerate(clips):
        q, chi = targets(clip, names)
        total = 0.0
        for s in range(len(q)):
            target = torch.tensor(q[s], dtype=lam.dtype)
            total = total - (target * torch.log_softmax(lam[i, s], -1)).sum()
            term = w * metric(sh[i, s] - (Z[i, s + 1] - Z[i, s]).detach()) ** 2
            total = total + torch.where(torch.tensor(bool(chi[s])), term, 0.0)
        values.append(total / len(q))
    return torch.stack(values)


def alpha_metric(alpha: torch.Tensor):
    """d_alpha of specs/perception.md, written here: ||alpha * delta||_2 over the last axis."""
    return lambda d: torch.sqrt(((alpha.to(d.dtype) * d) ** 2).sum(-1))
