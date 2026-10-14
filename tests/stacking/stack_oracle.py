"""Naive float64 references for specs/gui-stack.md (Notation, (1)-(4)) and gui-masks (8):
loops and numpy, nothing from src/."""

import math
from fractions import Fraction

import numpy as np
import torch


def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mu_i: mean of the non-NaN values of column i; s_i: 0 if they are all equal, else
    their population standard deviation."""
    mu, s = [], []
    for col in x.T:
        vals = [float(v) for v in col if not math.isnan(v)]
        m = sum(vals) / len(vals)
        spread = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        mu.append(m)
        s.append(0.0 if len(set(vals)) == 1 else spread)
    return np.array(mu), np.array(s)


def standardised(x: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """z = (x - mu) / s where x is not NaN and s > 0, else 0."""
    z = np.zeros(x.shape)
    for f, i in np.ndindex(*x.shape):
        if not math.isnan(x[f, i]) and s[i] > 0:
            z[f, i] = (x[f, i] - mu[i]) / s[i]
    return z


def init(c: int, h: int, n: int, seed: int) -> list[np.ndarray]:
    """(2): W1, b1, W2, b2 in this order from one CPU generator, (2u - 1) / sqrt(fan_in)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shapes = [((h, c), c), ((h,), c), ((n, h), h), ((n,), h)]
    draws = [
        (torch.rand(sh, generator=gen, dtype=torch.float32), f) for sh, f in shapes
    ]
    return [(2 * u.double().numpy() - 1) / math.sqrt(f) for u, f in draws]


def logits(theta: list, z: np.ndarray) -> np.ndarray:
    """(1) with explicit loops: l = W2 max(0, W1 z + b1) + b2."""
    w1, b1, w2, b2 = theta
    out = np.zeros((len(z), len(b2)))
    for f in range(len(z)):
        h = [max(0.0, b1[j] + sum(w1[j, i] * z[f, i] for i in range(z.shape[1])))
             for j in range(len(b1))]  # fmt: skip
        for k in range(len(b2)):
            out[f, k] = b2[k] + sum(w2[k, j] * h[j] for j in range(len(h)))
    return out


def log_softmax(l: np.ndarray) -> np.ndarray:
    m = l.max(axis=1, keepdims=True)
    return l - m - np.log(np.exp(l - m).sum(axis=1, keepdims=True))


def weights(y: list) -> np.ndarray:
    """w_f = 1 / n_{y_f}."""
    return np.array([1 / list(y).count(v) for v in y])


def loss(theta: list, z: np.ndarray, y: list, w: np.ndarray, l2: float) -> float:
    """(3): weighted mean of -log p(y_f | z_f), normalised by sum w, plus L2 on W1, W2."""
    lp = log_softmax(logits(theta, z))
    data = sum(w[f] * -lp[f, y[f]] for f in range(len(y))) / sum(w)
    return data + l2 * ((theta[0] ** 2).sum() + (theta[2] ** 2).sum())


def gradient(theta: list, z: np.ndarray, y: list, w: np.ndarray, l2: float) -> list:
    """Gradient of (3) by hand: dL/dl_f = w_f / sum(w) (p_f - onehot(y_f)), back through (1)."""
    w1, b1, w2, b2 = theta
    a = z @ w1.T + b1
    h = np.maximum(a, 0)
    p = np.exp(log_softmax(h @ w2.T + b2))
    dl = (w / w.sum())[:, None] * (p - np.eye(len(b2))[list(y)])
    da = (dl @ w2) * (a > 0)
    return [da.T @ z + 2 * l2 * w1, da.sum(0), dl.T @ h + 2 * l2 * w2, dl.sum(0)]


def adam(theta: list, z, y, w, l2, lr, betas, eps, steps) -> list:
    """(4): theta_1 .. theta_T of full-batch Adam from theta_0, bias-corrected as in [T]."""
    (b1, b2), out = betas, []
    m = [np.zeros_like(p) for p in theta]
    v = [np.zeros_like(p) for p in theta]
    for t in range(1, steps + 1):
        g = gradient(theta, z, y, w, l2)
        m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(m, g)]
        v = [b2 * vi + (1 - b2) * gi**2 for vi, gi in zip(v, g)]
        theta = [
            p - lr * (mi / (1 - b1**t)) / (np.sqrt(vi) / math.sqrt(1 - b2**t) + eps)
            for p, mi, vi in zip(theta, m, v)
        ]
        out.append(theta)
    return out


def matrices(classes: list, labellers: list) -> dict:
    """gui-masks (8): class -> (|F_k|, A_k as floats) for the classes some labeller assigns."""
    out = {}
    for k in classes:
        fk = [f for f in range(len(labellers[0])) if any(L[f] == k for L in labellers)]
        if fk:
            agree = [
                [sum(a[f] == b[f] for f in fk) for b in labellers] for a in labellers
            ]
            out[k] = (
                len(fk),
                [[float(Fraction(n, len(fk))) for n in r] for r in agree],
            )
    return out


def reference(world, cfg: dict) -> dict:
    """A run of specs/gui-stack.md on `world` of stack_world.py: S = train then validation
    frames; its mu, s, z, label indices y, w; theta_0 .. theta_T of (2), (4); z of the test
    frames."""
    x = np.array(world.train.x + world.validation.x, dtype=np.float64)
    y = [world.labels.index(v) for v in world.train.y + world.validation.y]
    mu, s = moments(x)
    z, w = standardised(x, mu, s), weights(y)
    theta0 = init(len(world.columns), cfg["hidden"], len(world.labels), cfg["seed"])
    args = (cfg["l2"], cfg["lr"], cfg["betas"], cfg["adam_eps"], cfg["steps"])
    path = [theta0, *adam(theta0, z, y, w, *args)]
    z_te = standardised(np.array(world.test.x), mu, s)
    return {"mu": mu, "s": s, "z": z, "y": y, "w": w, "path": path, "z_te": z_te}
