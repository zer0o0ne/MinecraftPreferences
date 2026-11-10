"""Naive float64 references for specs/gui-stack.md (Notation, (1)-(3')) and gui-masks (8):
loops and numpy, nothing from src/."""

import math
from fractions import Fraction

import numpy as np
import torch


def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mu_i: mean of the non-NaN values of column i, 0 when there is none; s_i: 0 if they are
    all equal (or none), else their population standard deviation."""
    mu, s = [], []
    for col in x.T:
        vals = [float(v) for v in col if not math.isnan(v)]
        m = math.fsum(vals) / len(vals) if vals else 0.0
        spread = math.sqrt(math.fsum((v - m) ** 2 for v in vals) / max(len(vals), 1))
        mu.append(m)
        s.append(0.0 if len(set(vals)) <= 1 else spread)
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


def penalty(theta: list, l2: float) -> float:
    """lambda (||W1||_F^2 + ||W2||_F^2)."""
    return l2 * ((theta[0] ** 2).sum() + (theta[2] ** 2).sum())


def weighted_nll(theta: list, z: np.ndarray, y: list, w: np.ndarray) -> float:
    """sum_f w_f (-log p(y_f | z_f)), log p by log-softmax over Y."""
    lp = log_softmax(logits(theta, z))
    return sum(w[f] * -lp[f, y[f]] for f in range(len(y)))


def loss(theta: list, z: np.ndarray, y: list, w: np.ndarray, l2: float) -> float:
    """(3) over the frames given (all of S)."""
    return weighted_nll(theta, z, y, w) / sum(w) + penalty(theta, l2)


def loss_batch(theta, z, y, w, w_total: float, m: int, l2: float) -> float:
    """(3') over the frames given (a mini-batch): M sum_B w (-log p) / sum_S w + L2."""
    return m * weighted_nll(theta, z, y, w) / w_total + penalty(theta, l2)


def gradient(theta: list, z, y: list, w: np.ndarray, coef: float, l2: float) -> list:
    """Gradient of coef sum_f w_f (-log p(y_f | z_f)) + L2 by hand: dl_f = coef w_f
    (p_f - onehot(y_f)), back through (1); coef = 1 / sum w for (3), M / sum_S w for (3')."""
    w1, b1, w2, b2 = theta
    a = z @ w1.T + b1
    h = np.maximum(a, 0)
    p = np.exp(log_softmax(h @ w2.T + b2))
    dl = coef * w[:, None] * (p - np.eye(len(b2))[list(y)])
    da = (dl @ w2) * (a > 0)
    return [da.T @ z + 2 * l2 * w1, da.sum(0), dl.T @ h + 2 * l2 * w2, dl.sum(0)]


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
