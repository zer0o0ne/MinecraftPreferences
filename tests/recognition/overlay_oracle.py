"""Naive references for specs/f3-overlay.md: w_tau, P, R, the dilation and Q by explicit
loops over pixels (P by drawn_oracle.py, specs/gui-masks.md (9)); (1), (2) in Python
integers and floats; nothing from src/."""

import drawn_oracle as do
import numpy as np

TEMPLATES = ("normal", "dimmed")


def width(font: np.ndarray, prefix: str) -> int:
    """Notation: w_tau = sum of adv(tau_i)."""
    return sum(do.advance(font, ord(ch)) for ch in prefix)


def masks(font: np.ndarray, prefix: str, w: int, h: int) -> tuple:
    """Notation: P = G_0 of the label (prefix, (2, 2, w_tau, 8)); Q = R minus the 3 x 3
    dilation of P, R = [1, 2 + w_tau) x [1, 10)."""
    wt = width(font, prefix)
    p = do.text_pixels(prefix, (2, 2, wt, 8), font, w, h)[0]
    q = np.zeros((h, w), dtype=bool)
    for y in range(1, 10):
        for x in range(1, 2 + wt):
            q[y, x] = not any(
                0 <= x + dx < w and 0 <= y + dy < h and p[y + dy, x + dx]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            )
    return p, q


def score(y: np.ndarray, p: np.ndarray, q: np.ndarray, lam, delta, beta) -> float:
    """(1), (2): S = max over templates of min(a_m, g_m); lam, delta, beta map template ->
    number."""
    best = 0.0
    for m in TEMPLATES:
        n_p = n_a = n_q = n_g = 0
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                v = int(y[r, c])
                if p[r, c]:
                    n_p += 1
                    n_a += abs(v - lam[m]) <= delta[m]
                if q[r, c]:
                    n_q += 1
                    n_g += v <= beta[m]
        best = max(best, min(n_a / n_p, n_g / n_q))
    return best
