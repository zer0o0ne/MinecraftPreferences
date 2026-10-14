"""Naive references for specs/gui-masks.md (1)-(5), (13), (14): explicit loops over pixels and
frames, Python integers and exact fractions; nothing from src/."""

import math
from fractions import Fraction

import numpy as np


def inside(rect, x: int, y: int) -> bool:
    """Half-open rectangle [x, x + w) x [y, y + h)."""
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def dist(a, b) -> int:
    """max over channels of |a - b| in Python integers."""
    return max(abs(int(a[ch]) - int(b[ch])) for ch in range(3))


def composite(blits: list, textures: dict, width: int, height: int) -> np.ndarray:
    """(1): RGBA canvas; with alpha 0 or 255 source-over keeps the topmost opaque pixel."""
    canvas = np.zeros((height, width, 4), dtype=np.int64)
    for b in blits:
        tex = textures[b["texture"]]
        for j in range(b["h"]):
            for i in range(b["w"]):
                if tex[b["v"] + j][b["u"] + i][3] == 255:
                    canvas[b["y"] + j, b["x"] + i] = tex[b["v"] + j][b["u"] + i]
    return canvas


def holes(cls: dict) -> list:
    """(1): hit rectangles, slot contents (x_s, y_s, 17, 18), dynamic and label rectangles."""
    rects = [e["hit"] for e in cls["elements"]]
    for e in cls["elements"]:
        if e["kind"] == "slot":
            rects.append([e["hit"][0] + 1, e["hit"][1] + 1, 17, 18])
    for m in cls["members"].values():
        rects += [r["rect"] for r in m["dynamic"] + m["labels"]]
    return rects


def outline_pairs(cls: dict, textures: dict, width: int, height: int) -> dict:
    """(1): member -> (RGBA canvas, mask), members with an empty mask left out."""
    areas = [cls["window"]] + ([cls["book_panel"]] if "book_panel" in cls else [])
    out = {}
    for name, member in cls["members"].items():
        canvas = composite(member["blits"], textures, width, height)
        mask = np.zeros((height, width), dtype=bool)
        for y in range(height):
            for x in range(width):
                shown = canvas[y, x, 3] == 255 and any(inside(a, x, y) for a in areas)
                mask[y, x] = shown and not any(inside(r, x, y) for r in holes(cls))
        if mask.any():
            out[name] = (canvas, mask)
    return out


def order_stat(values: list, q: float) -> int:
    """`quantile(..., method="inverted_cdf")`: the ceil(n q)-th smallest ([N])."""
    return sorted(values)[math.ceil(len(values) * q) - 1]


def spread_pair(frames: list, q: float, s_max: int) -> tuple:
    """(2): lower-median template, spread sigma, mask sigma <= s_max."""
    h, w = frames[0].shape[:2]
    t = np.zeros((h, w, 3), dtype=np.int64)
    sigma = np.zeros((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                t[y, x, ch] = order_stat([int(f[y, x, ch]) for f in frames], 0.5)
            sigma[y, x] = order_stat([dist(f[y, x], t[y, x]) for f in frames], q)
    return t, sigma, sigma <= s_max


def patch_masks(templates: dict, spreads: dict, p: int, n_p: int, eps) -> dict:
    """(3): union of the n_p patches of largest rho (exact), ties by top row then left column."""
    h, w = next(iter(spreads.values())).shape
    corners = [(top, left) for top in range(0, h, p) for left in range(0, w, p)]

    def rho(k: str, top: int, left: int) -> Fraction:
        cells = [(y, x) for y in range(top, top + p) for x in range(left, left + p)]
        sig = Fraction(sum(int(spreads[k][c]) for c in cells), p * p)
        deltas = [
            Fraction(sum(dist(templates[k][c], templates[j][c]) for c in cells), p * p)
            for j in templates
            if j != k
        ]
        return min(deltas) / (sig + Fraction(eps))

    out = {}
    for k in templates:
        best = sorted(corners, key=lambda c: (-rho(k, *c), c))[:n_p]
        out[k] = np.zeros((h, w), dtype=bool)
        for top, left in best:
            out[k][top : top + p, left : left + p] = True
    return out


def deviations(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d_T: max over channels of |x - t| in int64; on (f)'s E arrays (no channel axis)
    1000 d^f."""
    d = np.abs(x.astype(np.int64) - t.astype(np.int64))
    return d if d.ndim == 2 else d.max(axis=2)


def score(x: np.ndarray, pairs: list, bound: int) -> Fraction | None:
    """(4): best c / |M| over the pairs, c counting d <= bound (tau; 1000 tau for (f));
    None without a pair."""
    best = None
    for t, m in pairs:
        s = Fraction(int((m & (deviations(x, t) <= bound)).sum()), int(m.sum()))
        best = s if best is None or s > best else best
    return best


def feature(x: np.ndarray, t: np.ndarray, m: np.ndarray, q: float) -> int:
    """(14): the inverted_cdf quantile of the |M| deviations over the mask, repeats kept
    (1000 times the feature for (f))."""
    d = deviations(x, t)
    return order_stat([int(d[y, i]) for y, i in zip(*np.nonzero(m))], q)


def folds(episodes: list, k: int, seed: int) -> dict:
    """(13): episode pi[r] of the training episodes gets fold r mod K."""
    pi = np.random.Generator(np.random.PCG64(seed)).permutation(len(episodes))
    return {episodes[int(p)]: r % k for r, p in enumerate(pi)}


def predict(s: dict, order: list, theta: Fraction, mu: Fraction) -> str:
    """(5)."""
    if not s:
        return "other"
    ranked = sorted(s.values(), reverse=True)
    top = next(k for k in order if s.get(k) == ranked[0])
    second = ranked[1] if len(ranked) > 1 else 0
    return top if ranked[0] >= theta and ranked[0] - second >= mu else "other"
