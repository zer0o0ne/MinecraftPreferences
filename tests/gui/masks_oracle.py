"""Naive references for specs/gui-masks.md (1)-(8): explicit loops over pixels and frames,
Python integers and exact fractions; nothing from src/."""

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


def score(x: np.ndarray, pairs: list, tau: int) -> Fraction | None:
    """(4): best c / |M| over the pairs; None without a pair."""
    best = None
    for t, m in pairs:
        d = np.abs(x.astype(np.int64) - t.astype(np.int64)).max(axis=2)
        s = Fraction(int((m & (d <= tau)).sum()), int(m.sum()))
        best = s if best is None or s > best else best
    return best


def predict(s: dict, order: list, theta: Fraction, mu: Fraction) -> str:
    """(5)."""
    if not s:
        return "other"
    ranked = sorted(s.values(), reverse=True)
    top = next(k for k in order if s.get(k) == ranked[0])
    second = ranked[1] if len(ranked) > 1 else 0
    return top if ranked[0] >= theta and ranked[0] - second >= mu else "other"


def criteria(y: list, yhat: list) -> dict:
    """(6), (7): agreement and macro recall (None on an empty set)."""
    if not y:
        return {"agreement": None, "macro_recall": None}
    recalls = [recall(y, yhat, k) for k in dict.fromkeys(y)]
    agree = Fraction(sum(a == b for a, b in zip(y, yhat)), len(y))
    return {"agreement": agree, "macro_recall": sum(recalls) / len(recalls)}


def recall(y: list, yhat: list, k: str) -> Fraction:
    return Fraction(sum(a == b == k for a, b in zip(y, yhat)), y.count(k))


def choose(y: list, preds: dict, grid: tuple) -> dict:
    """(6): criterion -> the first (tau, theta, mu) in grid order with the largest value;
    `preds[(tau, theta, mu)]` are the predictions on the validation frames."""
    out = {}
    for crit in ("agreement", "macro_recall"):
        best = None
        for point in ((t, th, m) for t in grid[0] for th in grid[1] for m in grid[2]):
            value = criteria(y, preds[point])[crit]
            if best is None or value > best[0]:
                best = (value, point)
        out[crit] = best[1]
    return out


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else float(Fraction(num, den))


def metrics(ids: list, y: list, yhat: list, labels: list) -> dict:
    """(7) on one set of frames: fractions as floats, errors sorted, no zero confusion cells."""
    crit = {k: None if v is None else float(v) for k, v in criteria(y, yhat).items()}
    per_label, confusion = {}, {}
    for k in labels:
        tp = sum(a == b == k for a, b in zip(y, yhat))
        n_true, n_pred = y.count(k), yhat.count(k)
        per_label[k] = {"n_true": n_true, "n_pred": n_pred}
        per_label[k].update(precision=ratio(tp, n_pred), recall=ratio(tp, n_true))
        per_label[k]["f1"] = ratio(2 * tp, n_true + n_pred)
        per_label[k]["errors"] = sorted(
            i for i, a, b in zip(ids, y, yhat) if a == k != b
        )
    for a, b in zip(y, yhat):
        confusion.setdefault(a, {}).setdefault(b, 0)
        confusion[a][b] += 1
    return {**crit, "per_label": per_label, "confusion": confusion}


def matrices(classes: list, labellers: list) -> dict:
    """(8): class -> (|F_k|, matrix as floats) for the classes some labeller assigns."""
    out = {}
    for k in classes:
        fk = [
            f for f in range(len(labellers[0])) if any(lab[f] == k for lab in labellers)
        ]
        if fk:
            agree = [
                [sum(li[f] == lj[f] for f in fk) for lj in labellers]
                for li in labellers
            ]
            out[k] = (
                len(fk),
                [[float(Fraction(n, len(fk))) for n in row] for row in agree],
            )
    return out
