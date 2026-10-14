"""(4), (5), (14) of specs/gui-masks.md: scores, the predicted label and the features, per
frame and over the frames of a part; bars `features train`, `score <part>`."""

import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref.gui.label_files import OTHER, rgb_view
from mcpref.recognition.drawn_masks import SCALE, edges
from mcpref.recognition.masks import Pair, distance
from mcpref.recognition.pairs import FOLDED, Named

Scores = dict[int, list[dict[str, Fraction]]]
Point = tuple[int, Fraction, Fraction]
Pairs = Mapping[str, Sequence[Pair]]


def view(v: str, x: np.ndarray) -> np.ndarray:
    """Interface: the image variant v compares, E(x_f) with a channel axis of size 1 for
    (f), x_f otherwise."""
    return edges(x)[..., None] if v == "f" else x


def unit(v: str) -> int:
    """Interface: d^f is held in units of 1/1000, so the bound is 1000 tau in (4) and the
    quantile is divided by 1000 in (14)."""
    return SCALE if v == "f" else 1


def deviations(x: np.ndarray, pairs: Pairs) -> dict[str, list[np.ndarray]]:
    """(4), (14): d_T(f, p) over p in M, per pair of each class."""
    return {k: [distance(x[m], t[m]) for t, m in ps] for k, ps in pairs.items()}


def best(d: Mapping[str, Sequence[np.ndarray]], bound: int) -> dict[str, Fraction]:
    """(4): S_k = the largest c / |M| over k's pairs, c = |{p in M : d <= bound}|; classes
    without a pair have no score."""
    return {
        k: max(Fraction(int(np.count_nonzero(di <= bound)), di.size) for di in ds)
        for k, ds in d.items()
        if ds
    }


def scores(x: np.ndarray, pairs: Pairs, tau: int) -> dict[str, Fraction]:
    """(4): S_k(x) of the best pair of each class with a template."""
    return best(deviations(x, pairs), tau)


def predict(
    scores: Mapping[str, Fraction], order: Sequence[str], theta: Fraction, mu: Fraction
) -> str:
    """(5): k* if S_k* >= theta and S_k* - S_(2) >= mu, else `other`; ties: order."""
    ranked = [k for k in order if k in scores]
    if not ranked:
        return OTHER
    top = max(ranked, key=scores.__getitem__)
    second = max((scores[k] for k in ranked if k != top), default=Fraction(0))
    return top if scores[top] >= theta and scores[top] - second >= mu else OTHER


def quantile(d: np.ndarray, q: float) -> float:
    """(14): the inverted_cdf quantile at q of the |M| values d, as float64."""
    return float(np.quantile(d, q, method="inverted_cdf"))


def pair_feature(x: np.ndarray, t: np.ndarray, m: np.ndarray, q: float) -> float:
    """(14) for one pair: the quantile at q of d_T(f, p), p in M."""
    return quantile(distance(x[m], t[m]), q)


def columns(named: Mapping[str, Named]) -> dict[str, list[tuple[str, Pair]]]:
    """(14): per variant, (class, pair) of each column in column order."""
    return {
        v: [(k, p) for k, ps in by_class.items() for p in ps.values()]
        for v, by_class in named.items()
    }


def n_columns(named: Mapping[str, Named]) -> int:
    """(14): N_C."""
    return sum(len(c) for c in columns(named).values())


def train_row(
    x: np.ndarray, cols: dict[str, list[tuple[str, Pair]]], fold: dict, q: float
) -> list[float]:
    """(14) for a frame of F^tr_(l): (b), (c) columns use k's fold-l pair (NaN without
    one), the others the column's own pair."""
    row = []
    for v, vcols in cols.items():
        xv = view(v, x)
        for k, p in vcols:
            p = fold[v].get(k) if v in FOLDED else p
            row.append(np.nan if p is None else pair_feature(xv, *p, q) / unit(v))
    return row


def train_features(
    frames_dir: Path,
    size: tuple[int, int],
    named: Mapping[str, Named],
    folds: Mapping[int, dict],
    fold_of: Mapping[str, int],
    train: Sequence[dict],
    q: float,
) -> np.ndarray:
    """(14) over F^tr in order: frames x N_C, float64; bar `features train`."""
    if not train:
        print("features train: nothing to do", file=sys.stderr)
        return np.zeros((0, n_columns(named)))
    cols, rows = columns(named), []
    with tqdm(total=len(train), desc="features train", unit="frame") as bar:
        for f in train:
            x = rgb_view(frames_dir, f, size)
            rows.append(train_row(x, cols, folds[fold_of[f["relpath"]]], q))
            bar.update()
    return np.array(rows, dtype=np.float64)


def frame_record(
    x: np.ndarray, pairs: Pairs, v: str, tau_grid: Sequence[int], q: float
) -> tuple[dict[int, dict[str, Fraction]], list[float]]:
    """(4) at each tau of tau_grid and (14) with the column's own pairs, for variant v."""
    d = deviations(view(v, x), pairs)
    s = {tau: best(d, tau * unit(v)) for tau in tau_grid}
    return s, [quantile(di, q) / unit(v) for ds in d.values() for di in ds]


def part_scores(
    frames_dir: Path,
    size: tuple[int, int],
    tau_grid: Sequence[int],
    named: Mapping[str, Named],
    frames: Sequence[dict],
    part: str,
    q: float,
) -> tuple[dict[str, Scores], np.ndarray]:
    """(4) per variant and tau, (14) per frame of `part` in order (frames x N_C); bar
    `score <part>`."""
    if not frames:
        print(f"score {part}: nothing to do", file=sys.stderr)
        empty = {v: {tau: [] for tau in tau_grid} for v in named}
        return empty, np.zeros((0, n_columns(named)))
    recs = {v: [] for v in named}
    total = len(named) * len(frames)
    with tqdm(total=total, desc=f"score {part}", unit="frame") as bar:
        for v, by_class in named.items():
            pairs = {k: list(ps.values()) for k, ps in by_class.items()}
            for f in frames:
                x = rgb_view(frames_dir, f, size)
                recs[v].append(frame_record(x, pairs, v, tau_grid, q))
                bar.update()
    s = {
        v: {tau: [r[tau] for r, _ in rs] for tau in tau_grid} for v, rs in recs.items()
    }
    x = [np.array([row for _, row in rs], dtype=np.float64) for rs in recs.values()]
    return s, np.hstack(x)


def predictions(s: Scores, point: Point, order: Sequence[str]) -> list[str]:
    """(5) at the grid point (tau, theta, mu), for each scored frame in order."""
    tau, theta, mu = point
    return [predict(x, order, theta, mu) for x in s[tau]]
