"""(4)-(5) of specs/gui-masks.md: template scores and the predicted label, per frame and
over the frames of a part."""

import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref.gui.label_files import OTHER, rgb_view
from mcpref.gui.masks import distance

Scores = dict[int, list[dict[str, Fraction]]]
Point = tuple[int, Fraction, Fraction]


def pair_score(x: np.ndarray, t: np.ndarray, m: np.ndarray, tau: int) -> Fraction:
    """(4): c / |M| with c = |{p in M : d_T(x, p) <= tau}|, exact."""
    c = np.count_nonzero(distance(x[m], t[m]) <= tau)
    return Fraction(int(c), int(np.count_nonzero(m)))


def scores(
    x: np.ndarray,
    pairs: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
    tau: int,
) -> dict[str, Fraction]:
    """(4): S_k(x) of the best pair of each class with a template."""
    return {
        k: max(pair_score(x, t, m, tau) for t, m in ps) for k, ps in pairs.items() if ps
    }


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


def frame_scores(
    frames_dir: Path,
    size: tuple[int, int],
    tau_grid: Sequence[int],
    pairs: Mapping[str, Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]]],
    frames: Sequence[dict],
    part: str,
) -> dict[str, Scores]:
    """(4) per variant and tau of tau_grid, for each frame of `part` in order; bar
    `score <part>`."""
    out = {v: {tau: [] for tau in tau_grid} for v in pairs}
    if not frames:
        print(f"score {part}: nothing to do", file=sys.stderr)
        return out
    with tqdm(
        total=len(pairs) * len(frames), desc=f"score {part}", unit="frame"
    ) as bar:
        for v, p in pairs.items():
            for f in frames:
                x = rgb_view(frames_dir, f, size)
                for tau in tau_grid:
                    out[v][tau].append(scores(x, p, tau))
                bar.update()
    return out


def predictions(s: Scores, point: Point, order: Sequence[str]) -> list[str]:
    """(5) at the grid point (tau, theta, mu), for each scored frame in order."""
    tau, theta, mu = point
    return [predict(x, order, theta, mu) for x in s[tau]]
