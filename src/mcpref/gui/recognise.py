"""(4)-(5) of specs/gui-masks.md: template scores and the predicted label."""

from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

from mcpref.gui.label_files import OTHER
from mcpref.gui.masks import distance


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
