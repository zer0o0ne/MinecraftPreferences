"""(1) of specs/acyclic.md: the verdict of a comparison from the labels of its requests,
in exact arithmetic."""

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

from mcpref import exact

if TYPE_CHECKING:
    from mcpref.preferences.acyclic import AcyclicConfig

HALF = Fraction(1, 2)
Verdict = tuple[str, dict, dict, Fraction | None]


def tilde(t: str, o: int, cfg: "AcyclicConfig") -> Fraction:
    """(1): the weight of label t read for (a, b): pi(t) for order 0, 1 - pi(t) for 1."""
    pi = exact(f"weights.{t}", cfg.weights[t])
    return pi if o == 0 else 1 - pi


def verdict(a: dict, b: dict, labels: Sequence[str], cfg: "AcyclicConfig") -> Verdict:
    """(1): `dropped`, `incomparable` or `comparable`; the pair (u, v), else (a, b); w
    (None unless comparable). Request j of the 2s labels has order j // s."""
    valid = [(j, t) for j, t in enumerate(labels) if t in cfg.weights]
    weighted = [(j, t) for j, t in valid if cfg.weights[t] is not None]
    n_v, n_r = len(valid), len(valid) - len(weighted)
    if n_v == 0:
        return "dropped", a, b, None
    if Fraction(n_r, n_v) > exact("refusal_share", cfg.refusal_share):
        return "incomparable", a, b, None
    s = len(labels) // 2
    p = sum(tilde(t, j // s, cfg) for j, t in weighted) / (n_v - n_r)
    if p >= HALF:
        return "comparable", a, b, p
    return "comparable", b, a, 1 - p
