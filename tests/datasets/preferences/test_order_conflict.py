"""Two-order contradictions, using an exhaustive hand-computed preference table."""

from fractions import Fraction

import pytest

from datasets.preferences.acyclic_world import MB, MW, SB, START, SW, B, W
from datasets.preferences.test_verdict import A, Bref, cfg
from mcpref.datasets.preferences.verdict import verdict

TOKENS = [MB, B, SB, SW, W, MW]
# Rows: order 0; columns: order 1; entries: p(A > B), or incomparable.
P = [
    [None, None, None, "0.8", "0.9", "1"],
    [None, None, None, "0.7", "0.8", "0.9"],
    [None, None, "0.5", "0.6", "0.7", "0.8"],
    ["0.2", "0.3", "0.4", "0.5", None, None],
    ["0.1", "0.2", "0.3", None, None, None],
    ["0", "0.1", "0.2", None, None, None],
]


@pytest.mark.parametrize("i", range(len(TOKENS)))
@pytest.mark.parametrize("j", range(len(TOKENS)))
def test_two_order_table(i, j):
    p = P[i][j]
    if p is None:
        expected = ("incomparable", A, Bref, None)
    else:
        p = Fraction(p)
        expected = (
            ("comparable", A, Bref, p)
            if p >= Fraction(1, 2)
            else ("comparable", Bref, A, 1 - p)
        )
    assert verdict(A, Bref, [TOKENS[i], TOKENS[j]], cfg()) == expected


def test_token_meaning_is_independent_of_numeric_weights():
    weights = {**START, B: 0.1, SB: 0.9}
    assert verdict(A, Bref, [B, SB], cfg(weights)) == ("incomparable", A, Bref, None)


def test_multiple_samples_keep_the_existing_mean():
    assert verdict(A, Bref, [B, B, SB, SB], cfg()) == (
        "comparable",
        A,
        Bref,
        Fraction(3, 5),
    )


def test_numeric_token_configured_as_refusal_is_not_a_conflict():
    weights = {B: None, SB: 0.6}
    assert verdict(A, Bref, [B, SB], cfg(weights)) == (
        "comparable",
        Bref,
        A,
        Fraction(3, 5),
    )
