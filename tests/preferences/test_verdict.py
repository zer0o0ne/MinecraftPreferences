"""`verdict` of specs/acyclic.md (Interface, `preferences/verdict.py`): (1) on hand cases in
exact arithmetic (derivations in comments) and against the transcription of
acyclic_oracle.py on random labels and decimal weights."""

import random
from fractions import Fraction

import pytest

from mcpref.preferences.acyclic import AcyclicConfig, load_config
from mcpref.preferences.verdict import verdict

from . import acyclic_oracle as oracle
from .acyclic_world import INC, MB, MW, SB, START, SW, B, W, ref

A, Bref = ref("a_1"), ref("b_1")
INVALID, NONE = "invalid", "no_answer"


def cfg(weights: dict = START, tau=0.5) -> AcyclicConfig:
    return AcyclicConfig("w/votes", "w/labels", weights, tau)


# (labels in request order, tau, expected); START weights 1, .8, .6, .4, .2, 0, null
HAND = [
    ([INC, INC], 0.5, ("incomparable", A, Bref, None)),  # all refusals: share 1 > 1/2
    # share 1/2 = tau, not above: pi~ = 1 - 0.8 (order 1), p = 1/5 -> reversed, w = 4/5
    ([INC, B], 0.5, ("comparable", Bref, A, Fraction(4, 5))),
    # s = 2, share 2/4 = tau: pi~ = 0.8, 1 - 0.6; p = 3/5
    ([INC, B, INC, SB], 0.5, ("comparable", A, Bref, Fraction(3, 5))),
    ([INC, INC, INC, B], 0.5, ("incomparable", A, Bref, None)),  # share 3/4 > 1/2
    ([INVALID, NONE], 0.5, ("dropped", A, Bref, None)),  # n_v = 0
    ([INVALID, INVALID, NONE, NONE], 0.0, ("dropped", A, Bref, None)),
    # invalid and no_answer ignored in the mean: pi~ = 0.8, 1 - 0.2; p = 4/5
    ([B, INVALID, NONE, W], 0.5, ("comparable", A, Bref, Fraction(4, 5))),
    # ... and in the share: 1/2 > 0.4 (1/4 if they counted)
    ([INC, INVALID, NONE, B], 0.4, ("incomparable", A, Bref, None)),
    # both orders: 0.8 and 1 - 0.8; p = 1/2 exactly -> (a, b)
    ([B, B], 0.5, ("comparable", A, Bref, Fraction(1, 2))),
    # the spec's case: 1, 0.4 in order 0, 0.6, 0.8 in order 1: (1 + .4 + .4 + .2) / 4 = 1/2
    # exactly (0.49999999999999994 in float64)
    ([MB, SW, SB, B], 0.5, ("comparable", A, Bref, Fraction(1, 2))),
    ([W, MB], 0.5, ("comparable", Bref, A, Fraction(9, 10))),  # p = (.2 + 0) / 2 = 1/10
    ([MW, MB], 0.5, ("comparable", Bref, A, Fraction(1))),  # p = 0 -> w = 1
    ([MB, MW], 0.5, ("comparable", A, Bref, Fraction(1))),  # p = 1
    # (1 + 1 + (1 - .8)) / 3 = 11/15 (float64 arithmetic: 0.7333333333333334)
    ([MB, MB, B, INVALID], 0.5, ("comparable", A, Bref, Fraction(11, 15))),
    ([INC, MB], 0, ("incomparable", A, Bref, None)),  # tau = 0: one refusal is above it
    ([MB, SW], 0, ("comparable", A, Bref, Fraction(4, 5))),  # (1 + .6) / 2
    # s = 5, share 3/10 = tau 0.3 exactly (above it if 0.3 were read as a float64)
    ([INC, INC, INC, MB, MB, MW, MW, MW, MB, SW], 0.3,
     ("comparable", A, Bref, Fraction(4, 5))),  # (1 + 1 + 1 + 1 + 1 + 0 + .6) / 7
]  # fmt: skip


@pytest.mark.parametrize("labels, tau, expected", HAND)
def test_hand_cases(labels, tau, expected):
    """(1) by hand; w an exact Fraction."""
    got = verdict(A, Bref, labels, cfg(tau=tau))
    assert got == expected
    assert got[3] is None or type(got[3]) is Fraction


def test_hand_cases_agree_with_the_transcription():
    for labels, tau, expected in HAND:
        assert oracle.verdict(A, Bref, labels, START, tau) == expected, labels


def test_decimal_weights_exactly():
    """Weights 0.7, 0.3 (order 0) average to 1/2 exactly: (a, b) (as float64 values read
    exactly they would fall below 1/2)."""
    weights = {"UP": 0.7, "DOWN": 0.3, "NO": None}
    got = verdict(A, Bref, ["UP", "DOWN", INVALID, INVALID], cfg(weights))
    assert got == ("comparable", A, Bref, Fraction(1, 2))


def test_config_loaded_from_yaml(tmp_path):
    """The exact value of a YAML number is the decimal as written: weights `1.0e-1`, `0.9`,
    `1`, `0` and refusal_share `3.0e-1` loaded by `load_config`."""
    path = tmp_path / "c.yaml"
    path.write_text(
        'votes_dir: "v"\nout_dir: "o"\n'
        "weights:\n  X: 1.0e-1\n  Y: 0.9\n  ONE: 1\n  ZERO: 0\n  R: null\n"
        "refusal_share: 3.0e-1\n"
    )
    c = load_config(path)
    # (.1 + (1 - .9)) / 2 = 1/10 -> reversed, w = 9/10; ZERO in order 1: 1 - 0
    assert verdict(A, Bref, ["X", "Y"], c) == ("comparable", Bref, A, Fraction(9, 10))
    assert verdict(A, Bref, ["ONE", "ZERO"], c) == ("comparable", A, Bref, Fraction(1))
    labels = ["R", "R", "R", "X", "X", "X", "X", "X", "X", "X"]  # share 3/10 = tau
    assert verdict(A, Bref, labels, c)[0] == "comparable"


def test_exact_value_is_the_shortest_repr_of_the_float_read(tmp_path):
    """Notation: `0.69999999999999999` and `0.29999999999999999` read as the floats 0.7
    and 0.3, exact values 7/10 and 3/10 (the decimals as written, and the floats' own
    values, are below them): UP, DOWN in order 0 average to 1/2 -> (a, b); a refusal
    share 3/10 is not above tau."""
    path = tmp_path / "c.yaml"
    path.write_text(
        'votes_dir: "v"\nout_dir: "o"\n'
        "weights:\n  UP: 0.69999999999999999\n  DOWN: 0.3\n  R: null\n"
        "refusal_share: 0.29999999999999999\n"
    )
    c = load_config(path)
    assert (c.weights["UP"], c.refusal_share) == (0.7, 0.3)
    got = verdict(A, Bref, ["UP", "DOWN", INVALID, INVALID], c)  # s = 2
    assert got == ("comparable", A, Bref, Fraction(1, 2))
    # s = 5; order 0: R R R UP UP, order 1: DOWN x 5 (1 - 0.3); p = (2 .7 + 5 .7) / 7
    labels = ["R", "R", "R", "UP", "UP"] + ["DOWN"] * 5
    assert verdict(A, Bref, labels, c) == ("comparable", A, Bref, Fraction(7, 10))


def random_weights(rng: random.Random) -> dict:
    """2-6 categories with decimal weights in [0, 1] (1-2 digits) and one refusal."""
    tokens = [f"T{i}" for i in range(rng.randrange(1, 6))]
    weights = {
        t: rng.choice([rng.randrange(11) / 10, rng.randrange(101) / 100])
        for t in tokens
    }
    weights["REFUSE"] = None
    return dict(rng.sample(list(weights.items()), len(weights)))


def test_random_cases_match_the_transcription():
    """(1) against the transcription on random weights, tau, s and labels."""
    rng = random.Random(0)
    for _ in range(3000):
        weights = random_weights(rng)
        tau = rng.choice([0, 0.25, 0.3, 0.5, rng.randrange(100) / 100])
        s = rng.randrange(1, 4)
        pool = [*weights, INVALID, NONE]
        labels = [rng.choice(pool) for _ in range(2 * s)]
        expected = oracle.verdict(A, Bref, labels, weights, tau)
        assert verdict(A, Bref, labels, cfg(weights, tau)) == expected, (
            weights,
            tau,
            labels,
        )
