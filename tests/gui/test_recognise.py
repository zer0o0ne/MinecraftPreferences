"""src/mcpref/gui/recognise.py against specs/gui-masks.md (4)-(5): hand cases in exact
fractions."""

from fractions import Fraction as Fr

import numpy as np
import pytest

from mcpref.gui.recognise import predict, scores

X = np.array(
    [[[0, 0, 0], [255, 255, 255], [10, 10, 10], [100, 100, 100]]], dtype=np.uint8
)
T = np.array(
    [[[255, 0, 0], [0, 255, 255], [10, 10, 14], [100, 100, 100]]], dtype=np.uint8
)
ALL = np.ones((1, 4), dtype=bool)


TAUS = [(0, Fr(1, 4)), (3, Fr(1, 4)), (4, Fr(2, 4)), (254, Fr(2, 4)), (255, Fr(1))]


@pytest.mark.parametrize("tau, s", TAUS)
def test_score_tau_inclusive_without_wraparound(tau, s):
    """(4): d = 255, 255 (|0 - 255|, not 1), 4 (max over channels), 0; c counts d <= tau."""
    got = scores(X, {"k": [(T, ALL)]}, tau)
    assert got == {"k": s} and isinstance(got["k"], Fr)


def test_score_best_pair_by_ratio_not_count():
    """(4): pair 1 matches 2 of its 4 pixels at tau 4 (2/4); pair 2 its 1 pixel (1/1), pixels
    outside its mask ignored; S is that of the best pair, in either order."""
    one = np.array([[False, False, True, False]])
    for pairs in ([(T, ALL), (T, one)], [(T, one), (T, ALL)]):
        assert scores(X, {"k": pairs}, 4) == {"k": Fr(1)}
    assert scores(X, {"k": [(T, ~one)]}, 4) == {"k": Fr(1, 3)}


def test_score_classes_without_template():
    """(4): a class without a pair has no score."""
    got = scores(X, {"a": [], "b": [(T, ALL)], "c": []}, 0)
    assert got == {"b": Fr(1, 4)}
    assert scores(X, {"a": []}, 0) == {}


ORDER = ["a", "b", "c", "d"]
CASES = [  # scores, theta, mu, prediction
    ({}, Fr(0), Fr(0), "other"),  # no score
    (
        {"b": Fr(3, 10), "a": Fr(1, 10)},
        Fr(3, 10),
        Fr(1, 5),
        "b",
    ),  # 3/10 - 1/10 = 1/5 exactly
    ({"b": Fr(3, 10), "a": Fr(1, 10)}, Fr(31, 100), Fr(1, 5), "other"),  # S < theta
    ({"b": Fr(3, 10), "a": Fr(1, 10)}, Fr(3, 10), Fr(21, 100), "other"),  # margin < mu
    ({"c": Fr(1, 2)}, Fr(1, 2), Fr(1, 2), "c"),  # only one score: S_(2) = 0
    ({"c": Fr(1, 2)}, Fr(0), Fr(51, 100), "other"),
    ({"c": Fr(1, 2), "b": Fr(1, 2)}, Fr(1, 2), Fr(0), "b"),  # tie: catalogue order
    ({"c": Fr(1, 2), "b": Fr(1, 2)}, Fr(0), Fr(1, 100), "other"),  # tie: S_(2) = S_k*
    ({"a": Fr(1, 5), "d": Fr(4, 5), "c": Fr(3, 5)}, Fr(4, 5), Fr(1, 5), "d"),  # vs 3/5
    ({"a": Fr(1, 5), "d": Fr(4, 5), "c": Fr(3, 5)}, Fr(4, 5), Fr(21, 100), "other"),
    ({"a": Fr(0), "b": Fr(0)}, Fr(0), Fr(0), "a"),
]


@pytest.mark.parametrize("s, theta, mu, label", CASES)
def test_predict_hand_cases(s, theta, mu, label):
    """(5): k* by the largest score (ties: catalogue order); S >= theta and S - S_(2) >= mu."""
    assert predict(s, ORDER, theta, mu) == label


def test_predict_tie_follows_given_order():
    s = {"c": Fr(1, 2), "b": Fr(1, 2)}
    assert predict(s, ["d", "c", "b", "a"], Fr(0), Fr(0)) == "c"
