"""src/mcpref/recognition/recognise.py against specs/gui-masks.md (4), (5), (14): hand cases in
exact fractions; the one-channel (f) form of (12)."""

from fractions import Fraction as Fr

import numpy as np
import pytest

from mcpref.recognition.recognise import pair_feature, predict, scores

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


def one_channel(values: list) -> np.ndarray:
    """A 1 x n image with a trailing channel axis of size 1, int64 (E arrays of (12))."""
    return np.array([[[v] for v in values]], dtype=np.int64)


E_X = one_channel([255000, 0, 123456, 7])
E_T = one_channel([250000, 1000, 120456, 7])  # |d| = 5000, 1000, 3000, 0
BOUNDS = [(2999, Fr(2, 4)), (3000, Fr(3, 4)), (3001, Fr(3, 4)), (5000, Fr(1))]


@pytest.mark.parametrize("bound, s", BOUNDS)
def test_score_one_channel_edges_at_1000_tau(bound, s):
    """(4), (12) for (f): one-channel int64 E arrays far above 255, the bound 1000 tau
    (tau = 3: 3000 counts, 2999 does not), |0 - 1000| counted both ways."""
    assert scores(E_X, {"k": [(E_T, ALL)]}, bound) == {"k": s}


PIXELS = [  # (x, t, in mask): d = max over channels of |x - t|
    ((0, 0, 0), (255, 0, 0), True),  # 255, not 1
    ((255, 255, 255), (255, 255, 255), True),  # 0
    ((10, 13, 10), (10, 10, 10), True),  # 3
    ((7, 7, 7), (4, 4, 4), True),  # 3
    ((255, 252, 255), (255, 255, 255), True),  # 3
    ((250, 50, 3), (50, 50, 3), True),  # 200
    ((0, 100, 0), (0, 0, 0), False),  # 100, outside the mask
    ((7, 0, 0), (0, 0, 0), False),  # 7, outside the mask
]
FX = np.array([[p[0] for p in PIXELS]], dtype=np.uint8)
FT = np.array([[p[1] for p in PIXELS]], dtype=np.uint8)
FM = np.array([[p[2] for p in PIXELS]])
QUANTILES = [  # masked d sorted: 0 3 3 3 200 255; the ceil(6 q)-th
    (0.1, 0),  # ceil(0.6) = 1
    (0.25, 3),  # ceil(1.5) = 2
    (0.5, 3),  # 3
    (0.66, 3),  # ceil(3.96) = 4
    (0.67, 200),  # ceil(4.02) = 5, just above the step
    (0.75, 200),  # ceil(4.5) = 5
    (1.0, 255),
]


@pytest.mark.parametrize("q, value", QUANTILES)
def test_pair_feature_inverted_cdf_with_repeats(q, value):
    """(14): inverted_cdf quantile at q of the |M| deviations over the mask, repeats kept,
    no uint8 wrap-around, pixels outside the mask ignored; a float."""
    got = pair_feature(FX, FT, FM, q)
    assert isinstance(got, float) and got == value


def test_pair_feature_one_channel_edges():
    """(14) for (f): on E arrays the quantile of |E(x_f) - T|, which the caller divides by
    1000: masked 1000 1500 1500 254000 at q = 0.5 -> 1500 -> 1.5."""
    x = one_channel([1000, 2500, 0, 254000, 9])
    t = one_channel([0, 1000, 1500, 0, 999999])
    m = np.array([[True, True, True, True, False]])
    got = pair_feature(x, t, m, 0.5)
    assert got == 1500.0 and got / 1000 == 1.5
