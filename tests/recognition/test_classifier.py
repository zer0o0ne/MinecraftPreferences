"""gamma of src/mcpref/recognition/classifier.py against specs/gui-classifier.md (1): hand
cases on fake gui-masks outputs (classifier_world.py): exact-fraction boundaries at theta and
mu, ties between pairs and between classes, values near 0 and 255, no score."""

import numpy as np
import pytest
from classifier_world import Masks, both, load

W, H = 5, 2
ZERO = np.zeros((H, W, 3), np.uint8)
ALL = np.ones((H, W), bool)
pytestmark = pytest.mark.usefixtures("away")


def grey(values: list) -> np.ndarray:
    """An H x W x 3 image whose pixel i (row-major) has value values[i] in every channel."""
    return np.array(values, np.uint8).reshape(H, W, 1).repeat(3, axis=2)


def matching(n: int, size: int = W * H) -> tuple:
    """A pair matching a black frame at tau = 0 on its first n of `size` mask pixels (the
    rest of the mask holds 9): c / |M| = n / size."""
    m = np.zeros(W * H, bool)
    m[:size] = True
    return grey([0] * n + [9] * (W * H - n)), m.reshape(H, W)


def gamma(root, repo, classes: dict, tau: int, theta: str, mu: str):
    """gamma of a fake gui-masks output with `classes` and (tau, theta, mu)."""
    return load(Masks(root, repo, classes, both(tau, theta, mu), W, H).write())[0]


BOUNDARIES = [  # theta, mu, label: scores 3/10, 1/10 (3/10 - 1/10 is not 0.2 in floats)
    ("0.3", "0.2", "zeta"),
    ("0.30000001", "0.2", "other"),
    ("0.3", "0.20000001", "other"),
    ("0.29999999", "0.19999999", "zeta"),
]


@pytest.mark.parametrize("theta, mu, label", BOUNDARIES)
def test_exact_fraction_boundaries(tmp_path, repo, theta, mu, label):
    """(1): S = 3/10 >= theta and 3/10 - 1/10 >= mu compared as exact fractions."""
    three = grey([4, 4, 4] + [5] * 7)  # d = 4 on 3 pixels: 3/10 at tau 4
    one = grey([0] + [200] * 9)  # 1/10
    classes = {"zeta": {"z": (three, ALL)}, "alpha": {"a": (one, ALL)}}
    assert gamma(tmp_path, repo, classes, 4, theta, mu)([ZERO]) == [label]


X = np.array(
    [
        [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0], [10, 10, 10]],
        [[100, 100, 100], [0, 255, 0], [3, 0, 0], [128, 128, 128], [250, 5, 0]],
    ],
    np.uint8,
)
T = np.array(
    [
        [[255, 255, 255], [0, 0, 0], [254, 254, 254], [1, 1, 1], [10, 10, 14]],
        [[100, 100, 100], [0, 0, 0], [0, 3, 0], [124, 132, 128], [255, 0, 5]],
    ],
    np.uint8,
)
# d = max over channels of |X - T| in a signed type: 255, 255, 1, 1, 4 / 0, 255, 3, 4, 5
COUNTS = [(0, 1), (1, 3), (3, 4), (4, 6), (5, 7), (254, 7), (255, 10)]  # tau, c of 10


@pytest.mark.parametrize("tau, c", COUNTS)
def test_count_near_0_and_255(tmp_path, repo, tau, c):
    """(1): c counts d <= tau, d the largest channel difference, without uint8
    wrap-around; one class: labelled iff c / 10 >= theta (mu = 0, S_(2) = 0)."""
    classes = {"only": {"m": (T, ALL)}}
    at = gamma(tmp_path / "at", repo, classes, tau, str(c / 10), "0")
    assert at([X]) == ["only"]
    if c < 10:
        above = gamma(tmp_path / "above", repo, classes, tau, str((c + 1) / 10), "0")
        assert above([X]) == ["other"]


@pytest.mark.parametrize("first", [True, False])
def test_class_score_is_its_best_pair(tmp_path, repo, first):
    """(1): S_one = max(1/4, 3/4) whatever the pair order; S_two = 1/2; the margin 1/4."""
    low, high = matching(1, 4), matching(3, 4)
    one = {"p": low, "q": high} if first else {"p": high, "q": low}
    classes = {"one": one, "two": {"r": matching(2, 4)}}
    assert gamma(tmp_path, repo, classes, 0, "0.75", "0.25")([ZERO]) == ["one"]


def test_tied_pairs(tmp_path, repo):
    """(1): pairs scoring 2/4 and 1/2 give S = 1/2; over 1/4, margin 1/4."""
    classes = {"half": {"p": matching(2, 4), "q": matching(1, 2)}}
    classes["low"] = {"r": matching(1, 4)}
    assert gamma(tmp_path, repo, classes, 0, "0.5", "0.25")([ZERO]) == ["half"]


@pytest.mark.parametrize(
    "scores, mu, label",
    [
        ((2, 3, 3), "0", "alpha"),  # zeta 2/4, alpha 3/4, mid 3/4: alpha first
        ((2, 3, 3), "0.01", "other"),  # S_(2) = 3/4: margin 0
        ((3, 3, 1), "0", "zeta"),  # catalogue order, not name order
        ((1, 2, 3), "0.25", "mid"),
    ],
)
def test_ties_between_classes_in_catalogue_order(tmp_path, repo, scores, mu, label):
    """(1): k* the first class of K with the largest S; S_(2) the second-largest score,
    equal to S_k* on a tie."""
    names = ("zeta", "alpha", "mid")
    classes = {k: {"m": matching(n, 4)} for k, n in zip(names, scores)}
    assert gamma(tmp_path, repo, classes, 0, "0", mu)([ZERO]) == [label]


def test_no_score_is_other(tmp_path, repo):
    """(1): no class with a (d) pair: `other` even at theta = mu = 0 (the classes' other
    variants, which match, are not read)."""
    classes = {"x": {"m": None}, "y": {"n": None, "o": None}}
    assert gamma(tmp_path, repo, classes, 7, "0", "0")([ZERO, X]) == ["other"] * 2


def test_class_without_pair_has_no_score(tmp_path, repo):
    """(1): x has no score (not 0): y, scoring 0, is k*, labelled at theta = mu = 0."""
    classes = {"x": {"m": None}, "y": {"n": matching(0)}}
    assert gamma(tmp_path, repo, classes, 0, "0", "0")([ZERO]) == ["y"]


def test_single_score_against_zero(tmp_path, repo):
    """(1): one score: S_(2) = 0, so the margin is S_k* = 1/2."""
    classes = {"x": {"m": None}, "y": {"n": matching(1, 2)}}
    assert gamma(tmp_path, repo, classes, 0, "0.5", "0.5")([ZERO]) == ["y"]
    other = gamma(tmp_path / "b", repo, classes, 0, "0.5", "0.50001")
    assert other([ZERO]) == ["other"]


def test_members_without_d_keys_are_not_pairs(tmp_path, repo):
    """(1): P_x is its member k alone (1/4); m, without (d) keys, has (a), (e), (f) arrays
    matching everywhere at tau 7, which are not pairs."""
    classes = {"x": {"m": None, "k": matching(1, 4)}, "y": {"n": matching(2, 4)}}
    assert gamma(tmp_path, repo, classes, 7, "0.5", "0.25")([ZERO]) == ["y"]
