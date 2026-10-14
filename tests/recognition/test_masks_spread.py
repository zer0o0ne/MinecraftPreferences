"""`variance_pair` and `patch_masks` of src/mcpref/recognition/masks.py against
specs/gui-masks.md (2), (3): hand cases and the naive references of masks_oracle.py; the `step`
callbacks of the bars `templates b` and `templates c`."""

import math

import numpy as np
import pytest
from masks_oracle import patch_masks as naive_patch_masks
from masks_oracle import spread_pair

from mcpref.recognition.masks import patch_masks, variance_pair

W, H = 40, 32  # canvas of the (1) cases


def frames_of(n: int, seed: int, shape=(5, 4)) -> list:
    """n frames with values near 0 and 255 and in between."""
    values = np.array([0, 1, 2, 30, 127, 128, 225, 253, 254, 255], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return list(rng.choice(values, size=(n, *shape, 3)))


class Steps:
    """A `step` callback (Interface): counts its calls, made without arguments."""

    def __init__(self):
        self.n = 0

    def __call__(self) -> None:
        self.n += 1


def check_variance(frames: list, q: float, chunk_rows: int) -> None:
    """(2) against the naive reference; `step` called once per row block."""
    t, sigma, _ = spread_pair(frames, q, 0)
    s_max = int(np.sort(sigma, axis=None)[sigma.size // 2])  # some sigma equals s_max
    step = Steps()
    got = variance_pair(frames, q, s_max, chunk_rows, step)
    assert got[0].dtype == np.uint8 and np.array_equal(got[0], t)
    assert np.array_equal(got[1], sigma)
    assert got[2].dtype == bool and np.array_equal(got[2], sigma <= s_max)
    assert step.n == math.ceil(len(frames[0]) / chunk_rows)


@pytest.mark.parametrize("n", [1, 2, 4, 5])
@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9, 1.0])
def test_variance_pair_naive_reference(n, q):
    """(2), even and odd counts: lower median, ceil(n q)-th distance, sigma <= s_max."""
    check_variance(frames_of(n, seed=10 * n), q, chunk_rows=2)


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 5, 9])
def test_variance_pair_row_blocks(chunk_rows):
    """(2) is the same over row blocks of any height, the last one shorter (5 rows: 5, 3, 2,
    1, 1 blocks, so as many `step` calls)."""
    check_variance(frames_of(5, seed=3), 0.75, chunk_rows)


def test_variance_pair_hand_case():
    """Pixel 0: channel 0 = 255, 0, 200, 100 -> lower median 100, distances 155, 100, 100, 0;
    pixel 1: channel 2 = 255, 255, 255, 0 -> median 255, distances 0, 0, 0, 255 (not 1)."""
    pixels = [(255, 255), (0, 255), (200, 255), (100, 0)]
    frames = [np.array([[[a, 7, 7], [9, 9, b]]], dtype=np.uint8) for a, b in pixels]
    step = Steps()
    t, sigma, mask = variance_pair(frames, 1.0, 155, 1, step)
    assert t.tolist() == [[[100, 7, 7], [9, 9, 255]]]
    assert sigma.tolist() == [[155, 255]] and mask.tolist() == [[True, False]]
    assert variance_pair(frames, 0.75, 99, 1, step)[1].tolist() == [[100, 0]]
    assert step.n == 2  # one row block per call


def test_patch_masks_hand_case():
    """(3), two classes, two 4x4 patches: delta(left) = |0 - 255| = 255 (channel 1), delta(right)
    = 20; rho_a = 255/(0+1) vs 20/(4+1) -> left; rho_b = 255/(254+1) vs 20/(0+1) -> right."""
    t_a = np.zeros((4, 8, 3), dtype=np.uint8)
    t_b = t_a.copy()
    t_b[:, :4, 1], t_b[:, 4:, 0] = 255, 20
    s_a = np.zeros((4, 8), dtype=np.int64)
    s_b = s_a.copy()
    s_a[:, 4:], s_b[:, :4] = 4, 254
    step = Steps()
    got = patch_masks({"a": t_a, "b": t_b}, {"a": s_a, "b": s_b}, 4, 1, 1.0, step)
    left = np.zeros((4, 8), dtype=bool)
    left[:, :4] = True
    assert set(got) == {"a", "b"} and step.n == 2  # one call per class's mask
    assert np.array_equal(got["a"], left) and np.array_equal(got["b"], ~left)


TIES = [(2, [(0, 0), (0, 4)]), (4, [(0, 0), (0, 4), (0, 8), (4, 0)])]


@pytest.mark.parametrize("n_p, corners", TIES)
def test_patch_masks_ties_by_top_row_then_left_column(n_p, corners):
    """(3): equal rho everywhere; the first n_p patches by (top row, left column)."""
    templates = {
        k: np.full((8, 12, 3), v, dtype=np.uint8) for k, v in (("a", 0), ("b", 10))
    }
    spreads = {k: np.zeros((8, 12), dtype=np.int64) for k in templates}
    expected = np.zeros((8, 12), dtype=bool)
    for top, left in corners:
        expected[top : top + 4, left : left + 4] = True
    got = patch_masks(templates, spreads, 4, n_p, 1.0, Steps())
    assert all(np.array_equal(got[k], expected) for k in templates)


@pytest.mark.parametrize("n_p", [1, 2, 6])
@pytest.mark.parametrize("eps", [1.0, 0.5])
def test_patch_masks_naive_reference(n_p, eps):
    """(3) with three classes: min over the others, means over patches, rho in exact arithmetic."""
    rng = np.random.default_rng(n_p)
    values = np.array([0, 1, 3, 128, 252, 254, 255], dtype=np.uint8)
    templates = {k: rng.choice(values, size=(8, 12, 3)) for k in ("a", "b", "c")}
    spreads = {k: rng.integers(0, 60, size=(8, 12)) for k in templates}
    step = Steps()
    got = patch_masks(templates, spreads, 4, n_p, eps, step)
    expected = naive_patch_masks(templates, spreads, 4, n_p, eps)
    assert set(got) == set(expected) and step.n == 3  # one call per class's mask
    assert all(np.array_equal(got[k], expected[k]) for k in expected)
    assert all(got[k].sum() == 16 * n_p for k in got)
