"""`variance_pairs` and `patch_masks` of src/mcpref/recognition/masks.py against
specs/gui-masks.md (2), (3): hand cases and the naive references of masks_oracle.py on whole
frames; the `step` callbacks of the bars `templates b` and `templates c`."""

import math

import numpy as np
import pytest
from masks_oracle import patch_masks as naive_patch_masks
from masks_oracle import spread_pair

from mcpref.recognition.masks import patch_masks, variance_pairs


def frames_of(n: int, seed: int, shape=(6, 4)) -> np.ndarray:
    """n frames (frames x H x W x 3, uint8) with values near 0 and 255 and in between."""
    values = np.array([0, 1, 2, 30, 127, 128, 225, 253, 254, 255], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(n, *shape, 3))


class Steps:
    """A `step` callback (Interface): counts its calls, made without arguments."""

    def __init__(self):
        self.n = 0

    def __call__(self) -> None:
        self.n += 1


def blocks(frames: np.ndarray, chunk_rows: int):
    """(2): in row order, r0 and the rows [r0, r1) of every frame."""
    for r0 in range(0, frames.shape[1], chunk_rows):
        yield r0, frames[:, r0 : r0 + chunk_rows]


def check_builds(frames, builds: dict, q: float, s_max: int, chunk_rows: int) -> None:
    """(2) of every build against the naive reference on its whole frames; a build with a
    non-empty mask present (one with an empty mask leaves no template: present or not);
    `step` called once per row block."""
    step, height = Steps(), frames.shape[1]
    got = variance_pairs(blocks(frames, chunk_rows), builds, q, s_max, height, step)
    assert set(got) <= set(builds) and step.n == math.ceil(height / chunk_rows)
    for name, chosen in builds.items():
        t, sigma, mask = spread_pair(list(frames[chosen]), q, s_max)
        assert name in got or not mask.any(), name
        if name in got:
            gt, gs, gm = got[name]
            assert gt.dtype == np.uint8 and np.array_equal(gt, t), name
            assert np.array_equal(gs, sigma), name
            assert gm.dtype == bool and np.array_equal(gm, mask), name


def median_spread(frames, q: float) -> int:
    """A threshold equal to some sigma of the whole set."""
    sigma = spread_pair(list(frames), q, 0)[1]
    return int(np.sort(sigma, axis=None)[sigma.size // 2])


@pytest.mark.parametrize("n", [1, 2, 4, 5])
@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9, 1.0])
def test_variance_pairs_naive_reference(n, q):
    """(2), even and odd counts: lower median, ceil(n q)-th distance, sigma <= s_max."""
    frames = frames_of(n, seed=10 * n)
    check_builds(
        frames, {"all": np.ones(n, dtype=bool)}, q, median_spread(frames, q), 2
    )


@pytest.mark.parametrize("chunk_rows", [2, 4, 6, 8])
def test_variance_pairs_row_blocks(chunk_rows):
    """(2) is the same over row blocks of any height, the last one shorter (6 rows: 3, 2, 1, 1
    blocks, so as many `step` calls)."""
    frames = frames_of(5, seed=3)
    builds = {"all": np.ones(5, dtype=bool)}
    check_builds(frames, builds, 0.75, median_spread(frames, 0.75), chunk_rows)


SELECTORS = {  # build -> the frames it uses, of 7
    "all": [1, 1, 1, 1, 1, 1, 1],
    "fold0": [0, 1, 1, 0, 1, 1, 1],
    "fold1": [1, 0, 1, 1, 0, 1, 0],
    "fold2": [1, 1, 0, 1, 1, 0, 1],
    "fold3": [0, 0, 0, 0, 0, 0, 1],
}


@pytest.mark.parametrize("q", [0.5, 0.9])
def test_variance_pairs_every_build_from_the_same_blocks(q):
    """(2), (13): every build of a class from one pass over the row blocks, each over its own
    frames (`builds` selects them), including a build of one frame."""
    frames = frames_of(7, seed=11, shape=(8, 5))
    builds = {k: np.array(v, dtype=bool) for k, v in SELECTORS.items()}
    check_builds(frames, builds, q, median_spread(frames, q), 6)


def test_variance_pairs_hand_case():
    """Pixel 0: channel 0 = 255, 0, 200, 100 -> lower median 100, distances 155, 100, 100, 0;
    pixel 1: channel 2 = 255, 255, 255, 0 -> median 255, distances 0, 0, 0, 255 (not 1);
    two equal rows, one block."""
    pixels = [(255, 255), (0, 255), (200, 255), (100, 0)]
    frames = np.array([[[[a, 7, 7], [9, 9, b]]] * 2 for a, b in pixels], dtype=np.uint8)
    every = {"all": np.ones(4, dtype=bool)}
    step = Steps()
    t, sigma, mask = variance_pairs(blocks(frames, 2), every, 1.0, 155, 2, step)["all"]
    assert t.tolist() == [[[100, 7, 7], [9, 9, 255]]] * 2
    assert sigma.tolist() == [[155, 255]] * 2 and mask.tolist() == [[True, False]] * 2
    got = variance_pairs(blocks(frames, 2), every, 0.75, 99, 2, step)["all"]
    assert got[1].tolist() == [[100, 0]] * 2
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
