"""src/mcpref/gui/masks.py against specs/gui-masks.md (1)-(3): hand cases and the naive
references of masks_oracle.py; the `step` callbacks of the bars `templates b` and `templates c`."""

import math
from fractions import Fraction

import numpy as np
import pytest
from masks_oracle import outline_pairs, spread_pair
from masks_oracle import patch_masks as naive_patch_masks
from masks_world import blit, member
from PIL import Image

from mcpref.gui.masks import patch_masks, texture_pairs, variance_pair
from mcpref.gui.recognise import scores
from mcpref.gui.screens import load_screens

W, H = 40, 32  # canvas of the (1) cases


BOX = {  # window, book panel, a slot and a button; `hidden` lies inside the slot's hit rectangle
    "members": {
        "first": member(
            [
                blit("gui/rgba.png", 1, 2, 38, 30, 1, 1),
                blit("gui/pal.png", 0, 0, 6, 5, 22, 12),
            ],
            dynamic=[[9, 24, 3, 2]],
        ),
        "second": member(
            [blit("gui/pal.png", 2, 1, 4, 4, 8, 22)], labels=[[10, 23, 2, 1]]
        ),
        "hidden": member([blit("gui/rgba.png", 0, 0, 10, 10, 4, 4)]),
    },
    "description": "Box.",
    "window": [0, 0, 30, 26],
    "book_panel": [30, 0, 8, 10],
    "elements": [
        {
            "index": 1,
            "name": "s",
            "kind": "slot",
            "hit": [2, 2, 18, 18],
            "click": [11.0, 11.0],
        },
        {
            "index": 37,
            "name": "b",
            "kind": "button",
            "hit": [22, 2, 6, 4],
            "click": [25.0, 4.0],
        },
    ],
}
PINS = [  # first member: (x, y, in mask), by hand from (1)
    (2, 20, True),  # below the slot's hit rectangle, left of its content column x_s = 3
    (3, 20, False),  # slot content row y_s + 17
    (19, 20, False),
    (20, 20, True),
    (22, 5, False),  # button hit rectangle
    (22, 6, True),  # a button has no content row
    (29, 25, True),  # window corner
    (30, 25, False),
    (30, 9, True),  # book panel
    (37, 9, True),
    (30, 10, False),
    (10, 23, False),  # label rectangle of the member `second`
    (9, 25, False),  # its own dynamic rectangle
]


def textures(root) -> dict:
    """Writes gui/rgba.png (RGBA, alpha 0 where (3x + y) % 7 == 0) and gui/pal.png (palette,
    index 0 transparent); returns them as RGBA integer arrays, the oracle's input."""
    rng = np.random.default_rng(5)
    rgba = rng.integers(0, 256, (34, 40, 4))
    ys, xs = np.mgrid[0:34, 0:40]
    rgba[..., 3] = np.where((3 * xs + ys) % 7 == 0, 0, 255)
    idx = rng.integers(0, 4, (5, 6))
    palette = np.array([[0, 0, 0], [10, 20, 30], [200, 100, 50], [255, 254, 1]])
    pal = np.concatenate([palette[idx], np.where(idx == 0, 0, 255)[..., None]], axis=2)
    (root / "gui").mkdir(parents=True)
    Image.fromarray(rgba.astype(np.uint8)).save(root / "gui/rgba.png")
    image = Image.new("P", (6, 5))
    image.putdata(idx.flatten().tolist())
    image.putpalette(palette.flatten().tolist())
    image.save(root / "gui/pal.png", transparency=0)
    return {"gui/rgba.png": rgba, "gui/pal.png": pal}


@pytest.fixture
def box(tmp_path, catalogue, write_config):
    """(texture_pairs on BOX, the oracle's member -> (canvas, mask))."""
    tex = textures(tmp_path / "tex")
    hidden = {**BOX, "members": {"hidden": BOX["members"]["hidden"]}}
    catalogue["classes"] = {"box": BOX, "hidden_only": hidden}
    screens = load_screens(write_config(catalogue))
    got = {k: texture_pairs(screens[k], tmp_path / "tex", W, H) for k in screens}
    return got, outline_pairs(BOX, tex, W, H)


def test_texture_pairs_hand_pins(box):
    got, expected = box
    canvas, mask = expected["first"]
    assert all(
        canvas[y, x, 3] == 255 for x, y, _ in PINS
    )  # the pins test the rectangles
    assert [bool(mask[y, x]) for x, y, _ in PINS] == [m for _, _, m in PINS]
    assert any(np.array_equal(m, mask) for _, m in got["box"])


def test_texture_pairs_naive_reference(box):
    """(1): compositing in draw order, every subtraction, union over members, `hidden` skipped."""
    got, expected = box
    assert sorted(expected) == ["first", "second"] and len(got["box"]) == 2
    for canvas, mask in expected.values():
        (t, m), *rest = [(t, m) for t, m in got["box"] if np.array_equal(m, mask)]
        assert rest == [] and m.dtype == bool and m.shape == (H, W)
        assert np.array_equal(t[m][:, :3], canvas[mask][:, :3])


def test_texture_pairs_feed_scores(box):
    """(1) -> (4): the canvas colours score 1 at tau 0; one masked pixel off by one costs 1/|M|."""
    got, expected = box
    canvas, mask = expected["second"]
    pair = next(p for p in got["box"] if np.array_equal(p[1], mask))
    x = canvas[..., :3].astype(np.uint8)
    assert scores(x, {"box": [pair]}, 0) == {"box": Fraction(1)}
    y, i = np.argwhere(mask & (canvas[..., 0] < 255))[0]
    x[y, i, 0] += 1
    assert scores(x, {"box": [pair]}, 0) == {"box": 1 - Fraction(1, int(mask.sum()))}


def test_texture_pairs_class_without_pair(box):
    assert box[0]["hidden_only"] == []


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
