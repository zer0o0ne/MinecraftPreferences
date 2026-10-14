"""`texture_pairs` of src/mcpref/recognition/masks.py against specs/gui-masks.md (1): hand
pins and the naive reference of masks_oracle.py; (2), (3) in test_masks_spread.py."""

from fractions import Fraction

import numpy as np
import pytest
from masks_files import write_screens
from masks_oracle import outline_pairs
from masks_world import blit, member
from PIL import Image

from mcpref.gui.screens import load_screens
from mcpref.recognition.masks import texture_pairs
from mcpref.recognition.recognise import scores

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
            [blit("gui/pal.png", 2, 1, 4, 4, 8, 22)],
            labels=[("b", [10, 23, 2, 1], 0x404040)],
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
def box(tmp_path):
    """(texture_pairs on BOX, the oracle's member -> (canvas, mask))."""
    tex = textures(tmp_path / "tex")
    hidden = {**BOX, "members": {"hidden": BOX["members"]["hidden"]}}
    classes = {"box": BOX, "hidden_only": hidden}
    screens = load_screens(write_screens(tmp_path / "s.yaml", classes, "tex"))
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
