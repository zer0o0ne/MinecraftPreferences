"""src/mcpref/recognition/drawn_masks.py against specs/gui-masks.md (9): `text_pixels` on the
hand-made font of drawn_world.py and on the real ascii.png, against hand cases and
drawn_oracle.py; (10)-(12) in test_drawn_masks.py."""

import drawn_oracle as do
import numpy as np
import pytest
from drawn_world import H, W, hand_font, label, pixels
from PIL import Image

from mcpref.gui.screens import load_screens
from mcpref.gui.textures import load_textures_config
from mcpref.recognition.drawn_masks import text_pixels

TEXTS = [  # text, rectangle, G_0, G_1 by hand from (9)
    (  # A at pen 0, b at pen adv(A) = w(A) + 1 = 5; no shadow at height 8
        "Ab",
        (2, 3, 7, 8),
        {(2, 4), (3, 4), (4, 4), (5, 6), (7, 5), (7, 6), (7, 7), (7, 8)},
        set(),
    ),
    (  # space: pen 4, nothing drawn; c: empty cell, advance 1; b at 5, ~ at 7
        " cb~",
        (1, 2, 16, 8),
        {(6, 4), (6, 5), (6, 6), (6, 7), (8, 2), (15, 9)},
        set(),
    ),
    ("d", (3, 3, 3, 9), {(3, 4), (4, 5)}, {(4, 5), (5, 6)}),  # shadow at offset 1
    (  # height 18: scale 2, 2 x 2 blocks, the shadow 2 px off
        "d",
        (3, 3, 6, 18),
        {(x, y) for x in (3, 4) for y in (5, 6)}
        | {(x, y) for x in (5, 6) for y in (7, 8)},
        {(x, y) for x in (5, 6) for y in (7, 8)}
        | {(x, y) for x in (7, 8) for y in (9, 10)},
    ),
]


@pytest.mark.parametrize("text, rect, g0, g1", TEXTS)
def test_text_pixels_hand_cases(text, rect, g0, g1):
    """(9): widths, advances, pen positions, shadow offset, scale."""
    got = text_pixels(label(text, rect), hand_font(), W, H)
    assert all(g.dtype == bool and g.shape == (H, W) for g in got)
    assert (pixels(got[0]), pixels(got[1])) == (g0, g1)


@pytest.mark.parametrize("h0", [8, 9, 18])
def test_text_pixels_naive_reference(h0):
    """(9) on every glyph kind at once, against drawn_oracle.text_pixels; 2 x 24 + 1 = 49
    columns at scale 2."""
    rect, w = (1, 1, 24 * (2 if h0 == 18 else 1), h0), 52
    got = text_pixels(label("A~d cb", rect), hand_font(), w, H)
    expected = do.text_pixels("A~d cb", rect, hand_font(), w, H)
    assert all(np.array_equal(a, b) for a, b in zip(got, expected))


def test_hand_font_widths():
    """(9): w = 1 + the last opaque column of the cell, 0 for an empty cell; adv = w + 1,
    4 for the space (the oracle's reading, which the hand cases above pin)."""
    phi = hand_font()
    widths = {ch: do.glyph_width(phi, ord(ch)) for ch in "Abd~c"}
    assert widths == {"A": 4, "b": 1, "d": 2, "~": 8, "c": 0}
    assert [do.advance(phi, ord(ch)) for ch in " c"] == [4, 1]


@pytest.fixture(scope="module")
def real_font(repo) -> np.ndarray:
    """Phi of the real font/ascii.png under the catalogue's textures directory."""
    path = load_textures_config(repo / "configs/gui-screens.yaml").out_dir
    path = path / "font/ascii.png"
    if not path.exists():
        pytest.skip(f"{path} is absent: scripts/download_data.sh fetches it")
    return np.asarray(Image.open(path).convert("RGBA"))[..., 3]


@pytest.fixture(scope="module")
def catalogue_labels(repo) -> list:
    screens = load_screens(repo / "configs/gui-screens.yaml")
    return [
        lab for c in screens.values() for m in c.members.values() for lab in m.labels
    ]


def test_real_font_label_widths(real_font, catalogue_labels):
    """[R]: omega * sum of adv(s_i) equals the rectangle width of every catalogue label."""
    for lab in catalogue_labels:
        omega = 2 if lab.rect.h == 18 else 1
        width = omega * sum(do.advance(real_font, ord(ch)) for ch in lab.rect.name)
        assert width == lab.rect.w, lab


def test_real_font_text_pixels_naive_reference(real_font, catalogue_labels):
    """(9) on the real font for every catalogue label, 640 x 360 ([R])."""
    for lab in catalogue_labels:
        r = lab.rect
        got = text_pixels(lab, real_font, 640, 360)
        expected = do.text_pixels(r.name, (r.x, r.y, r.w, r.h), real_font, 640, 360)
        assert all(np.array_equal(a, b) for a, b in zip(got, expected)), lab
