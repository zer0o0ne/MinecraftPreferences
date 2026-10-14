"""src/mcpref/recognition/drawn_masks.py against specs/gui-masks.md (10)-(12): hand cases on
the hand-made font of drawn_world.py and tiny textures, the naive references of
drawn_oracle.py; (9) in test_drawn_masks_text.py."""

import drawn_oracle as do
import numpy as np
import pytest
from drawn_world import H, W, hand_font, pixels, texture, write_class
from masks_world import blit, member, screen

from mcpref.recognition.drawn_masks import edge_pairs, edges, label_pairs, ring_pairs

HALF = [blit("gui/half.png", 0, 0, 20, H, 0, 0)]  # opaque left half, hole at (9, 8)
LABELLED = screen(
    {
        "first": member(
            HALF,
            dynamic=[[6, 1, 3, 4]],
            labels=[
                ("Ad", [2, 1, 8, 9], 0x8D5A2F),  # channels not multiples of 4
                ("b", [5, 2, 2, 8], 0x123456),  # drawn over A's (5, 4)
                ("~", [12, 9, 9, 8], 0xFFFFFF),  # over second's dynamic, x = 20 clear
                ("b", [24, 2, 2, 9], 0x404040),  # over transparent canvas
            ],
        ),
        "second": member(HALF, dynamic=[[13, 10, 4, 4]]),  # no label
        "third": member([], labels=[("A", [30, 12, 5, 8], 0xFFFFFF)]),  # no blit
    },
    [0, 0, W, H],
    (38, 22),
)


@pytest.fixture
def labelled(tmp_path) -> tuple:
    """(label_pairs, the oracle's pairs) of LABELLED."""
    images = {"gui/half.png": texture(20, 11, (9, 8))}
    cls = write_class(tmp_path, LABELLED, images)
    phi = hand_font()
    expected = do.label_pairs(LABELLED, {**images}, phi, W, H)
    return label_pairs(cls, tmp_path / "tex", W, H), expected


def test_label_pairs_naive_reference(labelled):
    """(10): the template with every label drawn in order, the mask; `second` skipped."""
    got, expected = labelled
    assert set(got) == set(expected) == {"first", "third"}
    for m, (t, mask) in expected.items():
        assert got[m][0].shape == (H, W, 3) and np.array_equal(got[m][0], t)
        assert got[m][1].shape == (H, W) and np.array_equal(got[m][1], mask)


PIN_COLOURS = [  # (x, y), RGB of `first`'s template
    ((8, 3), (0x8D, 0x5A, 0x2F)),  # in G_0 and G_1 of "Ad": the main pass on top
    ((3, 3), (0x8D // 4, 0x5A // 4, 0x2F // 4)),  # shadow only: floor(chi / 4)
    ((5, 4), (0x12, 0x34, 0x56)),  # A's pixel under the later label "b"
    ((25, 8), (16, 16, 16)),  # shadow over transparent canvas
    ((24, 3), (0, 0, 0)),  # transparent, no text
]
PIN_MASK = [  # (x, y), in `first`'s mask
    ((2, 1), True),  # label rectangle over opaque canvas
    ((6, 1), False),  # inside its member's dynamic rectangle
    ((7, 2), True),  # text inside that dynamic rectangle
    ((9, 8), False),  # the texture's hole inside the rectangle
    ((10, 1), False),  # outside every label rectangle
    ((14, 11), True),  # inside the dynamic rectangle of `second`
    ((20, 12), False),  # transparent canvas, no text
    ((24, 4), True),  # text over transparent canvas
    ((25, 8), True),  # its shadow
]


def test_label_pairs_hand_pins(labelled):
    """(9), (10) by hand: shadow under the main pass, floor(chi / 4), later labels on top;
    rectangle parts over opaque canvas outside the member's own dynamic rectangles, text."""
    t, mask = labelled[0]["first"]
    assert [tuple(int(v) for v in t[y, x]) for (x, y), _ in PIN_COLOURS] == [
        c for _, c in PIN_COLOURS
    ]
    assert [bool(mask[y, x]) for (x, y), _ in PIN_MASK] == [m for _, m in PIN_MASK]


def test_label_pairs_transparent_canvas_keeps_text_only(labelled):
    """(10): a member without blits: M = the text pixels; T black but on the text."""
    t, mask = labelled[0]["third"]
    assert pixels(mask) == {(30, 13), (31, 13), (32, 13), (33, 15)}
    assert (t[mask] == 255).all() and (t[~mask] == 0).all()


FULL = [blit("gui/full.png", 0, 0, W, H, 0, 0)]  # opaque, hole at (10, 20)
RINGED = screen(
    {
        "first": member(
            FULL, dynamic=[[0, 10, 3, 2]], labels=[("b", [15, 0, 3, 8], 0x404040)]
        ),
        "second": member(FULL, dynamic=[[19, 15, 4, 1]]),
        "third": member([]),  # transparent: empty masks
    },
    [0, 0, W, H],
    (38, 0),
    slots=[(2, 3), (20, 3)],
)


@pytest.fixture
def ringed(tmp_path):
    """(the loaded RINGED, its textures)."""
    images = {"gui/full.png": texture(W, 12, (10, 20))}
    return write_class(tmp_path, RINGED, images), images, tmp_path / "tex"


def frame_of(x: int, y: int) -> set:
    """(11): the 68 pixels of (x, y, 18, 18) minus (x + 1, y + 1, 16, 16), by hand."""
    box = {(x + i, y + j) for i in range(18) for j in range(18)}
    return box - {(x + i, y + j) for i in range(1, 17) for j in range(1, 17)}


def test_ring_pairs_one_slot(tmp_path):
    """(11): one slot on an opaque canvas without rectangles: the 68-pixel frame."""
    cls = screen({"only": member(FULL)}, [0, 0, W, H], (38, 0), slots=[(5, 2)])
    images = {"gui/full.png": texture(W, 13, (39, 23))}
    (t, mask), *rest = ring_pairs(
        write_class(tmp_path, cls, images), tmp_path / "tex", W, H
    ).values()
    assert rest == [] and len(frame_of(5, 2)) == 68 and pixels(mask) == frame_of(5, 2)
    assert np.array_equal(t, do.rgb(images["gui/full.png"]))


def test_ring_pairs_subtractions(ringed):
    """(11): two abutting slots minus the dynamic and label rectangles of every member and the
    hole; the slots' hit and content rectangles are not subtracted; `third` skipped."""
    cls, images, tex = ringed
    got = ring_pairs(cls, tex, W, H)
    removed = {
        (2, 10),
        (2, 11),
        (15, 3),
        (16, 3),
        (17, 3),
        (19, 15),
        (20, 15),
        (10, 20),
    }
    kept = (frame_of(2, 3) | frame_of(20, 3)) - removed
    assert set(got) == {"first", "second"} and len(kept) == 128
    assert all(pixels(got[m][1]) == kept for m in got)
    expected = do.ring_pairs(RINGED, images, W, H)
    assert all(np.array_equal(got[m][0], expected[m][0]) for m in got)


def luma_image() -> np.ndarray:
    """3 x 4 RGB; L = 1000 v for grey v, 299 / 587 / 114 for the unit colours."""
    rows = [
        [(10, 10, 10), (0, 0, 0), (1, 0, 0), (255, 255, 255)],
        [(0, 1, 0), (0, 0, 1), (20, 20, 20), (3, 3, 3)],
        [(5, 5, 5), (0, 0, 0), (0, 0, 0), (9, 9, 9)],
    ]
    return np.array(rows, dtype=np.uint8)


def test_edges_hand_case():
    """(12): L = 10000 0 299 255000 / 587 114 20000 3000 / 5000 0 0 9000; E = max(|right - L|,
    |below - L|): 10000 (right), 299, 254701, 4413 (below), 19886, 20000 (below); 0 in the
    last row and column."""
    e = edges(luma_image())
    assert e.dtype == np.int64 and e.shape == (3, 4)
    assert e.tolist() == [
        [10000, 299, 254701, 0],
        [4413, 19886, 20000, 0],
        [0, 0, 0, 0],
    ]


def test_edges_naive_reference():
    x = np.random.default_rng(14).integers(0, 256, (7, 9, 3)).astype(np.uint8)
    assert np.array_equal(edges(x), do.edges(x))


EDGE_PINS = [  # (x, y) in `first`'s (f) mask: p, p + (1, 0), p + (0, 1) all in M^a u M^e
    ((1, 5), True),  # (1, 5), (1, 6) outline; (2, 5) ring
    ((2, 5), False),  # (3, 5) inside the slot: in neither mask
    ((2, 3), True),  # the ring's corner, its right and lower neighbours
    ((1, 9), False),  # (1, 10) in a dynamic rectangle
    ((38, 21), True),
    ((39, 21), False),  # last column
]


def test_edge_pairs(ringed):
    """(12): E of Gamma-bar and the mask rule over the masks of (1) and (11), against
    drawn_oracle.edge_pairs; hand pins; `third` (both masks empty) skipped."""
    cls, images, tex = ringed
    got = edge_pairs(cls, tex, W, H)
    expected = do.edge_pairs(RINGED, images, W, H)
    assert set(got) == set(expected) == {"first", "second"}
    for m, (t, mask) in expected.items():
        assert np.array_equal(got[m][0], t) and np.array_equal(got[m][1], mask)
    mask = got["first"][1]
    assert [bool(mask[y, x]) for (x, y), _ in EDGE_PINS] == [m for _, m in EDGE_PINS]
