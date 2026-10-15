"""src/mcpref/recognition/debug_overlay.py against specs/f3-overlay.md: Notation (P, R, the
dilation, Q) by `overlay_masks` on the hand-made font of drawn_world.py, `luma`,
`overlay_font`, and the real ascii.png; (1)-(3) in test_debug_overlay_score.py, the config
in test_debug_overlay_config.py."""

import os

import numpy as np
import overlay_oracle as oo
import pytest
import yaml
from drawn_world import hand_font, pixels
from masks_files import write_screens, write_textures
from masks_world import font_image
from PIL import Image

from mcpref.recognition.debug_overlay import (
    load_config,
    luma,
    overlay_font,
    overlay_masks,
    overlay_shown,
)


def box(x0: int, x1: int, y0: int, y1: int) -> set:
    """The (x, y) of [x0, x1] x [y0, y1], inclusive."""
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


MASKS = [  # prefix, W, H, P, Q by hand from Notation (pen at (2, 2); R = [1, 2 + w) x [1, 10))
    # b: column 0, rows 2..5 -> x = 2, y = 4..7; w = adv(b) = 2; dilation x 1..3, y 3..8
    ("b", 6, 12, box(2, 2, 4, 7), box(1, 3, 1, 9) - box(1, 3, 3, 8)),
    # trailing space: w = 2 + 4, R up to x = 7, nothing drawn for it
    ("b ", 9, 12, box(2, 2, 4, 7), box(1, 7, 1, 9) - box(1, 3, 3, 8)),
    # leading space: pen 4, b at x = 6; w = 6
    (" b", 9, 12, box(6, 6, 4, 7), box(1, 7, 1, 9) - box(5, 7, 3, 8)),
    # c: empty cell, advance 1; b at x = 3; w = 3
    ("cb", 6, 12, box(3, 3, 4, 7), box(1, 4, 1, 9) - box(2, 4, 3, 8)),
    # A at pen 0 (row 1 cols 0..2, (3, 3)), b at pen adv(A) = 5; w = 7
    ("Ab", 12, 12, box(2, 4, 3, 3) | {(5, 5)} | box(7, 7, 4, 7), None),
    # R fills the frame (W = 2 + 9, H = 10): row 7 of "1" at y = 9, its dilation to y = 10
    ("1", 11, 10, box(2, 9, 9, 9), box(1, 10, 1, 7)),
    # the corners of "~" at (2, 2), (9, 9); 90 - 9 - 6 pixels of R left
    ("~", 11, 10, {(2, 2), (9, 9)}, box(1, 10, 1, 9) - box(1, 3, 1, 3) - box(8, 10, 8, 9)),
]  # fmt: skip


@pytest.mark.parametrize("prefix, w, h, p, q", MASKS)
def test_overlay_masks_hand_cases(prefix, w, h, p, q):
    """Notation: P (positions, advance, space), R, the 3 x 3 dilation, Q = R minus it."""
    got = overlay_masks(hand_font(), prefix, w, h)
    assert len(got) == 2
    assert all(m.dtype == bool and m.shape == (h, w) for m in got)
    assert pixels(got[0]) == p
    expected_q = pixels(oo.masks(hand_font(), prefix, w, h)[1]) if q is None else q
    assert pixels(got[1]) == expected_q


@pytest.mark.parametrize(
    "prefix, w, h",
    [("A~d cb@1Q", 60, 16), ("Q1", 20, 10), ("~ ~", 40, 30), ("@B", 13, 10)],
)
def test_overlay_masks_naive_reference(prefix, w, h):
    """Notation against overlay_oracle.masks, also with R touching the right and bottom
    borders of the frame."""
    got = overlay_masks(hand_font(), prefix, w, h)
    expected = oo.masks(hand_font(), prefix, w, h)
    assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))


@pytest.mark.parametrize(
    "prefix, w, h",
    [
        (" ", 40, 16),  # |P| = 0: the space draws nothing
        ("c  c", 40, 16),  # |P| = 0: empty cells and spaces
        (
            "B",
            40,
            16,
        ),  # |Q| = 0: column 0 rows 0..7 dilate over all of R (x 1..3, y 1..9)
        ("b", 3, 16),  # R up to x = 3 needs W >= 4
        ("b", 40, 9),  # R up to y = 9 needs H >= 10
        ("A~d cb@1Q", 52, 16),  # w = 51: R needs W >= 53
    ],
)
def test_overlay_masks_value_error(prefix, w, h):
    with pytest.raises(ValueError):
        overlay_masks(hand_font(), prefix, w, h)


def test_overlay_masks_smallest_frame():
    """R fits exactly: W = 2 + w_tau, H = 10."""
    assert oo.width(hand_font(), "A~d cb@1Q") == 5 + 9 + 3 + 4 + 1 + 2 + 9 + 9 + 9 == 51
    p, q = overlay_masks(hand_font(), "A~d cb@1Q", 53, 10)
    assert p.shape == q.shape == (10, 53)


def test_luma_first_plane_row_major():
    """Notation: Y = the first W * H bytes as an H x W uint8 array; U, V ignored."""
    w, h = 6, 4
    y = bytes(range(10, 10 + w * h))
    got = luma(y + bytes(range(200, 200 + w * h // 2)), w, h)
    assert got.dtype == np.uint8 and got.shape == (h, w)
    assert all(got[r, c] == 10 + r * w + c for r in range(h) for c in range(w))


def test_luma_frame_size_of_the_caller():
    buf = np.random.default_rng(0).integers(0, 256, 640 * 360 * 3 // 2, dtype=np.uint8)
    got = luma(buf.tobytes(), 640, 360)
    assert got.shape == (360, 640)
    assert np.array_equal(got.ravel(), buf[: 640 * 360])


def write_config(root, repo, textures: dict) -> object:
    """An f3-overlay config whose catalogue's textures directory holds `textures`; paths
    relative to the repository root."""
    write_textures(root / "tex", textures)
    rel = os.path.relpath(root / "tex", repo)
    screens = os.path.relpath(write_screens(root / "screens.yaml", {}, rel), repo)
    data = yaml.safe_load((repo / "configs/f3-overlay.yaml").read_text())
    (root / "f3.yaml").write_text(yaml.safe_dump({**data, "screens": screens}))
    return load_config(root / "f3.yaml")


def test_overlay_font_reads_the_catalogue_textures(tmp_path, repo, monkeypatch):
    """Phi: the alpha of font/ascii.png under textures.out_dir of `screens`, whatever the
    working directory."""
    cfg = write_config(tmp_path, repo, {"font/ascii.png": font_image(hand_font())})
    monkeypatch.chdir(tmp_path / "tex")
    assert np.array_equal(overlay_font(cfg), hand_font())


def test_overlay_font_missing(tmp_path, repo):
    cfg = write_config(tmp_path, repo, {"font/other.png": font_image(hand_font())})
    with pytest.raises(FileNotFoundError):
        overlay_font(cfg)


@pytest.fixture(scope="module")
def real(repo):
    """The starting config, Phi by overlay_font and the alpha of the real ascii.png under
    textures.out_dir (specs/gui-screens.md: relative to the repository root)."""
    text = (repo / "configs/gui-screens.yaml").read_text()
    out_dir = yaml.load(text, Loader=yaml.CSafeLoader)["textures"]["out_dir"]
    path = repo / out_dir / "font/ascii.png"
    if not path.exists():
        pytest.skip(f"{path} is absent: scripts/download_data.sh fetches it")
    alpha = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    cfg = load_config(repo / "configs/f3-overlay.yaml")
    return cfg, overlay_font(cfg), alpha


def test_real_font_is_the_alpha(real):
    _, phi, alpha = real
    assert phi.shape == (128, 128) and np.array_equal(phi, alpha)


def test_real_font_counts(real):
    """Interface Tests: w_tau = 87, |P| = 169, |Q| = 217 for the starting prefix, 640 x 360;
    P, Q as the naive reference."""
    cfg, phi, _ = real
    assert oo.width(phi, cfg.prefix) == 87
    p, q = overlay_masks(phi, cfg.prefix, 640, 360)
    assert (int(p.sum()), int(q.sum())) == (169, 217)
    expected = oo.masks(phi, cfg.prefix, 640, 360)
    assert np.array_equal(p, expected[0]) and np.array_equal(q, expected[1])


def test_real_font_width_through_the_frame(real):
    """R = [1, 2 + 87) fits in W = 89 columns, not in 88."""
    cfg, phi, _ = real
    overlay_masks(phi, cfg.prefix, 89, 10)
    with pytest.raises(ValueError):
        overlay_masks(phi, cfg.prefix, 88, 10)


@pytest.mark.parametrize(
    "scene, text, fill, shown",
    [  # [R5]: text 208.38 / 73.7 dimmed, box 47.83 + 0.4353 Y_scene / [35.9, 59.4] dimmed
        (0, 206, 48, True),
        (235, 206, 150, True),
        (235, 73, 59, True),
        (0, None, None, False),
        (100, None, None, False),
        (208, None, None, False),  # text share 1, box share 0
        (255, None, None, False),
    ],
)
def test_real_font_frame_zero(real, scene, text, fill, shown):
    """The caller's use: overlay_shown on luma of a raw 640 x 360 frame with the overlay's
    line 0 drawn (fill over R, then the text) or not drawn."""
    cfg, phi, _ = real
    masks = overlay_masks(phi, cfg.prefix, 640, 360)
    y = np.full((360, 640), scene, dtype=np.uint8)
    if text is not None:
        y[1:10, 1:89] = fill
        y[masks[0]] = text
    buf = y.tobytes() + bytes([128]) * (640 * 360 // 2)
    assert overlay_shown(luma(buf, 640, 360), masks, cfg) is shown
