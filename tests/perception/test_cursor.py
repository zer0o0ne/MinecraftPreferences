"""(1) of specs/perception.md: `draw_cursor` against a transcription of VPT's composite
(oracle.composite); `load_sprite` (Interface): the download from a fake HTTP server, the
sha256 check on every load, an existing file not downloaded, the crop and the alpha."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mcpref.perception.cursor import draw_cursor, load_sprite

from .oracle import composite
from .world import S_ALPHA, S_RGB, SPRITE, SPRITE_SHA, config, line, png, sprite_rgba

H, W = 30, 70  # f = 30 / 720: mouse 24 -> 1 pixel
SPRITE_PAIR = (S_RGB, S_ALPHA)


def frame(seed: int = 0, h: int = H, w: int = W) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def mouse(x: float, y: float, **fields) -> dict:
    return line(mouse={"x": x, "y": y}, **fields)


POSITIONS = {  # mouse (x, y) in VPT's 720-high coordinates -> the cursor's top-left pixel
    "inside": (240.0, 120.0),  # (10, 5)
    "top-left corner": (0.0, 0.0),
    "clipped at the right": (1500.0, 48.0),  # x 62: 8 columns
    "clipped at the bottom": (480.0, 600.0),  # y 25: 5 rows
    "clipped at both": (1656.0, 696.0),  # (69, 29): one pixel
    "truncated toward zero": (263.9, 143.99),  # 10.996, 5.9996 -> (10, 5)
    "right edge, nothing drawn": (1680.0, 48.0),  # x = W
    "bottom edge, nothing drawn": (48.0, 720.0),  # y = H
    "beyond both edges": (5000.0, 5000.0),
}


@pytest.mark.parametrize("xy", POSITIONS.values(), ids=POSITIONS)
def test_draw_cursor_as_vpt(xy):
    v = frame()
    before = v.copy()
    got = draw_cursor(v, mouse(*xy), SPRITE_PAIR, 720)
    assert got.dtype == np.uint8 and got.shape == v.shape
    np.testing.assert_array_equal(got, composite(before, mouse(*xy)))
    np.testing.assert_array_equal(v, before)  # drawn on a copy


def test_draw_cursor_changes_the_region_only():
    """Inside: exactly the k_c x k_c region at (10, 5) may change."""
    v = frame(1)
    got = draw_cursor(v, mouse(240.0, 120.0), SPRITE_PAIR, 720)
    outside = np.ones((H, W), bool)
    outside[5:21, 10:26] = False
    np.testing.assert_array_equal(got[outside], v[outside])
    assert (got[~outside] != v[~outside]).any()


def test_draw_cursor_unconditional():
    """Drawn whatever the line's GUI flag."""
    v = frame(2)
    got = draw_cursor(v, mouse(240.0, 120.0, isGuiOpen=False), SPRITE_PAIR, 720)
    np.testing.assert_array_equal(got, composite(v, mouse(240.0, 120.0)))


def test_draw_cursor_hand_pixel():
    """One pixel by hand: R = 100, sprite colour 200 at alpha 51/255 = 0.2:
    100 * 0.8 + 200 * 0.2 = 120 (float64, truncated); alpha 0 keeps the frame, 255 the
    sprite."""
    rgb = np.full((16, 16, 3), 200, np.uint8)
    alpha = np.zeros((16, 16, 1))
    alpha[0, 0], alpha[0, 1] = 51 / 255, 1.0
    v = np.full((H, W, 3), 100, np.uint8)
    got = draw_cursor(v, mouse(0.0, 0.0), (rgb, alpha), 720)
    assert got[0, 0].tolist() == [120, 120, 120]
    assert got[0, 1].tolist() == [200, 200, 200]
    assert got[0, 2].tolist() == [100, 100, 100]


def test_draw_cursor_mouse_height():
    """f = H / mouse_height: with mouse_height 360 the mouse (120, 60) -> (10, 5)."""
    v = frame(3)
    got = draw_cursor(v, mouse(120.0, 60.0), SPRITE_PAIR, 360)
    np.testing.assert_array_equal(got, composite(v, mouse(240.0, 120.0)))


def test_draw_cursor_on_a_640x360_frame():
    """VPT's frame size: f = 0.5."""
    v = frame(4, 360, 640)
    for xy in [(968.0, 555.0), (190.5, 166.3), (1270.0, 715.0)]:
        got = draw_cursor(v, mouse(*xy), SPRITE_PAIR, 720)
        np.testing.assert_array_equal(got, composite(v, mouse(*xy)))


def requests_to(http, body: bytes):
    """A fake HTTP server answering every GET with 200 and `body`."""
    return http(lambda req: (200, body))


def test_load_sprite_existing_file(tmp_path, http):
    """A file at `cursor` with the right sha256: read, nothing downloaded; S_rgb the top-left
    16 x 16 colour (uint8), a = alpha / 255 (float64, a trailing axis of 1)."""
    server = requests_to(http, b"other")
    cfg = config(tmp_path, cursor_url=f"http://127.0.0.1:{server.port}/c.png")
    rgb, alpha = load_sprite(cfg)
    assert server.requests == []
    assert rgb.dtype == np.uint8 and rgb.shape == (16, 16, 3)
    assert alpha.dtype == np.float64 and alpha.shape == (16, 16, 1)
    np.testing.assert_array_equal(rgb, S_RGB)
    np.testing.assert_array_equal(alpha, S_ALPHA)


def test_load_sprite_crop(tmp_path):
    """k_c = cursor_crop: 12 gives the top-left 12 x 12."""
    rgb, alpha = load_sprite(config(tmp_path, cursor_crop=12))
    np.testing.assert_array_equal(rgb, sprite_rgba()[:12, :12, :3])
    np.testing.assert_array_equal(alpha, sprite_rgba()[:12, :12, 3:] / 255.0)


def test_load_sprite_converts_to_rgba(tmp_path):
    """A grey-and-alpha PNG goes through `.convert("RGBA")`: R = G = B = its grey."""
    la = sprite_rgba()[:, :, [0, 3]]
    data = png(la, "LA")
    cfg = config(tmp_path, sprite=data, cursor_sha256=hashlib.sha256(data).hexdigest())
    rgb, alpha = load_sprite(cfg)
    np.testing.assert_array_equal(rgb, np.repeat(la[:16, :16, :1], 3, axis=-1))
    np.testing.assert_array_equal(alpha, la[:16, :16, 1:] / 255.0)


def test_load_sprite_downloads_a_missing_file(tmp_path, http):
    """No file at `cursor`: one GET of `cursor_url`, the body written at `cursor` (its
    missing directories created), no temporary file left; then read as an existing one."""
    server = requests_to(http, SPRITE)
    url = f"http://127.0.0.1:{server.port}/cursors/c.png"
    cursor = str(tmp_path / "new" / "dir" / "cursor.png")
    cfg = config(tmp_path, sprite=None, cursor=cursor, cursor_url=url)
    rgb, alpha = load_sprite(cfg)
    assert [(r.method, r.path) for r in server.requests] == [("GET", "/cursors/c.png")]
    assert Path(cursor).read_bytes() == SPRITE
    assert sorted(p.name for p in Path(cursor).parent.iterdir()) == ["cursor.png"]
    np.testing.assert_array_equal(rgb, S_RGB)
    np.testing.assert_array_equal(alpha, S_ALPHA)
    load_sprite(cfg)
    assert len(server.requests) == 1


def test_load_sprite_retries_the_download(tmp_path, http):
    """The download through `download` (retried after a 5xx)."""
    answers = iter([(503, b"busy"), (200, SPRITE)])
    server = http(lambda req: next(answers))
    url = f"http://127.0.0.1:{server.port}/c.png"
    cfg = config(tmp_path, sprite=None, cursor_url=url)
    rgb, _ = load_sprite(cfg)
    assert len(server.requests) == 2
    np.testing.assert_array_equal(rgb, S_RGB)


def test_load_sprite_downloaded_sha256_mismatch(tmp_path, http):
    """Downloaded bytes of another sha256: RuntimeError naming `cursor` as written and the
    expected sha256; nothing written."""
    server = requests_to(http, SPRITE + b"x")
    url = f"http://127.0.0.1:{server.port}/c.png"
    cfg = config(tmp_path, sprite=None, cursor_url=url)
    cursor = str(tmp_path / "sprite" / "cursor.png")
    with pytest.raises(RuntimeError) as info:
        load_sprite(cfg)
    assert str(info.value) == f"sha256 of {cursor} is not {SPRITE_SHA}"
    parent = Path(cursor).parent
    assert not parent.exists() or list(parent.iterdir()) == []


def test_load_sprite_existing_file_sha256_mismatch(tmp_path, http):
    """Checked on every load: a file at `cursor` with another sha256 raises, without a
    download."""
    server = requests_to(http, SPRITE)
    url = f"http://127.0.0.1:{server.port}/c.png"
    cfg = config(tmp_path, sprite=SPRITE + b"\0", cursor_url=url)
    cursor = str(tmp_path / "sprite" / "cursor.png")
    with pytest.raises(RuntimeError) as info:
        load_sprite(cfg)
    assert str(info.value) == f"sha256 of {cursor} is not {SPRITE_SHA}"
    assert server.requests == []
    assert Path(cursor).read_bytes() == SPRITE + b"\0"


def test_load_sprite_checks_after_a_change(tmp_path):
    """A file replaced after a successful load: the next load raises."""
    cfg = config(tmp_path)
    load_sprite(cfg)
    (tmp_path / "sprite" / "cursor.png").write_bytes(
        png(sprite_rgba()[::-1].copy(), "RGBA")
    )
    with pytest.raises(RuntimeError):
        load_sprite(cfg)
