"""`VARIANTS` and `rgb_rows` of src/mcpref/recognition/pairs.py against specs/gui-masks.md
(Interface): `rgb_rows` against the float64 transcription of the RGB view (5) of
specs/vpt-gui-frames.md in masks_oracle.py; the template shards it writes are checked through
`main` in test_evaluate_outputs.py."""

import numpy as np
import pytest
from masks_oracle import yuv_rgb

from mcpref.recognition.pairs import VARIANTS, rgb_rows

W, H = 6, 8


def test_variants_in_order():
    """The variant names a ... f, in this order (Notation)."""
    assert list(VARIANTS) == ["a", "b", "c", "d", "e", "f"]


@pytest.fixture
def frame(tmp_path) -> tuple:
    """A raw yuv420p frame (Y: H x W, then U, V: H/2 x W/2) with every plane spanning
    0..255 (limited range clipped at both ends); (path, Y, U, V)."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 256, (H, W), dtype=np.uint8)
    y[0, :3] = [0, 16, 255]
    u, v = (rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8) for _ in range(2))
    u[0, 0], v[0, 0], u[1, 1], v[1, 1] = 0, 255, 255, 0
    path = tmp_path / "7.0_x_1.yuv"
    path.write_bytes(y.tobytes() + u.tobytes() + v.tobytes())
    return path, y, u, v


BLOCKS = [(0, 2), (2, 4), (0, 4), (2, 6), (4, 8), (6, 8), (0, 8)]


@pytest.mark.parametrize("r0, r1", BLOCKS)
def test_rgb_rows_against_the_rgb_view(frame, r0, r1):
    """Rows [r0, r1) of (5), r0 and r1 even, the last block included: from the Y rows
    [r0, r1) and the U, V rows [r0/2, r1/2); uint8, (r1 - r0) x W x 3."""
    path, y, u, v = frame
    got = rgb_rows(path, W, H, r0, r1)
    want = yuv_rgb(y[r0:r1], u[r0 // 2 : r1 // 2], v[r0 // 2 : r1 // 2])
    assert got.dtype == np.uint8 and got.shape == (r1 - r0, W, 3)
    assert np.array_equal(got, want)


def test_rgb_rows_clip_both_ends(frame):
    """(5): Y 0 with U 0, V 255 and Y 255 with U 255, V 0 reach 0 and 255."""
    path, *_ = frame
    got = rgb_rows(path, W, H, 0, 2)
    assert {0, 255} <= set(np.unique(got).tolist())
