"""Cursor clipping against an independent per-pixel compositing oracle."""

import numpy as np
import pytest

from mcpref.models.perception.cursor import draw_cursor


@pytest.mark.parametrize(
    "height,width,mouse_height,x,y",
    [
        (360, 640, 720, -5.0, 354.0),  # Real reference prefix 61, observation 28.
        (30, 70, 30, -1, 5),
        (30, 70, 30, 5, -1),
        (30, 70, 30, -1, -1),
        (30, 70, 30, -15, -15),  # One visible sprite pixel.
        (30, 70, 30, -16, 5),
        (30, 70, 30, 5, -16),
        (30, 70, 30, -17, -17),
        (30, 70, 30, -5000, 5),
        (30, 70, 30, 5, -5000),
        (30, 70, 60, -1.9, -1.9),  # Truncation toward zero, not floor.
        (30, 70, 60, -3.9, -3.9),
        (30, 70, 30, -3, 29),
        (30, 70, 30, 69, -3),
        (30, 70, 30, 70, -3),
        (30, 70, 30, -3, 30),
        (2, 4, 2, -3, -5),  # Both ends of each sprite axis are clipped.
    ],
)
def test_cursor_clips_every_edge(height, width, mouse_height, x, y):
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (16, 16, 1)) / 255.0
    before, expected = frame.copy(), frame.copy()
    left, top = int(x * height / mouse_height), int(y * height / mouse_height)
    for sy, sx in np.ndindex(16, 16):
        row, col = top + sy, left + sx
        if 0 <= row < height and 0 <= col < width:
            a = alpha[sy, sx]
            expected[row, col] = (before[row, col] * (1 - a) + rgb[sy, sx] * a).astype(
                np.uint8
            )
    got = draw_cursor(frame, {"mouse": {"x": x, "y": y}}, (rgb, alpha), mouse_height)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(frame, before)
    assert got.dtype == np.uint8 and not np.shares_memory(got, frame)
