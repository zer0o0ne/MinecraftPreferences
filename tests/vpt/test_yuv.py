"""(5) of specs/vpt-gui-frames.md: BT.601 limited-range yuv420p -> RGB, against exact
rational arithmetic."""

import math
from fractions import Fraction

import numpy as np
import pytest

from mcpref.vpt.gui_frames import yuv420p_to_rgb

K = Fraction(255, 224)
C_Y = Fraction(255, 219)
COEFFICIENTS = {  # channel -> coefficients of (Y - 16, U - 128, V - 128) in (5)
    "R": (C_Y, 0, K * Fraction("1.402")),
    "G": (C_Y, -K * Fraction("1.772") * Fraction("0.114") / Fraction("0.587"),
          -K * Fraction("1.402") * Fraction("0.299") / Fraction("0.587")),
    "B": (C_Y, K * Fraction("1.772"), 0),
}  # fmt: skip


def numerators(y, u, v) -> list[tuple]:
    """Per channel: (n, d) with n / d the exact value of (5) before rounding."""
    out = []
    for coefs in COEFFICIENTS.values():
        d = math.lcm(*(Fraction(c).denominator for c in coefs))
        a_y, a_u, a_v = (int(Fraction(c) * d) for c in coefs)
        out.append((a_y * (y - 16) + a_u * (u - 128) + a_v * (v - 128), d))
    return out


def exact_rgb(y, u, v) -> np.ndarray:
    """min(255, max(0, floor(x + 1/2))) in exact arithmetic; int64 inputs."""
    return np.stack(
        [np.clip((2 * n + d) // (2 * d), 0, 255) for n, d in numerators(y, u, v)], -1
    ).astype(np.uint8)


def frame(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> bytes:
    """Planes Y (h x w), then U, V (h/2 x w/2)."""
    return b"".join(np.asarray(p, np.uint8).tobytes() for p in (y, u, v))


def upsampled(c: np.ndarray, h: int, w: int) -> np.ndarray:
    """Chroma of pixel (r, c) = sample (r // 2, c // 2)."""
    return c[np.arange(h)[:, None] // 2, np.arange(w)[None, :] // 2]


@pytest.mark.parametrize(
    "yuv, rgb",
    [
        ((16, 128, 128), (0, 0, 0)),
        ((235, 128, 128), (255, 255, 255)),
        ((126, 128, 128), (128, 128, 128)),  # 255 * 110 / 219 = 128.08
        ((255, 128, 128), (255, 255, 255)),  # 278.29, clipped
        ((0, 128, 128), (0, 0, 0)),  # -18.63, clipped
        ((81, 90, 240), (254, 0, 0)),  # 254.44, -0.48, -0.97
        ((145, 54, 34), (0, 255, 1)),  # 0.18, 255.61 -> 256 clipped, 0.93 -> 1
        ((41, 240, 110), (0, 0, 255)),  # 0.38, -0.13, 255.04
        ((100, 200, 60), (0, 125, 243)),  # -10.72, 124.88 -> 125, 243.05
        ((16, 16, 16), (0, 135, 0)),  # -178.76, 134.93, -225.93
        ((235, 240, 240), (255, 120, 255)),  # 433.76, 120.07, 480.93
    ],
)
def test_hand_picked_pixels_eq5(yuv, rgb):
    y, u, v = yuv
    out = yuv420p_to_rgb(frame(np.full((2, 2), y), [[u]], [[v]]), 2, 2)
    assert out.dtype == np.uint8 and out.shape == (2, 2, 3)
    assert (out == np.array(rgb, np.uint8)).all()


def test_plane_layout_and_chroma_blocks():
    """6 x 4 frame: Y per pixel, U and V per 2 x 2 block (2 x 3 samples), U before V."""
    rng = np.random.default_rng(0)
    h, w = 4, 6
    y, u, v = (
        rng.integers(0, 256, s) for s in ((h, w), (h // 2, w // 2), (h // 2, w // 2))
    )
    out = yuv420p_to_rgb(frame(y, u, v), w, h)
    assert out.shape == (h, w, 3) and out.dtype == np.uint8
    assert (out == exact_rgb(y, upsampled(u, h, w), upsampled(v, h, w))).all()


def test_no_exact_half_for_any_yuv():
    """Every (Y, U, V): no channel of (5) is exactly k + 1/2, so the rounding is unique."""
    u, v = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    for y in range(256):
        for n, d in numerators(np.int64(y), u, v):
            assert not ((2 * n + d) % (2 * d) == 0).any(), y


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_every_yuv_triple_matches_exact_rounding():
    """64 frames of 512 x 512: block (U, V) = (row, col), its 4 pixels Y = 4k .. 4k+3."""
    u, v = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    big_u, big_v = upsampled(u, 512, 512), upsampled(v, 512, 512)
    offsets = np.tile([[0, 1], [2, 3]], (256, 256))
    for k in range(64):
        y = 4 * k + offsets
        out = yuv420p_to_rgb(frame(y, u, v), 512, 512)
        assert (out == exact_rgb(y, big_u, big_v)).all(), k
