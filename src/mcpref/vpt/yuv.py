"""(5) of specs/vpt-gui-frames.md: RGB view of a raw yuv420p frame."""

import numpy as np


def chroma(planes: np.ndarray, width: int, height: int, k: int) -> np.ndarray:
    """(5): chroma plane k (0: U, 1: V), repeated over its 2 x 2 block."""
    size = (width // 2) * (height // 2)
    start = width * height + k * size
    plane = planes[start : start + size].reshape(height // 2, width // 2)
    return plane.repeat(2, axis=0).repeat(2, axis=1)


def yuv420p_to_rgb(buf: bytes, width: int, height: int) -> np.ndarray:
    """(5): BT.601 limited range in float64, rounded half up, clipped; (height, width, 3) uint8."""
    planes = np.frombuffer(buf, np.uint8).astype(np.float64)
    y = 255 / 219 * (planes[: width * height].reshape(height, width) - 16)
    u = chroma(planes, width, height, 0) - 128
    v = chroma(planes, width, height, 1) - 128
    r = y + 255 / 224 * 1.402 * v
    g = y - 255 / 224 * (1.402 * 0.299 / 0.587) * v
    g = g - 255 / 224 * (1.772 * 0.114 / 0.587) * u
    b = y + 255 / 224 * 1.772 * u
    rgb = np.floor(np.stack([r, g, b], axis=-1) + 0.5)
    return np.clip(rgb, 0, 255).astype(np.uint8)
