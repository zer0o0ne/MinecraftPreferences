"""(1) of specs/perception.md: VPT's cursor sprite, fetched once and verified, drawn
on a frame."""

import hashlib
import io
from collections.abc import Mapping

import numpy as np
from PIL import Image

from mcpref import REPO
from mcpref.http.retry import download
from mcpref.perception.config import PerceptionConfig
from mcpref.shards.store import make_dirs, write_file

Sprite = tuple[np.ndarray, np.ndarray]
"""(1): S_rgb (uint8, k_c x k_c x 3) and a (float64, k_c x k_c x 1)."""


def checked(cfg: PerceptionConfig, data: bytes) -> bytes:
    """Interface `load_sprite`: `data`, whose sha256 must be `cursor_sha256`."""
    if hashlib.sha256(data).hexdigest() != cfg.cursor_sha256:
        raise RuntimeError(f"sha256 of {cfg.cursor} is not {cfg.cursor_sha256}")
    return data


def sprite_bytes(cfg: PerceptionConfig) -> bytes:
    """Interface `load_sprite`: the file at `cursor`, downloaded and written when
    missing; checked on every call."""
    path = REPO / cfg.cursor
    if path.is_file():
        return checked(cfg, path.read_bytes())
    make_dirs(path.parent)
    data = download(cfg.cursor_url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)
    write_file(path, checked(cfg, data))
    return data


def load_sprite(cfg: PerceptionConfig) -> Sprite:
    """(1): S_rgb and a = alpha / 255 of the top-left k_c x k_c of the RGBA sprite."""
    rgba = np.asarray(Image.open(io.BytesIO(sprite_bytes(cfg))).convert("RGBA"))
    S = rgba[: cfg.cursor_crop, : cfg.cursor_crop]
    return S[:, :, :3], S[:, :, 3:].astype(np.float64) / 255


def draw_cursor(
    frame: np.ndarray, line: Mapping, sprite: Sprite, mouse_height: int
) -> np.ndarray:
    """(1) on a copy of `frame`: the sprite composited at the scaled mouse position,
    clipped at the right and bottom edges."""
    S_rgb, a = sprite
    out = frame.copy()
    H, W = out.shape[:2]
    f = H / mouse_height
    x, y = int(line["mouse"]["x"] * f), int(line["mouse"]["y"] * f)
    h_c = max(0, min(H - y, S_rgb.shape[0]))
    w_c = max(0, min(W - x, S_rgb.shape[1]))
    if h_c == 0 or w_c == 0:
        return out
    R = out[y : y + h_c, x : x + w_c]
    a_c = a[:h_c, :w_c]
    R[...] = (R * (1 - a_c) + S_rgb[:h_c, :w_c] * a_c).astype(np.uint8)
    return out
