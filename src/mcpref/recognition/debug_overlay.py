"""Interface of specs/f3-overlay.md: the debug overlay's first line on frame luma."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from mcpref import config_mapping, number, repo_path
from mcpref.gui.screens import Label, Rect
from mcpref.gui.textures import load_textures_config
from mcpref.recognition.drawn_masks import advance, glyph, load_font, text_pixels
from mcpref.recognition.masks import union

TEMPLATES = ("normal", "dimmed")


@dataclass(frozen=True)
class OverlayConfig:
    """Notation: configs/f3-overlay.yaml; lambda_m, delta_m, beta_m by template m,
    theta = threshold; `screens` resolved against the repository root."""

    screens: Path
    prefix: str
    text_luma: dict[str, float]
    tolerance: dict[str, float]
    box_max: dict[str, float]
    threshold: float


def templates(name: str, value: object) -> dict[str, float]:
    """Notation: a mapping with exactly the keys normal, dimmed, of numbers."""
    if not isinstance(value, dict) or set(value) != set(TEMPLATES):
        raise ValueError(f"{name} {value!r} has not exactly the keys {TEMPLATES}")
    return {m: number(f"{name}.{m}", value[m]) for m in TEMPLATES}


def prefix_text(value: object) -> str:
    """Notation `prefix`: a non-empty string of characters 0x20-0x7E."""
    if isinstance(value, str) and value and all(0x20 <= ord(c) <= 0x7E for c in value):
        return value
    raise ValueError(f"prefix {value!r} is not a non-empty string of 0x20-0x7E")


def load_config(path: Path) -> OverlayConfig:
    """Interface: the YAML file loaded into OverlayConfig; violated constraints raise."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, OverlayConfig)
    cfg = OverlayConfig(
        screens=repo_path("screens", raw["screens"]),
        prefix=prefix_text(raw["prefix"]),
        text_luma=templates("text_luma", raw["text_luma"]),
        tolerance=templates("tolerance", raw["tolerance"]),
        box_max=templates("box_max", raw["box_max"]),
        threshold=number("threshold", raw["threshold"]),
    )
    if not all(t >= 0 for t in cfg.tolerance.values()):
        raise ValueError(f"tolerance {cfg.tolerance} is not >= 0")
    if not 0 < cfg.threshold <= 1:
        raise ValueError(f"threshold {cfg.threshold} is not in (0, 1]")
    return cfg


def overlay_font(cfg: OverlayConfig) -> np.ndarray:
    """Interface: Phi, font/ascii.png under textures.out_dir of cfg.screens."""
    return load_font(load_textures_config(cfg.screens).out_dir)


def dilation(p: np.ndarray) -> np.ndarray:
    """Notation: bar P = {p + (d_x, d_y) : p in P, d_x, d_y in {-1, 0, 1}}, in the frame."""
    h, w = p.shape
    padded = np.pad(p, 1, mode="constant", constant_values=False)
    shifted = [padded[i : i + h, j : j + w] for i in range(3) for j in range(3)]
    return np.logical_or.reduce(shifted)


def overlay_masks(
    font: np.ndarray, prefix: str, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Notation: P of the label (tau, (2, 2, w_tau, 8), 0xE0E0E0) and Q = R minus bar P,
    R = [1, 2 + w_tau) x [1, 10); bool, H x W."""
    w_tau = sum(advance(glyph(font, c), c) for c in prefix)
    box = Rect("R", 1, 1, 1 + w_tau, 9)
    if box.x + box.w > width or box.y + box.h > height:
        raise ValueError(f"R {box} does not fit in a {width} x {height} frame")
    label = Label(Rect(prefix, 2, 2, w_tau, 8), 0xE0E0E0)
    p, _ = text_pixels(label, font, width, height)
    q = union([box], width, height) & ~dilation(p)
    if not p.any() or not q.any():
        raise ValueError(f"|P| = {p.sum()}, |Q| = {q.sum()} for {prefix!r}")
    return p, q


def luma(buf: bytes, width: int, height: int) -> np.ndarray:
    """Notation Y: the first W H bytes of a raw yuv420p frame, H x W uint8."""
    return np.frombuffer(buf, np.uint8, count=width * height).reshape(height, width)


def shares(
    y: np.ndarray, masks: tuple[np.ndarray, np.ndarray], cfg: OverlayConfig, m: str
) -> tuple[float, float]:
    """(1): a_m, g_m in float64 with Y as integers."""
    p, q = masks
    near = np.abs(y[p].astype(np.int64) - cfg.text_luma[m]) <= cfg.tolerance[m]
    dark = y[q].astype(np.int64) <= cfg.box_max[m]
    return near.mean(), dark.mean()


def overlay_score(
    y: np.ndarray, masks: tuple[np.ndarray, np.ndarray], cfg: OverlayConfig
) -> float:
    """(2): S = max over m of S_m = min(a_m, g_m)."""
    return float(max(min(shares(y, masks, cfg, m)) for m in TEMPLATES))


def overlay_shown(
    y: np.ndarray, masks: tuple[np.ndarray, np.ndarray], cfg: OverlayConfig
) -> bool:
    """(3): the overlay is shown iff S >= theta."""
    return overlay_score(y, masks, cfg) >= cfg.threshold
