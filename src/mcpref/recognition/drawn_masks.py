"""(9)-(12) of specs/gui-masks.md: pairs of variants (d), (e), (f), from the catalogue and
textures only."""

from pathlib import Path

import numpy as np
from PIL import Image

from mcpref.gui.screens import Label, Rect, ScreenClass
from mcpref.gui.textures import FONT
from mcpref.recognition.masks import (
    Pair,
    drawn_rects,
    member_pairs,
    opaque_canvas,
    opaque_pairs,
    union,
)

LUMA = (299, 587, 114)
SCALE = 1000


def load_font(textures_dir: Path) -> np.ndarray:
    """Notation Phi: the alpha channel of font/ascii.png under the textures directory."""
    with Image.open(textures_dir / FONT) as font:
        return np.asarray(font.convert("RGBA"))[..., 3]


def glyph(font: np.ndarray, ch: str) -> np.ndarray:
    """Notation: Phi != 0 in the cell of code point c, rows 8 floor(c/16) + [0, 8),
    columns 8 (c mod 16) + [0, 8)."""
    row, col = divmod(ord(ch), 16)
    return font[8 * row : 8 * row + 8, 8 * col : 8 * col + 8] != 0


def advance(cell: np.ndarray, ch: str) -> int:
    """(9): adv(c) = 4 for the space, else w(c) + 1; w(c) = 1 + the last column with
    Phi != 0, 0 for an empty cell."""
    if ch == " ":
        return 4
    columns = np.flatnonzero(cell.any(axis=0))
    return (int(columns[-1]) + 1 if columns.size else 0) + 1


def glyph_pixels(cell: np.ndarray, x: int, y: int, omega: int) -> tuple:
    """(9): rows y + omega g_y + b and columns x + omega g_x + a of the cell's pixels
    Phi != 0, a, b in [0, omega)."""
    gy, gx = np.nonzero(cell)
    a = np.arange(omega)
    rows = y + omega * gy[:, None, None] + a[None, :, None]
    cols = x + omega * gx[:, None, None] + a[None, None, :]
    return tuple(np.broadcast_arrays(rows, cols))


def text_pixels(
    label: Label, font: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """(9): G_0, G_1 (bool, H x W) of the label's text at its rectangle; G_1 empty
    without a shadow."""
    r = label.rect
    omega = 2 if r.h == 18 else 1
    offsets = (0, 1) if r.h in (9, 18) else (0,)
    g = np.zeros((2, height, width), dtype=bool)
    u = 0
    for ch in r.name:
        cell = glyph(font, ch)
        if ch != " ":
            for o in offsets:
                g[o][
                    glyph_pixels(cell, r.x + omega * (u + o), r.y + omega * o, omega)
                ] = True
        u += advance(cell, ch)
    return g[0], g[1]


def channels(color: int) -> np.ndarray:
    """(9): chi = (chi_R, chi_G, chi_B) of a colour 0xRRGGBB."""
    return np.array([color >> 16 & 255, color >> 8 & 255, color & 255], dtype=np.uint8)


def draw(rgb: np.ndarray, label: Label, font: np.ndarray) -> np.ndarray:
    """(9): G_1 takes floor(chi / 4), then G_0 takes chi, in place; returns the text
    pixels G_0 | G_1."""
    g0, g1 = text_pixels(label, font, rgb.shape[1], rgb.shape[0])
    chi = channels(label.color)
    rgb[g1] = chi // 4
    rgb[g0] = chi
    return g0 | g1


def label_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> dict[str, Pair]:
    """(10): member -> (bar Gamma_m with its labels drawn, M); members without labels
    skipped."""
    font = load_font(textures_dir)
    pairs = {}
    for name, member in cls.members.items():
        if not member.labels:
            continue
        t, opaque = opaque_canvas(member, textures_dir, width, height)
        kept = opaque & ~union(member.dynamic, width, height)
        mask = np.zeros((height, width), dtype=bool)
        for lab in member.labels:
            text = draw(t, lab, font)
            mask |= (union([lab.rect], width, height) & kept) | text
        pairs[name] = (t, mask)
    return pairs


def ring(cls: ScreenClass, width: int, height: int) -> np.ndarray:
    """(11): R_k, the union of the 1-px frames of the slot hit rectangles (x, y, 18, 18)
    minus (x + 1, y + 1, 16, 16)."""
    r = np.zeros((height, width), dtype=bool)
    for e in cls.elements:
        if e.kind == "slot":
            inner = Rect(e.name, e.hit.x + 1, e.hit.y + 1, 16, 16)
            r |= union([e.hit], width, height) & ~union([inner], width, height)
    return r


def ring_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> dict[str, Pair]:
    """(11): member -> (bar Gamma_m, M); members with an empty M skipped."""
    kept = ring(cls, width, height) & ~union(drawn_rects(cls), width, height)
    return opaque_pairs(cls, kept, textures_dir, width, height)


def edges(rgb: np.ndarray) -> np.ndarray:
    """(12): E(x) (int64, H x W) from L(x) = 299 R + 587 G + 114 B; 0 in the last row
    and column."""
    lum = rgb.astype(np.int64) @ np.array(LUMA, dtype=np.int64)
    e = np.zeros(lum.shape, dtype=np.int64)
    right = np.abs(lum[:-1, 1:] - lum[:-1, :-1])
    down = np.abs(lum[1:, :-1] - lum[:-1, :-1])
    e[:-1, :-1] = np.maximum(right, down)
    return e


def edge_mask(u: np.ndarray) -> np.ndarray:
    """(12): {p : p, p + (1, 0), p + (0, 1) in u}."""
    m = np.zeros(u.shape, dtype=bool)
    m[:-1, :-1] = u[:-1, :-1] & u[:-1, 1:] & u[1:, :-1]
    return m


def edge_pairs_from(
    cls: ScreenClass, a: dict[str, Pair], e: dict[str, Pair]
) -> dict[str, Pair]:
    """(12): member -> (E(bar Gamma_m), M) from the pairs (bar Gamma_m, M^a_m), (bar Gamma_m,
    M^e_m) of (1), (11) (M^V_m empty where m was skipped); members with an empty M skipped."""
    pairs = {}
    for name in cls.members:
        built = [p[name] for p in (a, e) if name in p]
        if built:
            mask = edge_mask(np.logical_or.reduce([m for _, m in built]))
            if mask.any():
                pairs[name] = (edges(built[0][0]), mask)
    return pairs


def edge_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> dict[str, Pair]:
    """(12): member -> (E(bar Gamma_m), M), M from the masks M^a_m, M^e_m of (1), (11);
    members with an empty M skipped."""
    a = member_pairs(cls, textures_dir, width, height)
    return edge_pairs_from(cls, a, ring_pairs(cls, textures_dir, width, height))
