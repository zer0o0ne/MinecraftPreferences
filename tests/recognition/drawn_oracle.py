"""Naive references for specs/gui-masks.md (9)-(12): explicit loops over glyph cells and
pixels, Python integers; catalogue classes as plain YAML dicts; nothing from src/."""

import numpy as np
from masks_oracle import composite, inside, outline_pairs


def glyph_width(font: np.ndarray, c: int) -> int:
    """(9): 1 + the last column of c's cell with a non-zero alpha; 0 for an empty cell."""
    cols = [
        i
        for i in range(8)
        if any(font[8 * (c // 16) + j][8 * (c % 16) + i] != 0 for j in range(8))
    ]
    return 1 + max(cols) if cols else 0


def advance(font: np.ndarray, c: int) -> int:
    """(9): 4 for the space, w(c) + 1 otherwise."""
    return 4 if c == 0x20 else glyph_width(font, c) + 1


def text_pixels(text: str, rect, font: np.ndarray, width: int, height: int) -> tuple:
    """(9): G_0, G_1 of a label (text, rectangle (x0, y0, w0, h0)); G_1 empty without shadow."""
    x0, y0, _, h0 = rect
    scale, offsets = (2 if h0 == 18 else 1), ((0, 1) if h0 in (9, 18) else (0,))
    g = [np.zeros((height, width), dtype=bool) for _ in range(2)]
    u = 0
    for ch in text:
        c = ord(ch)
        for gy in range(8):
            for gx in range(8):
                if c == 0x20 or font[8 * (c // 16) + gy][8 * (c % 16) + gx] == 0:
                    continue
                for o in offsets:
                    for a in range(scale):
                        for b in range(scale):
                            px, py = (
                                x0 + scale * (u + gx + o) + a,
                                y0 + scale * (gy + o) + b,
                            )
                            g[o][py, px] = True
        u += advance(font, c)
    return g[0], g[1]


def rgb(canvas: np.ndarray) -> np.ndarray:
    """Gamma-bar of Notation: the canvas RGB, pixels of alpha < 255 black."""
    return np.where(canvas[..., 3:] == 255, canvas[..., :3], 0)


def colour(label: dict) -> np.ndarray:
    return np.array([label["color"] >> s & 255 for s in (16, 8, 0)])


def label_pairs(cls: dict, textures: dict, font, width: int, height: int) -> dict:
    """(10): member -> (template with the labels drawn by (9), mask); members without labels
    left out."""
    out = {}
    for name, member in cls["members"].items():
        if not member["labels"]:
            continue
        canvas = composite(member["blits"], textures, width, height)
        t, mask = rgb(canvas), np.zeros((height, width), dtype=bool)
        own = [r["rect"] for r in member["dynamic"]]
        for label in member["labels"]:
            g0, g1 = text_pixels(label["name"], label["rect"], font, width, height)
            t[g1] = colour(label) // 4
            t[g0] = colour(label)
            for y in range(height):
                for x in range(width):
                    shown = inside(label["rect"], x, y) and canvas[y, x, 3] == 255
                    free = shown and not any(inside(r, x, y) for r in own)
                    mask[y, x] |= free or g0[y, x] or g1[y, x]
        out[name] = (t, mask)
    return out


def ring(cls: dict, width: int, height: int) -> np.ndarray:
    """(11): R_k, the 1-px frames of the slot hit rectangles."""
    r = np.zeros((height, width), dtype=bool)
    for e in cls["elements"]:
        if e["kind"] == "slot":
            x, y = e["hit"][:2]
            for j in range(18):
                for i in range(18):
                    r[y + j, x + i] |= not (1 <= i <= 16 and 1 <= j <= 16)
    return r


def ring_pairs(cls: dict, textures: dict, width: int, height: int) -> dict:
    """(11): member -> (Gamma-bar, mask), members with an empty mask left out."""
    holes = [
        r["rect"] for m in cls["members"].values() for r in m["dynamic"] + m["labels"]
    ]
    frame, out = ring(cls, width, height), {}
    for name, member in cls["members"].items():
        canvas = composite(member["blits"], textures, width, height)
        mask = np.zeros((height, width), dtype=bool)
        for y in range(height):
            for x in range(width):
                shown = frame[y, x] and canvas[y, x, 3] == 255
                mask[y, x] = shown and not any(inside(r, x, y) for r in holes)
        if mask.any():
            out[name] = (rgb(canvas), mask)
    return out


def edges(x: np.ndarray) -> np.ndarray:
    """(12): E of an RGB image, Python integers; 0 in the last row and column."""
    h, w = x.shape[:2]

    def luma(i: int, j: int) -> int:
        return 299 * int(x[j, i, 0]) + 587 * int(x[j, i, 1]) + 114 * int(x[j, i, 2])

    e = np.zeros((h, w), dtype=np.int64)
    for j in range(h - 1):
        for i in range(w - 1):
            e[j, i] = max(
                abs(luma(i + 1, j) - luma(i, j)), abs(luma(i, j + 1) - luma(i, j))
            )
    return e


def edge_pairs(cls: dict, textures: dict, width: int, height: int) -> dict:
    """(12): member -> (E(Gamma-bar), mask), members with an empty mask left out."""
    outline = outline_pairs(cls, textures, width, height)
    rings = ring_pairs(cls, textures, width, height)
    out = {}
    for name, member in cls["members"].items():
        u = np.zeros((height, width), dtype=bool)
        for pairs in (outline, rings):
            if name in pairs:
                u |= pairs[name][1]
        mask = np.zeros((height, width), dtype=bool)
        for y in range(height - 1):
            for x in range(width - 1):
                mask[y, x] = u[y, x] and u[y, x + 1] and u[y + 1, x]
        if mask.any():
            canvas = composite(member["blits"], textures, width, height)
            out[name] = (edges(rgb(canvas)), mask)
    return out
