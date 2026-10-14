"""A hand-made font and helpers for the tests of specs/gui-masks.md (9)-(12): classes in the
YAML forms of specs/gui-screens.md, written with that font and loaded."""

import numpy as np
from masks_files import write_screens, write_textures
from masks_world import font_image

from mcpref.gui.screens import Label, Rect, load_screens

W, H = 40, 24
GLYPHS = {  # code point -> texels (gx, gy) of its cell; "c" (0x63) has an empty cell
    ord("A"): [(0, 1), (1, 1), (2, 1), (3, 3)],  # widest row 1 ends at 2; width 4
    ord("b"): [(0, 2), (0, 3), (0, 4), (0, 5)],  # width 1
    ord("d"): [(0, 1), (1, 2)],  # (1, 2) also lies in G_1 of (0, 1); width 2
    ord("~"): [(0, 0), (7, 7)],  # the cell's corners; width 8
    ord("@"): [(7, j) for j in range(8)],  # A's neighbours, never drawn
    ord("B"): [(0, j) for j in range(8)],
    ord("1"): [(i, 7) for i in range(8)],
    ord("Q"): [(i, 0) for i in range(8)],
}


def hand_font() -> np.ndarray:
    """Phi of GLYPHS: alpha 255 on their texels."""
    phi = np.zeros((128, 128), dtype=np.uint8)
    for c, texels in GLYPHS.items():
        for gx, gy in texels:
            phi[8 * (c // 16) + gy, 8 * (c % 16) + gx] = 255
    return phi


def pixels(mask: np.ndarray) -> set:
    """The (x, y) of a boolean H x W image."""
    return {(int(x), int(y)) for y, x in np.argwhere(mask)}


def label(text: str, rect: tuple, color: int = 0x404040) -> Label:
    return Label(Rect(text, *rect), color)


def write_class(root, cls: dict, images: dict):
    """Writes textures (with the hand font) and a catalogue of the class `k`; returns it loaded."""
    write_textures(root / "tex", {**images, "font/ascii.png": font_image(hand_font())})
    screens = load_screens(write_screens(root / "screens.yaml", {"k": cls}, "tex"))
    return screens["k"]


def texture(w: int, seed: int, hole: tuple) -> np.ndarray:
    """Opaque RGBA w x H of random colours, alpha 0 at the (x, y) `hole`."""
    image = np.random.default_rng(seed).integers(0, 256, (H, w, 4))
    image[..., 3] = 255
    image[hole[1], hole[0], 3] = 0
    return image
