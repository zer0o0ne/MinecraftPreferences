"""Synthetic inputs of `main` of specs/gui-masks.md: catalogue, textures with the font, frames,
labels, config; written by masks_files.py."""

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from drawn_oracle import text_pixels

W, H = 24, 20
CLASSES = ["alpha", "beta", "gamma", "delta"]
LABELS = [*CLASSES, "other"]
CRITERIA = ("agreement", "macro_recall")
VARIANTS = "abcdef"
N_EP = 18
EPISODES = [f"data/7.0/ep{7 * e % N_EP:02d}.mp4" for e in range(N_EP)]  # not sorted
CONFIG = {"split": ["0.5", "0.25", "0.25"], "seed": 230, "min_episodes": 2}
CONFIG.update(spread_quantile=0.75, spread_max=20, patch=4, patches_per_class=3, eps=1)
CONFIG.update(tau_grid=[12, 4, 40], theta_grid=["0.9", "0.5", "0.75"])
CONFIG.update(mu_grid=["0", "0.25", "0.1"], chunk_rows=3)
CONFIG.update(folds=3, fold_seed=3, feature_quantile=0.5)
GLYPHS = {  # code point -> top row, rows of the glyph from column 0
    ord("A"): (1, [".#.", "#.#", "###", "#.#"]),  # width 3
    ord("b"): (2, ["#.", "##", "##"]),  # width 2
    ord("."): (4, ["#"]),  # width 1
}


def blit(texture: str, u: int, v: int, w: int, h: int, x: int, y: int) -> dict:
    return {"texture": texture, "u": u, "v": v, "w": w, "h": h, "x": x, "y": y}


def member(blits: list, dynamic=(), labels=()) -> dict:
    """labels: (text, rectangle, colour)."""
    named = [{"name": f"d{i}", "rect": r} for i, r in enumerate(dynamic)]
    texts = [{"name": s, "rect": r, "color": c} for s, r, c in labels]
    return {"blits": blits, "dynamic": named, "labels": texts}


def whole(texture: str, w: int, x: int) -> list:
    """One blit of the whole texture (w x H) at (x, 0)."""
    return [blit(texture, 0, 0, w, H, x, 0)]


def screen(members: dict, window: list, go: tuple, slots=()) -> dict:
    """Slots at the hit rectangles (x, y, 18, 18) of `slots`, then the 2 x 2 button `go`."""
    hits = [[x, y, 18, 18] for x, y in slots] + [[*go, 2, 2]]
    elements = [
        {"index": i, "name": f"e{i}", "kind": "slot" if i <= len(slots) else "button"}
        | {"hit": r, "click": [r[0] + r[2] / 2, r[1] + r[3] / 2]}
        for i, r in zip([*range(1, len(slots) + 1), 37], hits)
    ]
    cls = {"members": members, "description": "-", "window": window}
    return {**cls, "elements": elements}


ONE = member(  # a label over a hole and over its own second dynamic rectangle
    whole("gui/alpha1.png", 12, 0),
    dynamic=[[0, 0, 2, 1], [5, 6, 3, 3]],
    labels=[("Ab", [1, 2, 7, 8], 0x404040)],
)
TWO = member(whole("gui/alpha2.png", 12, 0))  # no label
BETA = member(whole("gui/beta.png", 12, 12), labels=[(" b", [13, 11, 7, 9], 0xE0E0E0)])
GAMMA = member(whole("gui/clear.png", W, 0), labels=[("A.b", [2, 4, 9, 9], 0xFFFFFF)])
DELTA = member(  # the label partly over the transparent corner, a dynamic over the slot
    whole("gui/delta.png", W, 0),
    dynamic=[[20, 12, 3, 3]],
    labels=[("bA", [15, 0, 7, 9], 0x404040)],
)
CATALOGUE = {
    "alpha": screen({"one": ONE, "two": TWO}, [0, 0, 12, H], (6, 14)),
    "beta": screen({"only": BETA}, [12, 0, 12, H], (22, 0)),
    "gamma": screen({"only": GAMMA}, [0, 0, W, H], (0, 18), slots=[(4, 1)]),
    "delta": screen({"only": DELTA}, [0, 0, W, H], (22, 18), slots=[(3, 1)]),
}
_rng = np.random.default_rng(2026)
PATTERNS = {k: _rng.integers(16, 236, (H, 12 if k == "alpha" else W)) for k in CLASSES}
FAR_FROM_125 = [
    *range(16, 80),
    *range(172, 236),
]  # beta's inverse 251 - Y differs a lot
PATTERNS["beta"] = _rng.choice(FAR_FROM_125, (H, 12))
MISLABEL = {"alpha": "beta", "beta": "delta", "gamma": "other", "delta": "other"}
MISLABEL["other"] = "alpha"


def grey(y: np.ndarray) -> np.ndarray:
    """vpt-gui-frames (5) with U = V = 128: R = G = B = floor(255/219 (Y - 16) + 0.5), clipped."""
    v = np.floor(255 / 219 * (y.astype(np.float64) - 16) + 0.5)
    return np.clip(v, 0, 255).astype(np.int64)


def rgba(y: np.ndarray, alpha: np.ndarray, shift=(0, 0, 0)) -> np.ndarray:
    rgb = np.clip(grey(y)[..., None] + np.array(shift), 0, 255)
    return np.concatenate([rgb, alpha[..., None]], axis=2)


def font() -> np.ndarray:
    """Phi: the alpha of font/ascii.png, 255 on the texels of GLYPHS, 0 elsewhere."""
    phi = np.zeros((128, 128), dtype=np.int64)
    for c, (top, rows) in GLYPHS.items():
        for j, i in itertools.product(range(len(rows)), range(len(rows[0]))):
            if rows[j][i] == "#":
                phi[8 * (c // 16) + top + j, 8 * (c % 16) + i] = 255
    return phi


def font_image(phi: np.ndarray) -> np.ndarray:
    """font/ascii.png of the alpha phi: opaque texels white, the others transparent black."""
    return np.stack([np.where(phi > 0, 255, 0)] * 3 + [phi], axis=2)


def textures() -> dict:
    """The classes' own colours: alpha1 with two holes (one under its label), alpha2 brighter
    on the left, beta less red, delta transparent where its frames vary, clear fully
    transparent; the font white where opaque."""
    opaque = np.full((H, 12), 255)
    holes = opaque.copy()
    holes[19, 0] = holes[3, 3] = 0
    brighter = PATTERNS["alpha"] + np.where(np.arange(12) < 6, 26, 0)
    corner = np.full((H, W), 255)
    corner[:6, 16:] = 0
    phi = font()
    return {
        "gui/alpha1.png": rgba(PATTERNS["alpha"], holes),
        "gui/alpha2.png": rgba(brighter, opaque),
        "gui/beta.png": rgba(PATTERNS["beta"], opaque, (-3, 0, 0)),
        "gui/delta.png": rgba(PATTERNS["delta"], corner),
        "gui/clear.png": np.zeros((H, W, 4), dtype=np.int64),
        "font/ascii.png": font_image(phi),
    }


@dataclass
class Frame:
    id: str
    relpath: str
    status: str
    flag: bool  # line.isGuiInventory
    label: str | None  # VLM label; None: no record in labels.jsonl
    y: np.ndarray | None  # Y plane; None for a failed frame


def gamma_text(y: np.ndarray) -> None:
    """gamma's label drawn by (9): Y 70 (RGB 63 = floor(255 / 4)) on G_1, then 235 on G_0."""
    (label,) = GAMMA["labels"]
    g0, g1 = text_pixels(label["name"], label["rect"], font(), W, H)
    y[g1], y[g0] = 70, 235


def y_plane(cls: str, partial: bool, rng) -> np.ndarray:
    """alpha: its half (partial: a corner covered); beta: its half, the pattern or its inverse
    251 - Y (high spread); gamma: its pattern with its label; delta: the whole frame, random in
    its transparent corner; other: random (partial: alpha's top half). Noise +-5 in Y."""
    y = rng.integers(16, 236, (H, W))
    noise = rng.integers(-5, 6, (H, W))
    if cls == "alpha":
        cover = y[:6, :6].copy()
        y[:, :12] = PATTERNS["alpha"] + noise[:, :12]
        if partial:
            y[:6, :6] = cover
    elif cls == "other" and partial:
        y[:10, :12] = PATTERNS["alpha"][:10] + noise[:10, :12]
    elif cls == "beta":
        y[:, 12:] = PATTERNS["beta"] if rng.integers(2) else 251 - PATTERNS["beta"]
        y[:, 12:] += noise[:, 12:]
    elif cls in ("gamma", "delta"):
        corner = y[:6, 16:].copy()
        y = PATTERNS[cls] + noise
        if cls == "delta":
            y[:6, 16:] = corner
        else:
            gamma_text(y)
    return np.clip(y, 16, 235)


def frames(relabel: bool = False) -> list:
    """Episodes interleaved, 3 frames each; episodes 5 (test) and 10 (train) show gamma;
    mislabels, invalid, no_answer, a missing label; failed records first (of a test episode,
    of a ghost). relabel: the first episode unlabelled, episode 4 invalid."""
    rng, out = np.random.default_rng(7), []
    special = {(2, 1): "invalid", (9, 0): "no_answer", (12, 2): None}
    for s, e in itertools.product(range(3), range(N_EP)):
        cls = (
            "gamma"
            if e in (5, 10)
            else ["alpha", "beta", "delta", "other"][(e + s) % 4]
        )
        n = 3 * e + s
        label = special.get((e, s), MISLABEL[cls] if n % 10 == 7 else cls)
        if relabel and e in (0, 4):
            label = None if e == 0 else "invalid"
        stem, flag = Path(EPISODES[e]).stem, (cls == "alpha") != (n % 5 == 0)
        y = y_plane(cls, n % 3 == 2, rng)
        out.append(Frame(f"7.0_{stem}_{s + 10}", EPISODES[e], "ok", flag, label, y))
    out.insert(0, Frame("7.0_first_1", EPISODES[2], "failed", False, None, None))
    out.insert(
        3, Frame("7.0_ghost_1", "data/7.0/ghost.mp4", "failed", True, None, None)
    )
    return out
