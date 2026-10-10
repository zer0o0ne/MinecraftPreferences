"""(1)-(3) of specs/gui-masks.md: template and mask pairs of the three variants."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from mcpref.gui.screens import Element, Member, Rect, ScreenClass

Pair = tuple[np.ndarray, np.ndarray]


def distance(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Notation d_T: max over channels (last axis) of |x - T|, in a signed integer type."""
    return np.abs(x.astype(np.int16) - t.astype(np.int16)).max(axis=-1)


def union(rects: Iterable[Rect], width: int, height: int) -> np.ndarray:
    """(1): pixels of an H x W frame in any rectangle [x, x+w) x [y, y+h)."""
    ys, xs = np.ogrid[:height, :width]
    mask = np.zeros((height, width), dtype=bool)
    for r in rects:
        mask |= (r.x <= xs) & (xs < r.x + r.w) & (r.y <= ys) & (ys < r.y + r.h)
    return mask


def slot_content(e: Element) -> Rect:
    """(1): slot-content rectangle (x_s, y_s, 17, 18); the hit is (x_s - 1, y_s - 1, 18, 18)."""
    return Rect(e.name, e.hit.x + 1, e.hit.y + 1, 17, 18)


def allowed(cls: ScreenClass, width: int, height: int) -> np.ndarray:
    """(1): inside window or book_panel, outside hits, slot contents, dynamic and labels."""
    regions = [cls.window] + ([cls.book_panel] if cls.book_panel else [])
    hits = [e.hit for e in cls.elements]
    contents = [slot_content(e) for e in cls.elements if e.kind == "slot"]
    drawn = [r for m in cls.members.values() for r in m.dynamic + m.labels]
    excluded = union(hits + contents + drawn, width, height)
    return union(regions, width, height) & ~excluded


def canvas(member: Member, textures_dir: Path, width: int, height: int) -> np.ndarray:
    """(1): the member's blits, RGBA, source-over in draw order on a transparent canvas."""
    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for b in member.blits:
        with Image.open(textures_dir / b.texture) as texture:
            part = texture.convert("RGBA").crop((b.u, b.v, b.u + b.w, b.v + b.h))
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer.paste(part, (b.x, b.y))
        out = Image.alpha_composite(out, layer)
    return np.asarray(out)


def member_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> dict[str, Pair]:
    """(1): member -> (T, M); T black where alpha < 255; members with an empty M skipped."""
    region = allowed(cls, width, height)
    pairs = {}
    for name, member in cls.members.items():
        rgba = canvas(member, textures_dir, width, height)
        opaque = rgba[..., 3] == 255
        mask = opaque & region
        if mask.any():
            pairs[name] = (np.where(opaque[..., None], rgba[..., :3], 0), mask)
    return pairs


def texture_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> list[Pair]:
    """(1): variant (a), one (T, M) pair per member with a non-empty mask."""
    return list(member_pairs(cls, textures_dir, width, height).values())


def block_pair(frames: Sequence[np.ndarray], rows: slice, q: float) -> Pair:
    """(2) on one row block: lower-median template and spread quantile."""
    x = np.stack([f[rows] for f in frames])
    template = np.quantile(x, 0.5, axis=0, method="inverted_cdf")
    spread = np.quantile(distance(x, template), q, axis=0, method="inverted_cdf")
    return template, spread


def variance_pair(
    frames: Sequence[np.ndarray], q: float, s_max: int, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(2): variant (b) template (uint8), spread sigma_k, mask sigma_k <= s_max."""
    height = frames[0].shape[0]
    blocks = [
        block_pair(frames, slice(r, r + chunk_rows), q)
        for r in range(0, height, chunk_rows)
    ]
    template = np.concatenate([t for t, _ in blocks])
    spread = np.concatenate([s for _, s in blocks])
    return template, spread, spread <= s_max


def patch_means(a: np.ndarray, patch: int) -> np.ndarray:
    """(3): float64 mean of `a` over each P x P patch, patches in grid layout."""
    h, w = a.shape
    blocks = a.reshape(h // patch, patch, w // patch, patch)
    return blocks.mean(axis=(1, 3), dtype=np.float64)


def ratio(
    k: str,
    templates: Mapping[str, np.ndarray],
    spread: np.ndarray,
    patch: int,
    eps: float,
) -> np.ndarray:
    """(3): rho_k(Q) = delta_k(Q) / (mean sigma_k(Q) + eps) per patch, float64."""
    deltas = [
        patch_means(distance(templates[k], t), patch)
        for j, t in templates.items()
        if j != k
    ]
    return np.min(deltas, axis=0) / (patch_means(spread, patch) + eps)


def top_patches(rho: np.ndarray, n_p: int, patch: int) -> np.ndarray:
    """(3): union of the n_P patches of largest rho; ties by top row, then left column."""
    chosen = np.zeros(rho.size, dtype=bool)
    chosen[np.argsort(-rho.ravel(), kind="stable")[:n_p]] = True
    grid = chosen.reshape(rho.shape)
    return grid.repeat(patch, axis=0).repeat(patch, axis=1)


def patch_masks(
    templates: Mapping[str, np.ndarray],
    spreads: Mapping[str, np.ndarray],
    patch: int,
    n_p: int,
    eps: float,
) -> dict[str, np.ndarray]:
    """(3): variant (c) mask of each given class (the >= 2 of Q with a (b) template)."""
    return {
        k: top_patches(ratio(k, templates, spreads[k], patch, eps), n_p, patch)
        for k in templates
    }


def bc_pairs(
    built: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    patch: int,
    n_p: int,
    eps: float,
) -> tuple[dict[str, Pair], dict[str, Pair]]:
    """(2)-(3): from (T, sigma_k, M) of each class of Q, the (b) pairs of the classes
    with a non-empty M, and their (c) pairs when there are >= 2 of them."""
    b = {k: (t, m) for k, (t, _, m) in built.items() if m.any()}
    if len(b) < 2:
        return b, {}
    templates = {k: t for k, (t, _) in b.items()}
    spreads = {k: built[k][1] for k in b}
    masks = patch_masks(templates, spreads, patch, n_p, eps)
    return b, {k: (templates[k], m) for k, m in masks.items()}
