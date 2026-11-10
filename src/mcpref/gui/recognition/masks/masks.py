"""(1)-(3) of specs/gui-masks.md: template and mask pairs of variants (a), (b), (c)."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from mcpref.gui.screens import Element, Member, Rect, ScreenClass

Pair = tuple[np.ndarray, np.ndarray]


def distance(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Notation d_T: max over channels (last axis) of |x - T|, in a signed integer type
    (int16 for uint8 images, int64 for the edges of (12))."""
    signed = np.promote_types(np.promote_types(x.dtype, t.dtype), np.int16)
    return np.abs(x.astype(signed) - t.astype(signed)).max(axis=-1)


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


def drawn_rects(cls: ScreenClass) -> list[Rect]:
    """(1), (11): every dynamic and label rectangle of any member of k."""
    members = cls.members.values()
    return [r for m in members for r in m.dynamic] + [
        lab.rect for m in members for lab in m.labels
    ]


def allowed(cls: ScreenClass, width: int, height: int) -> np.ndarray:
    """(1): inside window or book_panel, outside hits, slot contents, dynamic and labels."""
    regions = [cls.window] + ([cls.book_panel] if cls.book_panel else [])
    hits = [e.hit for e in cls.elements]
    contents = [slot_content(e) for e in cls.elements if e.kind == "slot"]
    excluded = union(hits + contents + drawn_rects(cls), width, height)
    return union(regions, width, height) & ~excluded


def canvas(member: Member, textures_dir: Path, width: int, height: int) -> np.ndarray:
    """(1): Gamma_m, the member's blits, RGBA, source-over in draw order on a transparent
    canvas."""
    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for b in member.blits:
        with Image.open(textures_dir / b.texture) as texture:
            part = texture.convert("RGBA").crop((b.u, b.v, b.u + b.w, b.v + b.h))
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer.paste(part, (b.x, b.y))
        out = Image.alpha_composite(out, layer)
    return np.asarray(out)


def opaque_canvas(
    member: Member, textures_dir: Path, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Notation: bar Gamma_m (RGB, black where alpha < 255) and the pixels of alpha 255."""
    rgba = canvas(member, textures_dir, width, height)
    opaque = rgba[..., 3] == 255
    return np.where(opaque[..., None], rgba[..., :3], 0), opaque


def opaque_pairs(
    cls: ScreenClass, region: np.ndarray, textures_dir: Path, width: int, height: int
) -> dict[str, Pair]:
    """(1), (11): member -> (bar Gamma_m, M = pixels of `region` with alpha 255); members
    with an empty M skipped."""
    pairs = {}
    for name, member in cls.members.items():
        t, opaque = opaque_canvas(member, textures_dir, width, height)
        mask = opaque & region
        if mask.any():
            pairs[name] = (t, mask)
    return pairs


def member_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> dict[str, Pair]:
    """(1): member -> (T, M) of variant (a); members with an empty M skipped."""
    return opaque_pairs(cls, allowed(cls, width, height), textures_dir, width, height)


def texture_pairs(
    cls: ScreenClass, textures_dir: Path, width: int, height: int
) -> list[Pair]:
    """(1): variant (a), one (T, M) pair per member with a non-empty mask."""
    return list(member_pairs(cls, textures_dir, width, height).values())


def block_pair(x: np.ndarray, q: float) -> Pair:
    """(2) on one row block of a build's frames: lower-median template and spread sigma_k."""
    template = np.quantile(x, 0.5, axis=0, method="inverted_cdf")
    spread = np.quantile(distance(x, template), q, axis=0, method="inverted_cdf")
    return template, spread


def variance_pairs(
    blocks: Iterable[tuple[int, np.ndarray]],
    builds: Mapping[str, np.ndarray],
    q: float,
    s_max: int,
    height: int,
    step: Callable[[], None],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(2) of one class: build -> template (uint8), spread sigma_k, mask sigma_k <= s_max,
    each H x W, every build from the same row block (r0, frames x rows x W x 3); step()
    after each row block."""
    built = {}
    for r0, x in blocks:
        for b, selected in builds.items():
            rows = block_pair(x[selected], q)
            if b not in built:
                built[b] = [np.zeros((height, *a.shape[1:]), a.dtype) for a in rows]
            for full, a in zip(built[b], rows):
                full[r0 : r0 + len(a)] = a
        step()
    return {b: (t, s, s <= s_max) for b, (t, s) in built.items()}


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
    step: Callable[[], None],
) -> dict[str, np.ndarray]:
    """(3): variant (c) mask of each given class (the >= 2 with a (b) template);
    step() after each class's mask."""
    masks = {}
    for k in templates:
        masks[k] = top_patches(ratio(k, templates, spreads[k], patch, eps), n_p, patch)
        step()
    return masks
