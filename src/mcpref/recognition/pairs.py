"""(T) of specs/gui-masks.md: the template shards, (1)-(3), (10)-(13); bars `templates
catalogue`, `templates b`, `templates c`."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np

from mcpref import REPO
from mcpref.gui.label_files import yuv_path
from mcpref.gui.screens import ScreenClass
from mcpref.recognition.drawn_masks import edge_pairs_from, label_pairs, ring_pairs
from mcpref.recognition.frame_set import State
from mcpref.recognition.masks import Pair, member_pairs, patch_masks, variance_pairs
from mcpref.recognition.stored import (
    ALL,
    CATALOGUE,
    B,
    C,
    b_builds,
    b_path,
    builds,
    read_arrays,
    shard_bar,
    write_templates,
)
from mcpref.shards.store import complete_shards
from mcpref.vpt.gui_frames import yuv420p_to_rgb

if TYPE_CHECKING:
    from mcpref.recognition.evaluate import GuiMasksConfig

VARIANTS = ("a", "b", "c", "d", "e", "f")
B_KEYS = ("T", "sigma", "M")


def read_at(f: IO[bytes], offset: int, n: int) -> bytes:
    """`rgb_rows`: n bytes of an open file from `offset`."""
    f.seek(offset)
    return f.read(n)


def rgb_rows(path: Path, width: int, height: int, r0: int, r1: int) -> np.ndarray:
    """Interface: rows [r0, r1) of the RGB view (5) of a .yuv file, from its Y rows
    [r0, r1) and U, V rows [r0/2, r1/2) only."""
    half, chroma = width // 2, (width // 2) * (height // 2)
    with open(path, "rb") as f:
        planes = [read_at(f, width * r0, width * (r1 - r0))]
        for k in range(2):
            start = width * height + k * chroma + half * (r0 // 2)
            planes.append(read_at(f, start, half * ((r1 - r0) // 2)))
    return yuv420p_to_rgb(b"".join(planes), width, r1 - r0)


def class_pairs(
    cls: ScreenClass, textures_dir: Path, size: tuple[int, int]
) -> dict[str, dict[str, Pair]]:
    """(1), (10)-(12) of one class: variant -> member -> pair."""
    a, e = member_pairs(cls, textures_dir, *size), ring_pairs(cls, textures_dir, *size)
    d = label_pairs(cls, textures_dir, *size)
    return {"a": a, "d": d, "e": e, "f": edge_pairs_from(cls, a, e)}


def write_catalogue(
    directory: Path, s: int, pairs: Mapping[str, Mapping[str, Pair]]
) -> None:
    """Interface: catalogue shard s, keys <variant>/<member>/T and …/M, and its PNGs."""
    arrays, pngs = {}, {}
    for v, by_member in pairs.items():
        for m, (t, mask) in by_member.items():
            arrays[f"{v}/{m}/T"], arrays[f"{v}/{m}/M"] = t, mask
            pngs[f"{v}__{m}"] = (v, t, mask)
    write_templates(directory, s, "pairs.npz", arrays, pngs)


def catalogue_shards(
    out: Path,
    classes: Mapping[str, ScreenClass],
    textures_dir: Path,
    size: tuple[int, int],
) -> None:
    """(T): catalogue shard s = the pairs of (a), (d), (e), (f) of class s of K; bar
    `templates catalogue`."""
    directory, names = out / CATALOGUE, list(classes)
    h, n = complete_shards(directory), len(names)
    with shard_bar("templates catalogue", n, h, n, h, "class") as update:
        for s in range(h, n):
            write_catalogue(
                directory, s, class_pairs(classes[names[s]], textures_dir, size)
            )
            update()


def row_blocks(height: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Notation: the row blocks [r0, r1) of `chunk_rows` rows."""
    return [(r0, min(r0 + chunk_rows, height)) for r0 in range(0, height, chunk_rows)]


def blocks(
    frames_dir: Path, ids: list[str], size: tuple[int, int], chunk_rows: int
) -> Iterator[tuple[int, np.ndarray]]:
    """(2): r0 and the row block of the RGB views of the frames `ids`, each read by
    `rgb_rows`."""
    for r0, r1 in row_blocks(size[1], chunk_rows):
        x = [rgb_rows(yuv_path(frames_dir, {"id": i}), *size, r0, r1) for i in ids]
        yield r0, np.stack(x)


def selectors(folds: list[int], built: list[int]) -> dict[str, np.ndarray]:
    """(13): build -> its frames among a class's training frames (`folds` their folds):
    `all`, and fold l for each built l with a frame of F^tr_(-l)."""
    fold = np.array(folds)
    held = {f"fold{l}": fold != l for l in built}
    return {ALL: np.ones(len(folds), bool)} | {b: s for b, s in held.items() if s.any()}


def write_b(directory: Path, s: int, k: str, built: Mapping[str, tuple]) -> None:
    """Interface: (b) shard s, keys <build>/T, …/sigma, …/M of the builds with a non-empty
    M, and the PNGs of build `all`."""
    kept = {b: tsm for b, tsm in built.items() if tsm[2].any()}
    arrays = {f"{b}/{x}": a for b, tsm in kept.items() for x, a in zip(B_KEYS, tsm)}
    pngs = {f"b__{k}": ("b", kept[ALL][0], kept[ALL][2])} if ALL in kept else {}
    write_templates(directory, s, "pairs.npz", arrays, pngs)


def b_shards(cfg: "GuiMasksConfig", state: State) -> None:
    """(T), (2), (13): (b) shard s = the builds of class s of Q; bar `templates b`."""
    directory, height = REPO / cfg.out_dir / B, state.size[1]
    h, n = complete_shards(directory), len(state.q)
    per = len(row_blocks(height, cfg.chunk_rows))
    q, s_max = cfg.spread_quantile, cfg.spread_max
    with shard_bar("templates b", n, h, n * per, h * per, "step") as update:
        for s in range(h, n):
            ids, folds = map(list, zip(*state.train[state.q[s]]))
            x = blocks(REPO / cfg.frames_dir, ids, state.size, cfg.chunk_rows)
            sel = selectors(folds, state.built)
            built = variance_pairs(x, sel, q, s_max, height, update)
            write_b(directory, s, state.q[s], built)


def taking_part(out: Path, q: list[str], k_folds: int) -> list[list[str]]:
    """(3), (13): per build, the classes taking part in (c), from the (b) shards' keys:
    with a (b) template from all of F^tr and from the build, when at least two, and
    only when at least two have one from all of F^tr."""
    has = {k: set(b_builds(out, s)) for s, k in enumerate(q)}
    every = [k for k in q if ALL in has[k]]
    part = [[k for k in every if b in has[k]] for b in builds(k_folds)]
    return [ks if len(ks) >= 2 and len(every) >= 2 else [] for ks in part]


def c_shard(cfg: "GuiMasksConfig", q: list[str], s: int, ks: list[str], update) -> None:
    """(3) of build s over the classes ks, their templates and spreads read from the (b)
    shards one at a time; its shard with the PNGs of build `all`."""
    out, b = REPO / cfg.out_dir, builds(cfg.folds)[s]
    keys = (f"{b}/T", f"{b}/sigma")
    t, sigma = {}, {}
    for k in ks:
        t[k], sigma[k] = read_arrays(b_path(out, q.index(k)), keys)
    p = (cfg.patch, cfg.patches_per_class, cfg.eps)
    masks = patch_masks(t, sigma, *p, update)
    pngs = {f"c__{k}": ("c", t[k], m) for k, m in masks.items()} if b == ALL else {}
    write_templates(out / C, s, "masks.npz", masks, pngs)


def c_shards(cfg: "GuiMasksConfig", state: State) -> None:
    """(T), (3), (13): (c) shard s = the masks of build s, folds + 1 shards; bar
    `templates c`."""
    out = REPO / cfg.out_dir
    taking = taking_part(out, state.q, cfg.folds)
    h, counts = complete_shards(out / C), [len(ks) for ks in taking]
    total, initial = sum(counts), sum(counts[:h])
    with shard_bar("templates c", len(taking), h, total, initial, "class") as update:
        for s in range(h, len(taking)):
            c_shard(cfg, state.q, s, taking[s], update)
