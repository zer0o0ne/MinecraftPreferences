"""(1)-(3) of specs/vpt-gui-frames.md: eligible frames, episode reasons, the plan."""

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import NamedTuple

import numpy as np

from mcpref.vpt.config import GuiFramesConfig

Fetch = Callable[[str], bytes | None]


class PlannedFrame(NamedTuple):
    """Interface `plan`: frame i of episode e with its lines (l_{i-1}, l_i)."""

    id: str
    subset: str
    relpath: str
    version: str
    frame: int
    lines: tuple[Mapping, Mapping]


class EpisodeRecord(NamedTuple):
    """Interface `plan`: a drawn episode, its reason (2), its options or None."""

    subset: str
    relpath: str
    reason: str | None
    options: Mapping | None


def eligible_frames(lines: Sequence[Mapping]) -> list[int]:
    """(1): F_e, frames i in [1, n-1] with isGuiOpen in both l_i and l_{i-1}."""
    n = len(lines)
    return [
        i for i in range(1, n) if lines[i]["isGuiOpen"] and lines[i - 1]["isGuiOpen"]
    ]


def version_key(version: str) -> tuple[int, ...]:
    """Notation: a version compared as a tuple of integers."""
    return tuple(int(part) for part in version.split("."))


def version_ok(version: str, min_version: str) -> bool:
    """(2) `version`: v_e >= min_version."""
    return version_key(version) >= version_key(min_version)


def options_ok(options: Mapping, required: Mapping) -> bool:
    """(2) `options`: no required key differs from (or is absent in) the options file."""
    return all(
        key in options and options[key] == value for key, value in required.items()
    )


def json_lines(body: bytes) -> list:
    """(2) `malformed`: one JSON value per line; a final newline ends the last line."""
    return [json.loads(line) for line in body.removesuffix(b"\n").split(b"\n")]


def index(body: bytes) -> tuple[str, list[str]]:
    """(3): basedir and pool, the relpaths in index order with exact repeats removed."""
    data = json.loads(body)
    return data["basedir"], list(dict.fromkeys(data["relpaths"]))


def version_of(relpath: str) -> str:
    """Notation: v_e, the `<version>` component of data/<version>/<stem>.mp4."""
    return PurePosixPath(relpath).parts[1]


def examine(
    cfg: GuiFramesConfig, basedir: str, relpath: str, fetch: Fetch
) -> tuple[str | None, Mapping | None, list | None]:
    """(2): the reason of e (checks in order), then its options and lines once parsed."""
    if not version_ok(version_of(relpath), cfg.min_version):
        return "version", None, None
    stem = basedir + relpath.removesuffix(".mp4")
    options = fetch(stem + "-options.json")
    lines = None if options is None else fetch(stem + ".jsonl")
    if options is None or lines is None:
        return "missing", None, None
    if not options or not lines:
        return "empty", None, None
    try:
        options, lines = json.loads(options), json_lines(lines)
    except ValueError:
        return "malformed", None, None
    if not options_ok(options, cfg.required_options):
        return "options", options, lines
    return ("no_gui" if not eligible_frames(lines) else None), options, lines


def planned(
    subset: str, relpath: str, i: int, lines: Sequence[Mapping]
) -> PlannedFrame:
    """Interface: the PlannedFrame of frame i, id <version>_<stem>_<i>."""
    version, stem = version_of(relpath), PurePosixPath(relpath).stem
    pair = (lines[i - 1], lines[i])
    return PlannedFrame(f"{version}_{stem}_{i}", subset, relpath, version, i, pair)


def draw(
    cfg: GuiFramesConfig,
    rng: np.random.Generator,
    subset: str,
    fetch: Fetch,
    left: int,
    episodes: list[EpisodeRecord],
    basedir: str,
    pool: list[str],
) -> list[PlannedFrame]:
    """(3) within subset s: episodes popped until one is valid or the pool is empty."""
    while pool:
        relpath = pool.pop(rng.integers(len(pool)))
        reason, options, lines = examine(cfg, basedir, relpath, fetch)
        episodes.append(EpisodeRecord(subset, relpath, reason, options))
        if reason is None:
            eligible = eligible_frames(lines)
            r = min(cfg.per_episode, len(eligible), left)
            chosen = rng.choice(eligible, size=r, replace=False, shuffle=True)
            return [planned(subset, relpath, int(i), lines) for i in chosen]
    return []


def plan(
    cfg: GuiFramesConfig, fetch: Fetch
) -> tuple[list[PlannedFrame], list[EpisodeRecord]]:
    """(3): subset, episodes, frames, drawn by one Generator(PCG64(seed))."""
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    indices = {s: index(fetch(url)) for s, url in cfg.index_urls.items()}
    frames, episodes = [], []
    while len(frames) < cfg.n_frames:
        active = [s for s, (_, pool) in indices.items() if pool]
        if not active:
            break
        s = active[rng.integers(len(active))]
        left = cfg.n_frames - len(frames)
        frames += draw(cfg, rng, s, fetch, left, episodes, *indices[s])
        progress = f"{len(frames)}/{cfg.n_frames} frames, {len(episodes)} episodes"
        print(f"planned {progress}", file=sys.stderr)
    return frames, episodes
