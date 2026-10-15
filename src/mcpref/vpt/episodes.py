"""(1)-(3) of specs/vpt-gui-frames.md: eligible frames, episode reasons, the draws and
their frames; the reasons without `no_gui` and the draw without the frames, shared with
specs/perception-data.md (1)."""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import PurePosixPath
from typing import NamedTuple

import numpy as np

from mcpref.vpt.config import Access, GuiFramesConfig

Fetch = Callable[[str], bytes | None]


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


def episode_name(relpath: str) -> str:
    """Interface: <version>_<stem>, the start of an episode's ids and file names."""
    return f"{version_of(relpath)}_{PurePosixPath(relpath).stem}"


class Examined(NamedTuple):
    """(2) without `no_gui`: the reason, then the options file and the .jsonl, raw and
    parsed, as far as they were read and parsed."""

    reason: str | None
    options_file: bytes | None
    options: Mapping | None
    jsonl: bytes | None
    lines: list | None


def examine(cfg: Access, fetch: Fetch, basedir: str, relpath: str) -> Examined:
    """(2) without `no_gui`: the reason of e (checks in order), then what was read."""
    if not version_ok(version_of(relpath), cfg.min_version):
        return Examined("version", None, None, None, None)
    stem = basedir + relpath.removesuffix(".mp4")
    options_file = fetch(stem + "-options.json")
    jsonl = None if options_file is None else fetch(stem + ".jsonl")
    if options_file is None or jsonl is None:
        return Examined("missing", None, None, None, None)
    if not options_file or not jsonl:
        return Examined("empty", None, None, None, None)
    try:
        options, lines = json.loads(options_file), json_lines(jsonl)
    except ValueError:
        return Examined("malformed", options_file, None, None, None)
    reason = None if options_ok(options, cfg.required_options) else "options"
    return Examined(reason, options_file, options, jsonl, lines)


def examine_gui(
    cfg: GuiFramesConfig, fetch: Fetch, basedir: str, relpath: str
) -> Examined:
    """(2): the shared reasons, then `no_gui` (F_e empty)."""
    examined = examine(cfg, fetch, basedir, relpath)
    if examined.reason is None and not eligible_frames(examined.lines):
        return examined._replace(reason="no_gui")
    return examined


def frame_record(subset: str, relpath: str, i: int, lines: Sequence[Mapping]) -> dict:
    """Interface Produces: a frame of shard.json, id <version>_<stem>_<i>, l_{i-1}, l_i."""
    name, version = f"{episode_name(relpath)}_{i}", version_of(relpath)
    record = {"id": name, "subset": subset, "relpath": relpath, "version": version}
    return record | {"frame": i, "prev_line": lines[i - 1], "line": lines[i]}


def draw(
    rng: np.random.Generator,
    indices: Mapping[str, tuple[str, list[str]]],
    examine: Callable[[str, str], Examined],
) -> Iterator[tuple[str, str, Examined]]:
    """(3) without the frames, indices[s] = (basedir, pool): a subset s with a non-empty
    pool, its episodes popped and examined until one is valid or the pool is empty."""
    while active := [s for s, (_, pool) in indices.items() if pool]:
        s = active[rng.integers(len(active))]
        basedir, pool = indices[s]
        while pool:
            relpath = pool.pop(rng.integers(len(pool)))
            examined = examine(basedir, relpath)
            yield s, relpath, examined
            if examined.reason is None:
                break


def take(
    cfg: GuiFramesConfig,
    rng: np.random.Generator,
    subset: str,
    relpath: str,
    lines: Sequence[Mapping],
    left: int,
) -> list[dict]:
    """(3): r = min(m, |F_e|, N - planned) frames of a valid e, without replacement."""
    eligible = eligible_frames(lines)
    r = min(cfg.per_episode, len(eligible), left)
    chosen = rng.choice(eligible, size=r, replace=False, shuffle=True)
    return [frame_record(subset, relpath, int(i), lines) for i in chosen]
