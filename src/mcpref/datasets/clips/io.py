"""Notation of specs/pretrain.md: a clip of perception-data's format read from disk."""

import json
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Clip:
    """Notation: the L state frames (yuv420p rows), W, H, clip.json, lines.jsonl."""

    frames: np.ndarray
    width: int
    height: int
    info: dict
    lines: list[dict]


def read_clip(path: Path, states: int, width: int, height: int) -> Clip:
    """Notation: the clip at the resolved `path`; frames of shape (L, W H 3 / 2)."""
    frames = np.fromfile(path / "frames.yuv", dtype=np.uint8)
    frames = frames.reshape(states, width * height * 3 // 2)
    info = json.loads((path / "clip.json").read_bytes())
    pieces = (path / "lines.jsonl").read_bytes().split(b"\n")[:-1]
    lines = [json.loads(piece.decode("utf-8")) for piece in pieces]
    return Clip(frames, width, height, info, lines)


def clip_tail(clip: Clip, history: int) -> Clip:
    """Keep the final states and aligned intervals, retaining absolute tick indices."""
    start = max(0, len(clip.frames) - history)
    if start == 0:
        return clip
    fields = ("states", "gui", "classes", "pitch", "actions", "n")
    info = clip.info | {
        key: clip.info[key][start:] for key in fields if key in clip.info
    }
    offset = info["states"][0] - clip.info["states"][0]
    return Clip(clip.frames[start:], clip.width, clip.height, info, clip.lines[offset:])


def _read_paths(read: Callable[[str], Clip], paths: list[str]) -> list[Clip]:
    return [read(path) for path in paths]


def prefetch_clips(
    read: Callable[[str], Clip], batches: Iterable[list[str]]
) -> Iterator[list[Clip]]:
    """Read only the current and next batches, preserving consumption and error order."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        paths = iter(batches)
        try:
            first = next(paths)
        except StopIteration:
            return
        clips = _read_paths(read, first)
        for next_paths in paths:
            future = executor.submit(_read_paths, read, next_paths)
            yield clips
            clips = future.result()
        yield clips
