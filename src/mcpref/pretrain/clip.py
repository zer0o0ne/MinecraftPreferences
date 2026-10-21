"""Notation of specs/pretrain.md: a clip of perception-data's format read from disk."""

import json
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
