"""`read_clip` and `Clip` of src/mcpref/pretrain/clip.py against specs/pretrain.md
(Notation: Clip at p) on hand-written files."""

import dataclasses

import numpy as np
import pytest

from mcpref.pretrain.clip import Clip, read_clip

LINES = (  # lines.jsonl: UTF-8, each line with its "\n"; U+2028 and NEL inside strings
    '{"tick": 0, "s": "été"}\n'
    '{"tick": 1, "s": "a b\x85c", "keys": ["w", {"x": null}]}\n'
    '{"tick": 2}\n'
)


def write(directory, frames: bytes, info: bytes, lines: bytes) -> None:
    directory.mkdir(parents=True)
    (directory / "frames.yuv").write_bytes(frames)
    (directory / "clip.json").write_bytes(info)
    (directory / "lines.jsonl").write_bytes(lines)


@pytest.mark.parametrize("states, width, height", [(3, 4, 2), (2, 6, 4), (1, 2, 2)])
def test_read_clip(tmp_path, states, width, height):
    """frames: the bytes as uint8, row i = state frame i (W H 3/2 bytes); W, H as given;
    info: clip.json parsed; lines: the lines split at "\\n" only, each parsed."""
    size = width * height * 3 // 2
    frames = bytes((7 * i + 3) % 256 for i in range(states * size))
    info = '{"id": "6.13_x_10", "relpath": "é", "n": [1, 2]}'.encode()
    write(tmp_path / "c", frames, info, LINES.encode("utf-8"))
    got = read_clip(tmp_path / "c", states, width, height)
    assert type(got) is Clip
    assert got.frames.dtype == np.uint8 and got.frames.shape == (states, size)
    for i in range(states):
        assert got.frames[i].tobytes() == frames[i * size : (i + 1) * size]
    assert (got.width, got.height) == (width, height)
    assert type(got.width) is int and type(got.height) is int
    assert got.info == {"id": "6.13_x_10", "relpath": "é", "n": [1, 2]}
    assert got.lines == [
        {"tick": 0, "s": "été"},
        {"tick": 1, "s": "a b\x85c", "keys": ["w", {"x": None}]},
        {"tick": 2},
    ]


def test_clip_is_a_frozen_dataclass():
    names = [f.name for f in dataclasses.fields(Clip)]
    assert names == ["frames", "width", "height", "info", "lines"]
    clip = Clip(np.zeros((1, 6), np.uint8), 2, 2, {}, [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        clip.width = 3
