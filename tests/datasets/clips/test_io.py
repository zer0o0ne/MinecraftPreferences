"""`read_clip` and `Clip` of src/mcpref/datasets/clips/io.py against specs/pretrain.md
(Notation: Clip at p) on hand-written files."""

import dataclasses
from threading import Event

import numpy as np
import pytest

from mcpref.datasets.clips.io import Clip, prefetch_clips, read_clip

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


def test_prefetch_preserves_order_and_never_reads_beyond_next():
    reads, started, release = [], Event(), Event()

    def read(path):
        reads.append(path)
        if path == "b":
            started.set()
            assert release.wait(2)
        return path

    batches = prefetch_clips(read, [["a"], ["b"], ["c"]])
    assert next(batches) == ["a"]
    assert started.wait(2)
    assert reads == ["a", "b"]
    release.set()
    assert next(batches) == ["b"]
    assert next(batches) == ["c"]
    with pytest.raises(StopIteration):
        next(batches)
    assert reads == ["a", "b", "c"]


def test_prefetch_propagates_error_at_normal_consumption_point():
    reads, failure = [], ValueError("broken next clip")

    def read(path):
        reads.append(path)
        if path == "bad":
            raise failure
        return path

    batches = prefetch_clips(read, [["a"], ["bad"], ["not-read"]])
    assert next(batches) == ["a"]
    with pytest.raises(ValueError) as caught:
        next(batches)
    assert caught.value is failure
    assert reads == ["a", "bad"]
