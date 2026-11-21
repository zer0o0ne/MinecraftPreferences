"""Read grid clip references with one bounded, verified shard per reader thread."""

import hashlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.frames import decoded
from mcpref.datasets.continuous.views.clips import FORMAT
from mcpref.infrastructure.shards.store import shard_path

_LOCAL = threading.local()


class ShardReader:
    def __init__(self, root: Path, ffmpeg: str, timeout: int, width: int, height: int):
        self.root = root
        self.cfg = SimpleNamespace(
            ffmpeg=ffmpeg, ffmpeg_timeout=timeout, width=width, height=height
        )
        self.shard, self.items, self.frames = None, [], []
        self.state = {}

    def metadata(self, index: int):
        if self.shard != index:
            path = shard_path(self.root / "shards", index)
            state = json.loads((path / "state.json").read_text())
            payload = path / "observations.jsonl"
            if digest(payload) != state["files"]["observations.jsonl"]:
                raise ValueError("Observation shard checksum differs")
            self.items = [json.loads(line) for line in payload.read_text().splitlines()]
            if len(self.items) != state["observations"]:
                raise ValueError("Observation shard length differs")
            self.shard, self.frames, self.state = index, [], state
        return self.items

    def frame(self, index: int, position: int):
        self.metadata(index)
        if not self.frames:
            path = shard_path(self.root / "shards", index) / "frames.mkv"
            if digest(path) != self.state["files"]["frames.mkv"]:
                raise ValueError("Video shard checksum differs")
            frames = [frame for _, frame in decoded(self.cfg, path, len(self.items))]
            checksum = hashlib.sha256()
            for frame in frames:
                checksum.update(frame)
            if checksum.hexdigest() != self.state["raw_sha256"]:
                raise ValueError("Decoded video differs from original observations")
            self.frames = frames
        return self.frames[position]


def reader_for(root: Path, width: int, height: int):
    run = json.loads((root / "run.json").read_text())
    if not run.get("complete"):
        raise ValueError("Referenced pretraining dataset is incomplete")
    if (width, height) != (run["width"], run["height"]):
        raise ValueError("Referenced dataset frame dimensions differ")
    key = (root, width, height, run["fingerprint"])
    if getattr(_LOCAL, "key", None) != key:
        transport = run["fingerprint"]["source_config"]["config"]
        ffmpeg = transport["ffmpeg"]
        if not Path(ffmpeg).exists():
            ffmpeg = shutil.which(Path(ffmpeg).name)
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to read the shared dataset")
        _LOCAL.reader = ShardReader(
            root, ffmpeg, transport["ffmpeg_timeout"], width, height
        )
        _LOCAL.key = key
    return _LOCAL.reader


def read_reference(path: Path, width: int, height: int, *, reader=None):
    from mcpref.datasets.clips.io import Clip

    info = json.loads((path / "clip.json").read_text())
    if info["format"] != FORMAT:
        raise ValueError("Unknown clip reference format")
    root = REPO / info["dataset"]
    reader = reader if reader is not None else reader_for(root, width, height)
    frames, lines = [], []
    for expected, location in zip(info["observation_ids"], info["frames"], strict=True):
        shard, position = location
        item = reader.metadata(shard)[position]
        if item["id"] != expected:
            raise ValueError("Clip reference points to a different observation")
        frames.append(reader.frame(shard, position))
        lines.append(item["meta"])
    return Clip(
        np.frombuffer(b"".join(frames), np.uint8).reshape(len(frames), -1),
        width,
        height,
        info,
        lines,
    )
