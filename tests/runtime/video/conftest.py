"""Small saved-format trajectories with independently identifiable observations."""

import json
from fractions import Fraction

import pytest
import torch

from mcpref.runtime.video.config import VideoConfig


def write_trajectory(source, shard, steps):
    directory = source / "shards" / f"{shard:06d}" / "steps"
    directory.mkdir(parents=True)
    for local in range(steps):
        index = shard * steps + local
        torch.save(
            {
                "index": index,
                "before": {
                    "frame": torch.full((24, 32, 3), index * 20, dtype=torch.uint8)
                },
                "after": {
                    "frame": torch.full(
                        (24, 32, 3), (index + 1) * 20, dtype=torch.uint8
                    )
                },
                "tree": {"unneeded": torch.zeros(3)},
            },
            directory / f"{local:06d}.pt",
        )
    (directory.parent / "state.pt").write_bytes(b"must not be loaded")


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "sampling"
    write_trajectory(source, 0, 3)
    write_trajectory(source, 1, 3)
    (source / "shards/000002.part").mkdir()
    (source / "run.json").write_text(
        json.dumps({"config": {"steps": 3}, "counts": {"trajectories": 0, "steps": 0}})
    )
    return source


@pytest.fixture
def cfg():
    return VideoConfig(
        "data/test-sampling",
        "data/test.mp4",
        None,
        None,
        4,
        "ffmpeg",
        Fraction(10),
        18,
        "veryfast",
    )
