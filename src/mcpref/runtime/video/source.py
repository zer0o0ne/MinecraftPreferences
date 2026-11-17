"""Bounded reads from an immutable snapshot of completed sampling shards."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mcpref.config import positive_int
from mcpref.infrastructure.shards.store import complete_shards, name, shard_path


@dataclass(frozen=True)
class Recording:
    directory: Path
    steps_per_shard: int
    available: int

    def select(
        self, start: int | None, stop: int | None, last: int | None
    ) -> tuple[int, int]:
        if last is not None:
            if start is not None or stop is not None:
                raise ValueError("last_steps cannot be combined with a step range")
            if not 1 <= last <= self.available:
                raise ValueError(f"last steps must be in [1, {self.available}]")
            start, stop = self.available - last, self.available
        if start is None or stop is None or not 0 <= start < stop <= self.available:
            raise ValueError(
                f"require 0 <= start_step < stop_step <= {self.available} "
                "(completed steps)"
            )
        return start, stop


def _trajectory_steps(directory: Path) -> int:
    run = json.loads((directory / "run.json").read_text())
    if "steps" in run["config"]:
        return positive_int("sampling steps", run["config"]["steps"])
    counts = run.get("counts", {})
    trajectories, steps = counts.get("trajectories"), counts.get("steps")
    if (
        type(trajectories) is int
        and trajectories > 0
        and type(steps) is int
        and steps > 0
        and steps % trajectories == 0
    ):
        return steps // trajectories
    raise ValueError(
        "sampling run has no recorded trajectory length; "
        "retry after run.json is updated"
    )


def snapshot(directory: Path) -> Recording:
    count = complete_shards(directory / "shards")
    if not count:
        raise ValueError("no completed sampling trajectories")
    steps = shard_path(directory / "shards", 0) / "steps"
    entries = {entry.name for entry in steps.iterdir()}
    n = len(entries)
    if not n or entries != {f"{name(i)}.pt" for i in range(n)}:
        raise ValueError(f"{steps}: expected contiguous step files starting at zero")
    if n != _trajectory_steps(directory):
        raise ValueError(f"{steps}: trajectory length differs from the saved run")
    return Recording(directory, n, n * count)


def _frame(value: object, path: Path) -> np.ndarray:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 3
        or value.shape[-1] != 3
        or any(size <= 0 or size % 2 for size in value.shape[:2])
    ):
        raise ValueError(
            f"{path}: expected uint8 RGB frame with positive even dimensions"
        )
    return value.numpy()


def frames(recording: Recording, start: int, stop: int) -> Iterator[np.ndarray]:
    """Emit s_start, ..., s_stop once each, including both endpoint observations."""
    shape = None
    for index in range(start, stop):
        shard, local = divmod(index, recording.steps_per_shard)
        path = shard_path(recording.directory / "shards", shard)
        path = path / "steps" / f"{name(local)}.pt"
        record = torch.load(path, map_location="cpu", weights_only=True)
        if type(record["index"]) is not int or record["index"] != index:
            raise ValueError(f"{path}: expected global step {index}")
        for side in ("before", "after") if index == start else ("after",):
            frame = _frame(record[side]["frame"], path)
            if shape is not None and frame.shape != shape:
                raise ValueError(f"{path}: frame dimensions changed")
            shape = frame.shape
            yield frame
        del record
