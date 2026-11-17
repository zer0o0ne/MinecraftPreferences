"""Independent playback and encoder settings: specs/gameplay-video.md."""

from dataclasses import dataclass
from fractions import Fraction
from math import inf
from pathlib import Path

from mcpref.config import (
    config_mapping,
    exact,
    integer,
    non_empty,
    positive_int,
    read_yaml,
)

PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)


@dataclass(frozen=True)
class VideoConfig:
    sampling_dir: str
    output: str
    start_step: int | None
    stop_step: int | None
    last_steps: int | None
    ffmpeg: str
    fps: Fraction
    crf: int
    preset: str


def _selection(raw: dict) -> None:
    start, stop, last = (raw[key] for key in ("start_step", "stop_step", "last_steps"))
    if last is not None:
        positive_int("last_steps", last)
        if start is not None or stop is not None:
            raise ValueError("last_steps requires start_step and stop_step to be null")
    else:
        integer("start_step", start, 0, inf)
        positive_int("stop_step", stop)
        if stop <= start:
            raise ValueError("stop_step must be greater than start_step")


def load_config(path: Path) -> VideoConfig:
    raw = config_mapping(read_yaml(path), path, VideoConfig)
    _selection(raw)
    for name in ("sampling_dir", "output", "ffmpeg"):
        non_empty(name, raw[name])
    fps = exact("fps", raw["fps"])
    if fps <= 0:
        raise ValueError("fps must be positive")
    preset = non_empty("preset", raw["preset"])
    if preset not in PRESETS:
        raise ValueError(f"unsupported x264 preset: {preset}")
    crf = integer("crf", raw["crf"], 0, 51)
    return VideoConfig(**raw | {"fps": fps, "crf": crf})
