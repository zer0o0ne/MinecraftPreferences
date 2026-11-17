"""Video rate is an independent finite positive value, not the agent rate."""

from fractions import Fraction

import pytest
import yaml

from mcpref.runtime.video.config import load_config

FIELDS = {
    "sampling_dir": "data/sampling",
    "output": "data/clip.mp4",
    "start_step": None,
    "stop_step": None,
    "last_steps": 5,
    "ffmpeg": "encoder",
    "fps": 2.5,
    "crf": 21,
    "preset": "fast",
}


def test_configured_rate_is_exact(tmp_path):
    path = tmp_path / "video.yaml"
    path.write_text(yaml.safe_dump(FIELDS))
    cfg = load_config(path)
    assert (cfg.fps, cfg.crf, cfg.preset, cfg.ffmpeg) == (
        Fraction(5, 2),
        21,
        "fast",
        "encoder",
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("fps", 0),
        ("fps", -3),
        ("fps", True),
        ("fps", float("nan")),
        ("fps", float("inf")),
        ("fps", "10"),
        ("crf", -1),
        ("crf", 52),
        ("crf", 2.5),
        ("preset", "bogus"),
        ("ffmpeg", ""),
        ("unknown", 3),
        ("sampling_dir", ""),
        ("output", False),
        ("last_steps", None),
        ("last_steps", True),
        ("last_steps", 1.5),
        ("last_steps", 0),
        ("start_step", 0),
        ("stop_step", 5),
    ],
)
def test_invalid_config(tmp_path, key, value):
    path = tmp_path / "video.yaml"
    path.write_text(yaml.safe_dump(FIELDS | {key: value}))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "start,stop", [(False, 3), (0, True), (1.5, 3), (-1, 3), (2, 2), (3, 2)]
)
def test_invalid_range_config(tmp_path, start, stop):
    path = tmp_path / "video.yaml"
    path.write_text(
        yaml.safe_dump(
            FIELDS
            | {
                "start_step": start,
                "stop_step": stop,
                "last_steps": None,
            }
        )
    )
    with pytest.raises(ValueError):
        load_config(path)
