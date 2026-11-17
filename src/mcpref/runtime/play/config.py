"""Launch settings of specs/runtime-play.md."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, positive_int, read_yaml, relative
from mcpref.training.devices import device


@dataclass(frozen=True)
class PlayConfig:
    mode: str
    trajectories: int
    out_dir: str
    device: str
    sampling: str
    rl_cycle: str


def load_config(path: Path) -> PlayConfig:
    raw = config_mapping(read_yaml(path), path, PlayConfig)
    if raw["mode"] not in ("pretrained", "rl"):
        raise ValueError("mode must be pretrained or rl")
    positive_int("trajectories", raw["trajectories"])
    device(raw["device"])
    for name in ("out_dir", "sampling", "rl_cycle"):
        relative(name, raw[name])
    return PlayConfig(**raw)
