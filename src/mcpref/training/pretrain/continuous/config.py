"""Configuration of specs/pretrain-continuous.md."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, read_yaml
from mcpref.training.pretrain.config import POSITIVE_INTS, TrainingConfig, parse_values


@dataclass(frozen=True)
class ContinuousConfig(TrainingConfig):
    tbptt_states: int
    preference_micro_batch: int
    archive_shard_size: int
    archive_cache_size: int


def load_config(path: Path) -> ContinuousConfig:
    raw = config_mapping(read_yaml(path), path, ContinuousConfig)
    positive = POSITIVE_INTS
    positive += (
        "tbptt_states",
        "preference_micro_batch",
        "archive_shard_size",
        "archive_cache_size",
    )
    return ContinuousConfig(**parse_values(raw, positive))
