"""Configuration of specs/pretrain-continuous.md."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import boolean, config_mapping, read_yaml
from mcpref.training.pretrain.config import POSITIVE_INTS, TrainingConfig, parse_values


@dataclass(frozen=True)
class ContinuousConfig(TrainingConfig):
    publish_diagnostics: bool
    tbptt_states: int
    batch_size: int
    preference_micro_batch: int
    archive_shard_size: int
    archive_cache_size: int
    archive_keep_shards: int


def load_config(path: Path) -> ContinuousConfig:
    raw = config_mapping(read_yaml(path), path, ContinuousConfig)
    positive = POSITIVE_INTS
    positive += (
        "tbptt_states",
        "batch_size",
        "preference_micro_batch",
        "archive_shard_size",
        "archive_cache_size",
        "archive_keep_shards",
    )
    values = parse_values(raw, positive)
    values["publish_diagnostics"] = boolean(
        "publish_diagnostics", raw["publish_diagnostics"]
    )
    return ContinuousConfig(**values)
