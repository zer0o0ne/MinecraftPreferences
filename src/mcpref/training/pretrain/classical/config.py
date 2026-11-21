"""Classical minibatch settings over the common pretraining configuration."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, read_yaml
from mcpref.training.pretrain.config import POSITIVE_INTS, TrainingConfig, parse_values


@dataclass(frozen=True)
class PretrainConfig(TrainingConfig):
    batch: int
    micro_batch: int


def load_config(path: Path) -> PretrainConfig:
    raw = config_mapping(read_yaml(path), path, PretrainConfig)
    return PretrainConfig(**parse_values(raw, (*POSITIVE_INTS, "batch", "micro_batch")))
