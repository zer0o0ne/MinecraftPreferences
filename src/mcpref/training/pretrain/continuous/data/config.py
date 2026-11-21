"""Index settings derive mathematical choices from the selected training config."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import REPO, config_mapping, positive_int, read_yaml, relative


@dataclass(frozen=True)
class IndexConfig:
    dataset: str
    out_dir: str
    training_config: str
    pair_clips_dir: str | None
    pair_labels_dir: str | None
    items_per_shard: int


def load_config(path: Path) -> IndexConfig:
    raw = config_mapping(read_yaml(path), path, IndexConfig)
    for key in ("dataset", "out_dir", "training_config"):
        relative(key, raw[key])
    if (raw["pair_clips_dir"] is None) != (raw["pair_labels_dir"] is None):
        raise ValueError("Additional preference paths must both be present or null")
    for key in ("pair_clips_dir", "pair_labels_dir"):
        if raw[key] is not None:
            relative(key, raw[key])
    positive_int("items_per_shard", raw["items_per_shard"])
    return IndexConfig(**raw)


def mathematics(cfg: IndexConfig) -> dict:
    train = read_yaml(REPO / cfg.training_config)
    preference = read_yaml(REPO / train["preference"])
    return {
        "tbptt_states": positive_int("tbptt_states", train["tbptt_states"]),
        "preference_history": positive_int("preference.history", preference["history"]),
        "action_mask": train["action_mask"],
    }
