"""Notation of specs/policy.md: configs/policy.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, even_heads, fraction, integer, positive, positive_int

POSITIVE_INTS = ("history", "width", "heads", "hidden", "layers", "shift_hidden")
POSITIVE_INTS += ("latency_calls",)
POSITIVE_REALS = ("rope_base", "norm_eps", "shift_weight", "latency_ms")
POSITIVE_REALS += ("latency_fail_ms",)


@dataclass(frozen=True)
class PolicyConfig:
    """Notation: config fields."""

    history: int
    width: int
    heads: int
    hidden: int
    layers: int
    shift_hidden: int
    rope_base: float
    norm_eps: float
    action_dropout: float
    shift_weight: float
    seed: int
    latency_ms: float
    latency_fail_ms: float
    latency_calls: int


def load_config(path: Path) -> PolicyConfig:
    """Interface: the YAML file, validated, into PolicyConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PolicyConfig)
    read = {k: positive_int(k, raw[k]) for k in POSITIVE_INTS}
    read |= {k: positive(k, raw[k]) for k in POSITIVE_REALS}
    read |= {"seed": integer("seed", raw["seed"], 0, 2**32 - 1)}
    read |= {"action_dropout": fraction("action_dropout", raw["action_dropout"])}
    even_heads(read["width"], read["heads"])
    return PolicyConfig(**read)
