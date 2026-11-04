"""Notation of specs/mcts.md: configs/mcts.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, number, positive, positive_int

POSITIVE_INTS = ("simulations", "min_n_sampling", "depth", "width", "latency_calls")
POSITIVE_REALS = ("temperature", "min_p", "epsilon", "latency_ms", "latency_fail_ms")


@dataclass(frozen=True)
class MctsConfig:
    """Notation: config fields."""

    simulations: int
    min_n_sampling: int
    depth: int
    width: int
    latency_calls: int
    c_puct: float
    discount: float
    temperature: float
    min_p: float
    epsilon: float
    latency_ms: float
    latency_fail_ms: float


def load_config(path: Path) -> MctsConfig:
    """Interface: the YAML file, validated, into MctsConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, MctsConfig)
    read = {k: positive_int(k, raw[k]) for k in POSITIVE_INTS}
    read |= {k: positive(k, raw[k]) for k in POSITIVE_REALS}
    read |= {k: number(k, raw[k]) for k in ("c_puct", "discount")}
    if not read["c_puct"] >= 0:
        raise ValueError(f"c_puct {raw['c_puct']!r} is not >= 0")
    if not 0 < read["discount"] <= 1:
        raise ValueError(f"discount {raw['discount']!r} is not in (0, 1]")
    if read["min_n_sampling"] > read["simulations"]:
        raise ValueError("min_n_sampling must be <= simulations")
    return MctsConfig(**read)
