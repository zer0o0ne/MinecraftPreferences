"""Notation of specs/preference.md: configs/preference.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, integer, positive, positive_int

POSITIVE_INTS = ("history", "width", "heads", "hidden", "latency_calls")
POSITIVE_REALS = ("rope_base", "norm_eps", "latency_ms", "latency_fail_ms")
KINDS = ("S", "X")


@dataclass(frozen=True)
class PreferenceConfig:
    """Notation: config fields; `layers` as a tuple of "S" / "X"."""

    history: int
    width: int
    heads: int
    hidden: int
    layers: tuple[str, ...]
    rope_base: float
    norm_eps: float
    seed: int
    latency_ms: float
    latency_fail_ms: float
    latency_calls: int


def kinds(value: object) -> tuple[str, ...]:
    """Notation: a non-empty list of the strings "S", "X", as a tuple."""
    if not isinstance(value, list) or not value or not all(k in KINDS for k in value):
        raise ValueError(f"layers {value!r} is not a non-empty list of 'S' and 'X'")
    return tuple(value)


def load_config(path: Path) -> PreferenceConfig:
    """Interface: the YAML file, validated, into PreferenceConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PreferenceConfig)
    read = {k: positive_int(k, raw[k]) for k in POSITIVE_INTS}
    read |= {k: positive(k, raw[k]) for k in POSITIVE_REALS}
    read |= {"seed": integer("seed", raw["seed"], 0, 2**32 - 1)}
    read |= {"layers": kinds(raw["layers"])}
    width, heads = read["width"], read["heads"]
    if width % heads or width // heads % 2:
        raise ValueError(f"width {width} is not heads {heads} times an even number")
    return PreferenceConfig(**read)
