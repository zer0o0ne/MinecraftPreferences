"""Frozen configuration of producer-owned diagnostics."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import (
    config_mapping,
    integer,
    number,
    positive_int,
    quantiles,
    read_yaml,
    relative,
)


@dataclass(frozen=True)
class DiagnosticsConfig:
    out_dir: str
    seed: int
    ci_level: float
    bootstrap_samples: int
    min_cluster_count: int
    quantiles: tuple[float, ...]


def load_config(path: Path) -> DiagnosticsConfig:
    raw = config_mapping(read_yaml(path), path, DiagnosticsConfig)
    relative("out_dir", raw["out_dir"])
    level = number("ci_level", raw["ci_level"])
    if not 0 < level < 1:
        raise ValueError("ci_level must be in (0, 1)")
    values = {
        "seed": integer("seed", raw["seed"], 0, 2**32 - 1),
        "bootstrap_samples": positive_int(
            "bootstrap_samples", raw["bootstrap_samples"]
        ),
        "min_cluster_count": positive_int(
            "min_cluster_count", raw["min_cluster_count"]
        ),
        "ci_level": level,
        "quantiles": quantiles(raw["quantiles"]),
    }
    return DiagnosticsConfig(**raw | values)
