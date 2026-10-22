"""Notation of specs/perception.md: configs/perception.yaml."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, integer, number, positive, positive_int, repo_path

POSITIVE_INTS = ("head_width", "min_pixels", "max_pixels", "cursor_crop")
POSITIVE_INTS += ("mouse_height",)
POSITIVE_REALS = ("c", "beta", "dual_init", "dual_slack", "rescale_factor")
POSITIVE_REALS += ("http_timeout", "retry_delay")
UNBOUNDED_INTS = ("head_layers", "decimals", "http_retries")
NON_EMPTY = ("model", "revision", "no_items", "marker", "cursor_url")
PLACEHOLDERS = {
    "meta": ("{inventory}", "{hotbar}", "{yaw}", "{pitch}", "{x}", "{y}", "{z}"),
    "inventory_line": ("{items}",),
    "item_line": ("{type}", "{quantity}"),
}
SHA256 = re.compile("[0-9a-f]{64}")


@dataclass(frozen=True)
class PerceptionConfig:
    """Notation: config fields; `cursor` as written, resolved against the repository
    root where read or written."""

    model: str
    revision: str
    seed: int
    alpha: tuple[float, ...]
    head_layers: int
    head_width: int
    c: float
    beta: float
    dual_init: float
    dual_slack: float
    tau: float
    min_pixels: int
    max_pixels: int
    rescale_factor: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    decimals: int
    meta: str
    inventory_line: str
    item_line: str
    no_items: str
    marker: str
    cursor: str
    cursor_url: str
    cursor_sha256: str
    cursor_crop: int
    mouse_height: int
    http_timeout: float
    http_retries: int
    retry_delay: float


def non_empty(name: str, value: object) -> str:
    """Notation: a non-empty string."""
    if type(value) is not str or not value:
        raise ValueError(f"{name} {value!r} is not a non-empty string")
    return value


def template(name: str, value: object) -> str:
    """Notation: a string containing every placeholder of `name`."""
    if type(value) is not str or not all(p in value for p in PLACEHOLDERS[name]):
        raise ValueError(f"{name} {value!r} is not a string with {PLACEHOLDERS[name]}")
    return value


def numbers(name: str, value: object, length: int | None) -> tuple[float, ...]:
    """Notation: a non-empty list of numbers (exactly `length` if given), as floats."""
    if not isinstance(value, list) or not value or length not in (None, len(value)):
        raise ValueError(
            f"{name} {value!r} is not a list of {length or 'some'} numbers"
        )
    return tuple(number(name, v) for v in value)


def all_positive(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    """Notation: every entry > 0."""
    if not all(v > 0 for v in values):
        raise ValueError(f"{name} {values} has an entry not > 0")
    return values


def sha256_hex(value: object) -> str:
    """Notation: 64 lowercase hex characters."""
    if type(value) is not str or not SHA256.fullmatch(value):
        raise ValueError(f"cursor_sha256 {value!r} is not 64 lowercase hex characters")
    return value


def nonnegative(name: str, value: object) -> float:
    """Notation: a `number` >= 0."""
    x = number(name, value)
    if not x >= 0:
        raise ValueError(f"{name} {value!r} is not >= 0")
    return x


def load_config(path: Path) -> PerceptionConfig:
    """Interface: the YAML file, validated, into PerceptionConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PerceptionConfig)
    repo_path("cursor", raw["cursor"])
    read = {k: non_empty(k, raw[k]) for k in NON_EMPTY}
    read |= {k: template(k, raw[k]) for k in PLACEHOLDERS}
    read |= {k: positive_int(k, raw[k]) for k in POSITIVE_INTS}
    read |= {k: positive(k, raw[k]) for k in POSITIVE_REALS}
    read |= {k: integer(k, raw[k], 0, math.inf) for k in UNBOUNDED_INTS}
    read |= {
        "seed": integer("seed", raw["seed"], 0, 2**32 - 1),
        "tau": nonnegative("tau", raw["tau"]),
        "alpha": all_positive("alpha", numbers("alpha", raw["alpha"], None)),
        "image_mean": numbers("image_mean", raw["image_mean"], 3),
        "image_std": all_positive(
            "image_std", numbers("image_std", raw["image_std"], 3)
        ),
        "cursor_sha256": sha256_hex(raw["cursor_sha256"]),
    }
    if read["min_pixels"] > read["max_pixels"]:
        raise ValueError(
            f"min_pixels {raw['min_pixels']} > max_pixels {raw['max_pixels']}"
        )
    return PerceptionConfig(**raw | read)
