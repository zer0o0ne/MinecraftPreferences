"""Shared repository paths and configuration readers."""

import hashlib
import math
import re
from dataclasses import fields
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PurePosixPath

import yaml

REPO = Path(__file__).resolve().parents[2]
SHA256 = re.compile("[0-9a-f]{64}")
GROUP_KEYS = {"lr", "weight_decay"}


def config_mapping(raw: object, path: Path, cls: type) -> dict:
    """A YAML mapping with exactly the fields of the config dataclass cls."""
    names = [f.name for f in fields(cls)]
    keys = raw if isinstance(raw, dict) else {}
    missing = [n for n in names if n not in keys]
    unknown = [k for k in keys if k not in names]
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    return keys


def repo_path(name: str, value: object) -> Path:
    """A config path string, resolved against the repository root."""
    if isinstance(value, str):
        return REPO / value
    raise ValueError(f"{name} {value!r} is not a path string")


def non_empty(name: str, value: object) -> str:
    """A non-empty string."""
    if type(value) is not str or not value:
        raise ValueError(f"{name} {value!r} is not a non-empty string")
    return value


def number(name: str, value: object) -> float:
    """A YAML integer or float, not a boolean, read with float(); an integer float()
    cannot represent is not a number."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} {value!r} is not an integer or float")
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"{name} {value!r} is too large for a float") from None


def positive(name: str, value: object) -> float:
    """A `number` > 0."""
    x = number(name, value)
    if not x > 0:
        raise ValueError(f"{name} {value!r} is not > 0")
    return x


def fraction(name: str, value: object) -> float:
    """A `number` in [0, 1)."""
    x = number(name, value)
    if not 0 <= x < 1:
        raise ValueError(f"{name} {value!r} is not in [0, 1)")
    return x


def exact(name: str, value: object) -> Fraction:
    """The exact value Fraction(str(x)) of a finite YAML integer or float, not a
    boolean."""
    if not math.isfinite(number(name, value)):
        raise ValueError(f"{name} {value!r} is not finite")
    return Fraction(str(value))


def integer(name: str, value: object, low: float, high: float) -> int:
    """A YAML integer, not a boolean, in [low, high]."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} {value!r} is not an integer in [{low}, {high}]")
    return value


def positive_int(name: str, value: object) -> int:
    """A YAML integer, not a boolean, >= 1."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} {value!r} is not an integer >= 1")
    return value


def relative(name: str, value: object) -> None:
    """A non-empty relative path string without a `..` component."""
    path = PurePosixPath(value) if type(value) is str and value else None
    if path is None or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{name} {value!r} is not a non-empty relative path without .."
        )


def even_heads(width: int, heads: int) -> None:
    """`width` divisible by `heads` with an even quotient d_h (the pairs of RoPE)."""
    if width % heads or width // heads % 2:
        raise ValueError(f"width {width} is not heads {heads} times an even number")


def betas(value: object) -> tuple[float, float]:
    """Notation: a list of exactly two numbers, each in [0, 1)."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"betas {value!r} are not a list of two")
    return fraction("betas", value[0]), fraction("betas", value[1])


def quantiles(value: object) -> tuple[float, ...]:
    """(R): distinct quantiles strictly increasing in (0, 1)."""
    if not isinstance(value, list) or not value:
        raise ValueError("quantiles must be a non-empty list")
    result = tuple(number("quantiles", item) for item in value)
    if any(not 0 < q < 1 for q in result) or any(b <= a for a, b in pairwise(result)):
        raise ValueError("quantiles must be distinct and strictly increasing in (0, 1)")
    return result


def group(name: object, value: object) -> dict[str, float]:
    """Notation: a group: exactly `lr` (> 0) and `weight_decay` (>= 0)."""
    if type(name) is not str or not isinstance(value, dict) or set(value) != GROUP_KEYS:
        raise ValueError(f"group {name!r}: {value!r} is not lr and weight_decay")
    decay = number("weight_decay", value["weight_decay"])
    if not decay >= 0:
        raise ValueError(f"group {name}: weight_decay {decay} is not >= 0")
    return {"lr": positive("lr", value["lr"]), "weight_decay": decay}


def groups(value: object) -> dict[str, dict[str, float]]:
    """Notation: a mapping from string names to groups, in config order."""
    if type(value) is not dict:
        raise ValueError(f"groups {value!r} is not a mapping")
    return {name: group(name, g) for name, g in value.items()}


def boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} {value!r} is not a boolean")
    return value


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _remove(config: dict, dotted: str, path: Path) -> None:
    parts = dotted.split(".")
    if not dotted or not all(parts):
        raise ValueError(f"{path}: remove path {dotted!r} is not a dotted mapping path")
    current = config
    for part in parts[:-1]:
        if not isinstance(current.get(part), dict):
            raise ValueError(f"{path}: unknown remove path {dotted!r}")  # noqa: TRY004
        current = current[part]
    if parts[-1] not in current:
        raise ValueError(f"{path}: unknown remove path {dotted!r}")
    del current[parts[-1]]


def _string_list(raw: dict, key: str, path: Path) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError(f"{path}: {key} must be a list of strings")
    return value


def read_yaml(path: Path) -> object:
    """Read YAML and resolve ordered `extends` parents and dotted `remove` paths."""
    path = path.resolve()
    return _read_yaml(path, ())


def resolve_config(raw: dict, path: Path, parents: tuple[Path, ...] = ()) -> dict:
    """Resolve inheritance of a YAML mapping already read from disk."""
    extends = _string_list(raw, "extends", path)
    remove = _string_list(raw, "remove", path)
    result = {}
    for name in extends:
        relative("extends", name)
        parent = _read_yaml(REPO / name, (*parents, path))
        if not isinstance(parent, dict):
            raise ValueError(f"{path}: parent {name!r} is not a mapping")  # noqa: TRY004
        result = _merge(result, parent)
    for dotted in remove:
        _remove(result, dotted, path)
    own = {key: value for key, value in raw.items() if key not in {"extends", "remove"}}
    return _merge(result, own)


def _read_yaml(path: Path, parents: tuple[Path, ...]) -> object:
    if path in parents:
        chain = " -> ".join(str(item) for item in (*parents, path))
        raise ValueError(f"config extends cycle: {chain}")
    raw = yaml.safe_load(path.read_text())
    return resolve_config(raw, path, parents) if isinstance(raw, dict) else raw


def config_sha256(path: Path) -> str:
    """Hash a YAML file and every ancestor in extends order, without path names."""
    digest = hashlib.sha256()

    def add(path: Path, parents: tuple[Path, ...]) -> None:
        path = path.resolve()
        if path in parents:
            chain = " -> ".join(str(item) for item in (*parents, path))
            raise ValueError(f"config extends cycle: {chain}")
        raw = yaml.safe_load(path.read_text())
        extends = _string_list(raw, "extends", path) if isinstance(raw, dict) else []
        digest.update(path.read_bytes())
        for name in extends:
            relative("extends", name)
            digest.update(b"\0")
            add(REPO / name, (*parents, path))

    add(path, ())
    return digest.hexdigest()
