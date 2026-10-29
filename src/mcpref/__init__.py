"""Minecraft agent training pipeline (AGENTS.md); REPO: the repository root; the config
readers shared by the blocks' load_config, SHA256 the pattern of a sha256 they check."""

import math
import re
from dataclasses import fields
from fractions import Fraction
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
SHA256 = re.compile("[0-9a-f]{64}")


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
