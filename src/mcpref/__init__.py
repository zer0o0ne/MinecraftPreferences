"""Minecraft agent training pipeline (AGENTS.md); REPO: the repository root; the config
readers shared by the blocks' load_config."""

from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


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


def number(name: str, value: object) -> float:
    """A YAML integer or float, not a boolean, read with float()."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} {value!r} is not an integer or float")
    return float(value)


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
