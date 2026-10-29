"""Interface config, J and (7) of specs/engine.md: configs/engine.yaml."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import (
    SHA256,
    config_mapping,
    integer,
    non_empty,
    number,
    positive,
    positive_int,
    relative,
    repo_path,
)

M = ("lwjgl", "lwjgl-opengl", "lwjgl-glfw", "lwjgl-stb", "lwjgl-openal")
M += ("lwjgl-jemalloc", "lwjgl-tinyfd")
KINDS = ["classes", "natives"]
NON_EMPTY = ("engine_url", "lwjgl_url", "lwjgl_version")
JAR = "engine/build/libs/mcprec-6.13.jar"


@dataclass(frozen=True)
class EngineConfig:
    """Interface: configs/engine.yaml; minestudio_dir absolute under the repository root."""

    minestudio_dir: Path
    engine_url: str
    engine_size: int
    engine_sha256: str
    lwjgl_url: str
    lwjgl_version: str
    lwjgl_sha256: dict[str, dict[str, str]]
    http_timeout: float
    http_retries: int
    retry_delay: float
    render_size: tuple[int, int]
    seed: int
    warmup_steps: int
    steps: int
    min_rate: float


def sha256(name: str, value: object) -> str:
    """Interface: 64 characters of [0-9a-f]."""
    if type(value) is not str or not SHA256.fullmatch(value):
        raise ValueError(f"{name} {value!r} is not 64 lowercase hex digits")
    return value


def mapping(name: str, value: object, keys: list[str]) -> dict:
    """Interface: a mapping with exactly `keys`, in this order."""
    if type(value) is not dict:
        raise ValueError(f"{name} {value!r} is not a mapping")
    if list(value) != keys:
        raise ValueError(f"{name} keys {list(value)} are not {keys}")
    return value


def lwjgl_sha256(value: object) -> dict[str, dict[str, str]]:
    """Interface: per module of M, the sha256 of its classes and natives jars."""
    modules = mapping("lwjgl_sha256", value, list(M))
    return {
        m: {
            kind: sha256(f"lwjgl_sha256[{m}][{kind}]", h)
            for kind, h in mapping(f"lwjgl_sha256[{m}]", hashes, KINDS).items()
        }
        for m, hashes in modules.items()
    }


def retry_delay(value: object) -> float:
    """Interface: a finite `number` >= 0."""
    x = number("retry_delay", value)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"retry_delay {value!r} is not a finite number >= 0")
    return x


def render_size(value: object) -> tuple[int, int]:
    """Interface: a list of two integers >= 1, width then height."""
    if (
        type(value) is not list
        or len(value) != 2
        or not all(type(x) is int and x >= 1 for x in value)
    ):
        raise ValueError(f"render_size {value!r} is not a list of two integers >= 1")
    return value[0], value[1]


def load_config(path: Path) -> EngineConfig:
    """Interface: the YAML file, validated, into EngineConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, EngineConfig)
    relative("minestudio_dir", raw["minestudio_dir"])
    read = {k: non_empty(k, raw[k]) for k in NON_EMPTY}
    read |= {
        "minestudio_dir": repo_path("minestudio_dir", raw["minestudio_dir"]),
        "engine_size": positive_int("engine_size", raw["engine_size"]),
        "engine_sha256": sha256("engine_sha256", raw["engine_sha256"]),
        "lwjgl_sha256": lwjgl_sha256(raw["lwjgl_sha256"]),
        "http_timeout": positive("http_timeout", raw["http_timeout"]),
        "http_retries": integer("http_retries", raw["http_retries"], 0, 100),
        "retry_delay": retry_delay(raw["retry_delay"]),
        "render_size": render_size(raw["render_size"]),
        "seed": integer("seed", raw["seed"], 0, 2**63 - 1),
        "warmup_steps": integer("warmup_steps", raw["warmup_steps"], 0, 10**6),
        "steps": positive_int("steps", raw["steps"]),
        "min_rate": positive("min_rate", raw["min_rate"]),
    }
    return EngineConfig(**raw | read)


def engine_jar(cfg: EngineConfig) -> Path:
    """Notation J: the engine jar MineStudio's check_engine requires."""
    return cfg.minestudio_dir / JAR


def use_engine(cfg: EngineConfig) -> None:
    """(7): J missing raises, nothing set; else MineStudio's directory and CPU rendering."""
    jar = engine_jar(cfg)
    if not jar.exists():
        raise FileNotFoundError(f"engine missing: {jar}; run scripts/setup_env.sh")
    os.environ["MINESTUDIO_DIR"] = str(cfg.minestudio_dir)
    os.environ["MINESTUDIO_GPU_RENDER"] = "0"
