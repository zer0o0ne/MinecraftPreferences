"""`load_config`, `engine_jar` and `use_engine` of specs/engine.md (Interface: Config,
starting values, Modules; Behaviour (7)): each field's checks and messages."""

import dataclasses
import math
import os

import pytest

from mcpref.engine.config import EngineConfig, engine_jar, load_config, use_engine

from .engine_world import FIELDS, JAR, STARTING, TINY, VARS, write_config
from .jars import MODULES

SHA = "0123456789abcdef" * 4
STRING = "is not a non-empty string"
HEX = "is not 64 lowercase hex digits"
FINITE = "is not a finite number >= 0"
SIZE = "is not a list of two integers >= 1"


def load(tmp_path, **changes) -> EngineConfig:
    return load_config(write_config(tmp_path / "c.yaml", {**TINY, **changes}))


def error(tmp_path, **changes) -> str:
    with pytest.raises(ValueError) as e:
        load(tmp_path, **changes)
    return str(e.value)


def test_starting_values(repo):
    """configs/engine.yaml holds the starting values; `minestudio_dir` absolute under the
    repository root; `render_size` a tuple; `min_rate` a float; frozen."""
    cfg = load_config(repo / "configs/engine.yaml")
    assert isinstance(cfg, EngineConfig)
    assert {f.name for f in dataclasses.fields(cfg)} == set(FIELDS)
    plain = [k for k in FIELDS if k not in ("minestudio_dir", "render_size")]
    assert {k: getattr(cfg, k) for k in plain} == {k: STARTING[k] for k in plain}
    assert cfg.minestudio_dir == repo / "data/minestudio"
    assert cfg.render_size == (640, 360) and type(cfg.min_rate) is float
    assert engine_jar(cfg) == repo / "data/minestudio" / JAR
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


def test_tiny_config(root, tmp_path):
    """Every field as written; `minestudio_dir` resolved against the repository root, not
    the working directory; J under it."""
    cfg = load(tmp_path)
    plain = [k for k in FIELDS if k not in ("minestudio_dir", "render_size")]
    assert {k: getattr(cfg, k) for k in plain} == {k: TINY[k] for k in plain}
    assert cfg.minestudio_dir == root / "ms/sim" and cfg.minestudio_dir.is_absolute()
    assert cfg.render_size == (64, 36) and type(cfg.render_size) is tuple
    assert list(cfg.lwjgl_sha256) == list(MODULES)
    assert engine_jar(cfg) == root / "ms/sim/engine/build/libs/mcprec-6.13.jar"


def test_min_rate_is_a_float(tmp_path):
    assert type(load(tmp_path, min_rate=20).min_rate) is float


@pytest.mark.parametrize("key", FIELDS)
def test_missing_key(tmp_path, key):
    with pytest.raises(ValueError):
        load_config(
            write_config(
                tmp_path / "c.yaml", {k: v for k, v in TINY.items() if k != key}
            )
        )


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, extra=1)


@pytest.mark.parametrize(
    "key, value",
    [("minestudio_dir", v) for v in ("", "/abs/ms", "a/../b", "..", 5, None)]
    + [("engine_size", v) for v in (0, -1, True, 1.0, "5", None)]
    + [("lwjgl_url", v) for v in ("", 5, None, ["x"])]
    + [("lwjgl_version", v) for v in ("", 3.2, None)]
    + [("http_timeout", v) for v in (0, -1.5, True, "60", None)]
    + [("http_retries", v) for v in (-1, 101, 1.0, True, "3")]
    + [("seed", v) for v in (-1, 2**63, True, 0.0)]
    + [("warmup_steps", v) for v in (-1, 10**6 + 1, False, 2.0)]
    + [("steps", v) for v in (0, -3, True, 1.5)]
    + [("min_rate", v) for v in (0, -20, False, "20", None)]
    + [("retry_delay", v) for v in ("1", True, None)],
)
def test_readers_reject(tmp_path, key, value):
    """The readers' form f"{name} {value!r} is not …" (`retry_delay`: `number`'s own
    message for a value that is no number)."""
    assert error(tmp_path, **{key: value}).startswith(f"{key} {value!r} is not ")


@pytest.mark.parametrize(
    "key, value, text",
    [("engine_url", v, STRING) for v in ("", 5, None, ["x"])]
    + [("engine_sha256", v, HEX) for v in ("0" * 63, "0" * 65, "A" * 64, "g" * 64,
                                           SHA + "\n", "0x" + SHA[2:], 123, None)]
    + [("retry_delay", v, FINITE) for v in (-1, -0.5, math.inf, -math.inf)]
    + [("render_size", v, SIZE) for v in ([640], [640, 360, 1], [0, 360], [640, 0],
                                          [640.0, 360], [True, 360], "640x360", None,
                                          {"w": 640, "h": 360}, [])],
)  # fmt: skip
def test_stated_messages(tmp_path, key, value, text):
    assert error(tmp_path, **{key: value}) == f"{key} {value!r} {text}"


@pytest.mark.parametrize("value", ["1", True, None, [0]])
def test_retry_delay_non_number(tmp_path, value):
    """`number` first, with its own message: not the finite / ≥ 0 one."""
    message = error(tmp_path, retry_delay=value)
    assert message.startswith(f"retry_delay {value!r} is not ")
    assert message != f"retry_delay {value!r} {FINITE}"


def test_retry_delay_nan(tmp_path):
    assert error(tmp_path, retry_delay=math.nan) == f"retry_delay nan {FINITE}"


def sha_table(**changes) -> dict:
    table = {m: {"classes": SHA, "natives": SHA} for m in MODULES}
    return table | changes


SWAPPED = dict(sha_table())
SWAPPED = {MODULES[1]: SWAPPED.pop(MODULES[1]), **SWAPPED}


@pytest.mark.parametrize(
    "value, message",
    [
        (["x"], "lwjgl_sha256 ['x'] is not a mapping"),
        (None, "lwjgl_sha256 None is not a mapping"),
        (SHA, f"lwjgl_sha256 {SHA!r} is not a mapping"),
        ({m: v for m, v in sha_table().items() if m != "lwjgl-stb"},
         f"lwjgl_sha256 keys {[m for m in MODULES if m != 'lwjgl-stb']} are not {list(MODULES)}"),
        (SWAPPED, f"lwjgl_sha256 keys {list(SWAPPED)} are not {list(MODULES)}"),
        (sha_table(**{"lwjgl-vulkan": {"classes": SHA, "natives": SHA}}),
         f"lwjgl_sha256 keys {[*MODULES, 'lwjgl-vulkan']} are not {list(MODULES)}"),
        (sha_table(**{"lwjgl-glfw": SHA}), f"lwjgl_sha256[lwjgl-glfw] {SHA!r} is not a mapping"),
        (sha_table(lwjgl=None), "lwjgl_sha256[lwjgl] None is not a mapping"),
        (sha_table(**{"lwjgl-stb": {"natives": SHA, "classes": SHA}}),
         "lwjgl_sha256[lwjgl-stb] keys ['natives', 'classes'] are not ['classes', 'natives']"),
        (sha_table(**{"lwjgl-tinyfd": {"classes": SHA}}),
         "lwjgl_sha256[lwjgl-tinyfd] keys ['classes'] are not ['classes', 'natives']"),
        (sha_table(lwjgl={"classes": SHA, "natives": SHA, "sources": SHA}),
         "lwjgl_sha256[lwjgl] keys ['classes', 'natives', 'sources'] are not ['classes', 'natives']"),
        (sha_table(**{"lwjgl-openal": {"classes": SHA, "natives": SHA.upper()}}),
         f"lwjgl_sha256[lwjgl-openal][natives] {SHA.upper()!r} {HEX}"),
        (sha_table(**{"lwjgl-jemalloc": {"classes": 7, "natives": SHA}}),
         f"lwjgl_sha256[lwjgl-jemalloc][classes] 7 {HEX}"),
    ],
)  # fmt: skip
def test_lwjgl_sha256_messages(tmp_path, value, message):
    assert error(tmp_path, lwjgl_sha256=value) == message


@pytest.mark.parametrize(
    "key, value",
    [("http_retries", 0), ("http_retries", 100), ("seed", 0), ("seed", 2**63 - 1),
     ("warmup_steps", 0), ("warmup_steps", 10**6), ("retry_delay", 0), ("retry_delay", 3),
     ("retry_delay", 0.25), ("render_size", [1, 1]), ("engine_size", 1), ("steps", 1),
     ("http_timeout", 0.001), ("http_timeout", 60), ("min_rate", 0.001)],
)  # fmt: skip
def test_boundaries_accepted(tmp_path, key, value):
    got = getattr(load(tmp_path, **{key: value}), key)
    assert got == (tuple(value) if key == "render_size" else value)


@pytest.mark.parametrize("present", ["nothing", "directories"])
def test_use_engine_without_jar(root, clean_env, tmp_path, monkeypatch, present):
    """J missing (also with its directories present): the error; no variable set or
    changed."""
    cfg = load(tmp_path)
    if present == "directories":
        (root / "ms/sim" / JAR).parent.mkdir(parents=True)
    monkeypatch.setenv("MINESTUDIO_GPU_RENDER", "1")
    with pytest.raises(FileNotFoundError) as e:
        use_engine(cfg)
    assert (
        str(e.value)
        == f"engine missing: {root / 'ms/sim' / JAR}; run scripts/setup_env.sh"
    )
    assert "MINESTUDIO_DIR" not in os.environ
    assert os.environ["MINESTUDIO_GPU_RENDER"] == "1"


def test_use_engine_sets_variables(root, clean_env, tmp_path):
    cfg = load(tmp_path)
    (root / "ms/sim" / JAR).parent.mkdir(parents=True)
    (root / "ms/sim" / JAR).write_bytes(b"jar")
    assert use_engine(cfg) is None
    assert {v: os.environ[v] for v in VARS} == {
        "MINESTUDIO_DIR": str(root / "ms/sim"),
        "MINESTUDIO_GPU_RENDER": "0",
    }
