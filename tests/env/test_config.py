"""EnvConfig and load_config of specs/env.md Interface: order, paths, numbers and delta."""

import dataclasses
import inspect
import math
from pathlib import Path

import pytest
import yaml

from mcpref.env.config import EnvConfig, load_config

FIELDS = {
    "engine": "configs/engine.yaml",
    "classifier": "configs/gui-classifier.yaml",
    "action_mask": "configs/action-mask.yaml",
    "clips": "configs/clips.yaml",
    "cursor_scale": 6.666666666666667,
}


def write(tmp_path, **changes):
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(FIELDS | changes, sort_keys=False))
    return path


def test_load_config_public_signature():
    assert list(inspect.signature(load_config).parameters) == ["path"]
    hints = inspect.get_annotations(load_config, eval_str=True)
    assert hints["path"] is Path and hints["return"] is EnvConfig


def test_starting_values_and_public_record(tmp_path):
    cfg = load_config(write(tmp_path))
    assert isinstance(cfg, EnvConfig)
    assert [f.name for f in dataclasses.fields(cfg)] == list(FIELDS)
    assert cfg == EnvConfig(**FIELDS)
    assert all(getattr(cfg, k) == v for k, v in FIELDS.items())
    assert type(cfg.cursor_scale) is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cursor_scale = 1.0


def test_paths_kept_as_written(env_root, tmp_path):
    fields = FIELDS | {
        "engine": "e.yaml",
        "classifier": "c.yaml",
        "action_mask": "m.yaml",
        "clips": "configs/clips.yaml",
        "cursor_scale": 2,
    }
    cfg = load_config(write(tmp_path, **fields))
    assert tuple(getattr(cfg, k) for k in tuple(FIELDS)[:4]) == (
        "e.yaml",
        "c.yaml",
        "m.yaml",
        "configs/clips.yaml",
    )
    assert cfg.cursor_scale == 2.0 and type(cfg.cursor_scale) is float


@pytest.mark.parametrize("key", FIELDS)
def test_missing_field(tmp_path, key):
    data = {k: v for k, v in FIELDS.items() if k != key}
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_field(tmp_path):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, unknown=1))


@pytest.mark.parametrize("key", tuple(FIELDS)[:4])
@pytest.mark.parametrize(
    "value", ["", "/absolute/x.yaml", "../x.yaml", "a/../x.yaml", 3, None]
)
def test_path_constraints(tmp_path, key, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, **{key: value}))


@pytest.mark.parametrize(
    "value", [0, -1, -0.5, -math.inf, math.nan, True, False, "6.5", None, []]
)
def test_cursor_scale_must_be_a_number_above_zero(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, cursor_scale=value))


def test_positive_infinity_is_a_number_above_zero(tmp_path):
    assert load_config(write(tmp_path, cursor_scale=math.inf)).cursor_scale == math.inf


def test_delta_below_five_has_exact_error(env_root, tmp_path):
    (env_root / "configs/clips.yaml").write_text(
        "delta: 4\nactions: configs/vpt-actions.yaml\n"
    )
    with pytest.raises(ValueError) as error:
        load_config(write(tmp_path))
    assert str(error.value) == "delta must be >= 5"


def test_delta_five_is_accepted(env_root, tmp_path):
    (env_root / "configs/clips.yaml").write_text(
        "delta: 5\nactions: configs/vpt-actions.yaml\n"
    )
    assert load_config(write(tmp_path)) == EnvConfig(**FIELDS)
