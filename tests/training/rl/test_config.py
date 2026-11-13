"""RLCycleConfig field order, validation, and approved starting values."""

import dataclasses

import pytest
import torch
import yaml

from mcpref.training.rl.config import RLCycleConfig, load_config
from training.rl.learning.helpers import BASE, GROUPS

FIELDS = list(BASE)


MISSING = object()


def load(tmp_path, **changes):
    values = {
        key: value for key, value in (BASE | changes).items() if value is not MISSING
    }
    path = tmp_path / "rl.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return load_config(path)


def test_field_order_types_and_frozen(tmp_path):
    cfg = load(tmp_path, seed=2**32 - 1, betas=[0, 0.5], lr_floor=0)
    assert [field.name for field in dataclasses.fields(RLCycleConfig)] == FIELDS
    assert cfg.seed == 2**32 - 1 and type(cfg.seed) is int
    assert cfg.betas == (0.0, 0.5) and all(type(x) is float for x in cfg.betas)
    assert cfg.lr_floor == 0
    assert cfg.gradient_diagnostics is True
    assert torch.device(cfg.device) == torch.device("cpu")
    assert list(cfg.groups) == list(GROUPS)
    assert all(
        type(x) is float for group in cfg.groups.values() for x in group.values()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 2


@pytest.mark.parametrize("key", FIELDS)
def test_missing_key(tmp_path, key):
    with pytest.raises(ValueError):
        load(tmp_path, **{key: MISSING})


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, unknown=1)


PATH_BAD = ["", "/absolute", "a/../b", "..", 0, None, True, ["x"]]
POSITIVE = ["epochs", "batch", "micro_batch", "items_per_shard", "checkpoint_steps",
            "keep_checkpoints"]  # fmt: skip
WEIGHTS = ["w_ppo", "w_delta", "w_metra", "w_P", "adam_eps", "grad_clip"]
BAD = {
    **{f"{key} {raw}": {key: raw} for key in FIELDS[:4] for raw in PATH_BAD},
    "seed negative": {"seed": -1},
    "seed too large": {"seed": 2**32},
    "seed bool": {"seed": True},
    "seed float": {"seed": 1.0},
    "seed string": {"seed": "1"},
    "retrain integer": {"retrain": 1},
    "retrain string": {"retrain": "true"},
    "gradient diagnostics integer": {"gradient_diagnostics": 1},
    "gradient diagnostics string": {"gradient_diagnostics": "false"},
    **{
        f"{key} {raw}": {key: raw}
        for key in POSITIVE
        for raw in [0, -1, 1.5, True, "1"]
    },
    **{
        f"{key} {raw}": {key: raw}
        for key in WEIGHTS
        for raw in [0, -0.5, True, "1", ".nan"]
    },
    "warmup negative": {"warmup_steps": -1},
    "warmup bool": {"warmup_steps": False},
    "warmup float": {"warmup_steps": 1.5},
    "floor negative": {"lr_floor": -0.1},
    "floor above one": {"lr_floor": 1.1},
    "floor bool": {"lr_floor": True},
    "one beta": {"betas": [0.9]},
    "three betas": {"betas": [0.8, 0.9, 0.99]},
    "beta one": {"betas": [0.9, 1.0]},
    "beta negative": {"betas": [-0.1, 0.9]},
    "beta bool": {"betas": [False, 0.9]},
    "groups list": {"groups": []},
    "groups null": {"groups": None},
    "group missing rate": {"groups": {"pi": {"weight_decay": 0.0}}},
    "group missing decay": {"groups": {"pi": {"lr": 1.0}}},
    "group unknown option": {
        "groups": {"pi": {"lr": 1.0, "weight_decay": 0.0, "eps": 1}}
    },
    "group zero rate": {"groups": {"pi": {"lr": 0.0, "weight_decay": 0.0}}},
    "group negative decay": {"groups": {"pi": {"lr": 1.0, "weight_decay": -0.1}}},
    "bad device": {"device": "gpu"},
    "device not string": {"device": 0},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_invalid_values(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_valid_boundaries(tmp_path):
    cfg = load(tmp_path, seed=0, warmup_steps=0, lr_floor=1, betas=[0, 0],
               **{key: 1 for key in POSITIVE})  # fmt: skip
    assert cfg.lr_floor == 1 and cfg.betas == (0.0, 0.0)
    assert all(getattr(cfg, key) == 1 for key in POSITIVE)


def test_starting_values_and_group_order(tmp_path):
    starting = {
        "batch": 32, "items_per_shard": 256, "checkpoint_steps": 100,
        "warmup_steps": 100, "device": "cuda",
    }  # fmt: skip
    cfg = load(tmp_path, **starting)
    expected = BASE | starting
    assert dataclasses.asdict(cfg) | {"device": "cuda"} == expected | {
        "betas": (0.9, 0.999)
    }
    assert list(cfg.groups) == list(GROUPS)
    assert "phi_dual" not in cfg.groups and cfg.groups["metra_dual"] == {
        "lr": 1.0e-4,
        "weight_decay": 0.0,
    }
