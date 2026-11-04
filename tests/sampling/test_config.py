"""SamplingConfig validation and starting values of specs/env-sampling.md."""

import dataclasses

import pytest
import yaml

from mcpref.sampling.config import SamplingConfig, load_config

FIELDS = [
    "pretrain", "env", "mcts", "perception_data", "out_dir", "steps", "seed",
    "safety_limit",
]  # fmt: skip
VALID = {
    "pretrain": "configs/pretrain.yaml",
    "env": "configs/env.yaml",
    "mcts": "configs/mcts.yaml",
    "perception_data": "configs/perception-data.yaml",
    "out_dir": "data/sampling",
    "steps": 2,
    "seed": 7,
    "safety_limit": 25.0,
}


def load(tmp_path, **changes):
    value = VALID | changes
    value = {key: item for key, item in value.items() if item is not None}
    path = tmp_path / "sampling.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return load_config(path)


def test_fields_order_values_and_frozen(tmp_path):
    cfg = load(tmp_path, pretrain="./configs//pretrain.yaml", safety_limit=0)
    assert [field.name for field in dataclasses.fields(SamplingConfig)] == FIELDS
    assert cfg.pretrain == "./configs//pretrain.yaml"
    assert (cfg.steps, cfg.seed, cfg.safety_limit) == (2, 7, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.steps = 3


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_boundaries(tmp_path, seed):
    assert load(tmp_path, seed=seed).seed == seed


def test_safety_budget_boundary(tmp_path):
    assert load(tmp_path, safety_limit=299.5).safety_limit == 299.5
    with pytest.raises(ValueError):
        load(tmp_path, safety_limit=300)


PATHS = ("pretrain", "env", "mcts", "perception_data", "out_dir")
BAD = {
    "missing": {"steps": None},
    "unknown": {"extra": 1},
    **{f"{name} empty": {name: ""} for name in PATHS},
    **{f"{name} absolute": {name: "/tmp/x"} for name in PATHS},
    **{f"{name} parent": {name: "a/../x"} for name in PATHS},
    **{f"{name} non-string": {name: 1} for name in PATHS},
    **{f"steps {value}": {"steps": value} for value in (0, -1, True, 1.0, "1")},
    **{f"seed {value}": {"seed": value} for value in (-1, 2**32, True, 1.0, "1")},
    **{
        f"safety {value}": {"safety_limit": value}
        for value in (-0.1, True, "1", float("nan"))
    },
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_invalid_config_raises_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_starting_values(repo):
    assert dataclasses.asdict(
        load_config(repo / "configs/env-sampling.yaml")
    ) == VALID | {
        "steps": 1000,
        "seed": 0,
    }
