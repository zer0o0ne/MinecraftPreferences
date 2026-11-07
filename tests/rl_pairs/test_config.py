"""Configuration contract of specs/rl-preference-pairs.md (Interface)."""

import dataclasses

import pytest
import yaml
from runtime_configs import RL_PAIRS_STARTING as BASE

from mcpref.rl_pairs.config import RLPreferencePairsConfig, load_config


def text(**changes) -> str:
    values = {k: v for k, v in (BASE | changes).items() if v is not None}
    return yaml.safe_dump(values, sort_keys=False)


def load(tmp_path, raw: str):
    path = tmp_path / "config.yaml"
    path.write_text(raw)
    return load_config(path)


def with_raw(key, raw) -> str:
    return text(**{key: None}) + f"{key}: {raw}\n"


def test_starting_values_from_test_owned_config(tmp_path):
    cfg = load(tmp_path, text())
    assert dataclasses.astuple(cfg) == tuple(BASE.values())


def test_fields_order_and_frozen(tmp_path):
    cfg = load(tmp_path, text())
    assert [f.name for f in dataclasses.fields(RLPreferencePairsConfig)] == list(BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.capacity = 2


@pytest.mark.parametrize("key", list(BASE))
def test_missing_key(tmp_path, key):
    with pytest.raises(ValueError):
        load(tmp_path, text(**{key: None}))


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, text() + "unknown: 1\n")


@pytest.mark.parametrize(
    "key",
    [
        "sample_states",
        "temporal_per_state",
        "nearest_per_state",
        "branch",
        "beam_width",
        "beam_depth",
    ],
)
def test_zero_disables_nonnegative_counts(tmp_path, key):
    assert getattr(load(tmp_path, with_raw(key, "0")), key) == 0


@pytest.mark.parametrize("key", ["capacity", "items_per_shard", "history_per_shard"])
def test_positive_counts_accept_one(tmp_path, key):
    assert getattr(load(tmp_path, with_raw(key, "1")), key) == 1


BAD = [
    *((key, raw) for key in ["history_dir", "out_dir"]
      for raw in ['""', '"/absolute"', '"a/../b"', '".."', "5", "null", "true", '["x"]']),
    *((key, raw) for key in ["sample_states", "temporal_per_state", "nearest_per_state",
                             "branch", "beam_width", "beam_depth"]
      for raw in ["-1", "1.5", "true", '"1"', "null", ".inf"]),
    *((key, raw) for key in ["capacity", "items_per_shard", "history_per_shard"]
      for raw in ["0", "-1", "1.5", "false", '"1"', "null"]),
    *((key, raw) for key in ["base_threshold", "threshold_scale"]
      for raw in ["0", "0.0", "1", "1.0", "-0.1", "1.1", "true", '"0.5"',
                  "null", ".nan", ".inf"]),
    *(("temporal_penalty", raw) for raw in ["-0.1", "true", '"1"', "null", ".nan", ".inf"]),
]  # fmt: skip


@pytest.mark.parametrize("key,raw", BAD, ids=[f"{k}={r}" for k, r in BAD])
def test_violated_constraints(tmp_path, key, raw):
    with pytest.raises(ValueError):
        load(tmp_path, with_raw(key, raw))


def test_temporal_penalty_accepts_zero_and_integer(tmp_path):
    assert load(tmp_path, with_raw("temporal_penalty", "0")).temporal_penalty == 0
    assert load(tmp_path, with_raw("temporal_penalty", "2")).temporal_penalty == 2


def test_paths_and_numbers_kept_as_written(tmp_path):
    raw = text().replace("data/rl_preference_history", "data/./history")
    raw = raw.replace("temporal_penalty: 1.0", "temporal_penalty: 0")
    cfg = load(tmp_path, raw)
    assert cfg.history_dir == "data/./history"
    assert cfg.temporal_penalty == 0 and type(cfg.temporal_penalty) is int
