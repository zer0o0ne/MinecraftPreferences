"""`load_config` of specs/preference-pool.md (Notation, Interface): the fields in order, a
missing or unknown key, YAML integers, the ranges, tick_slack and cycles (numbers, not
booleans, finite and representable as a float, kept as written; a huge integer a
ValueError),
resetting_stats, the relative paths kept as written, and the starting values of
configs/preference-pool.yaml."""

import dataclasses
import math
import sys

import pytest
import yaml

from mcpref.pool.run import PreferencePoolConfig, load_config

from .pool_site import POOL, write_config

VALID = POOL | {"perception": "cfg/perception-data.yaml", "out_dir": "out"}
INTEGERS = ("head_bytes", "head_workers", "entries_per_shard", "tick_ms", "episodes",
            "per_episode", "n_clips", "partners", "seed", "workers")  # fmt: skip
STARTING = {
    "perception": "configs/perception-data.yaml", "head_bytes": 32768, "head_workers": 16,
    "entries_per_shard": 1000, "tick_ms": 50, "tick_slack": 0.03,
    "resetting_stats": ["minecraft.custom:minecraft.time_since_rest",
                        "minecraft.custom:minecraft.time_since_death"],
    "episodes": 5, "per_episode": 4, "n_clips": 10000, "partners": 2, "cycles": 0.5,
    "seed": 0, "workers": 4, "out_dir": "data/pair_clips",
}  # fmt: skip


def load(tmp_path, **changes) -> PreferencePoolConfig:
    return load_config(write_config(tmp_path / "pool.yaml", **changes))


def values(cfg: PreferencePoolConfig) -> dict:
    """The fields; resetting_stats as a list (its sequence type is free)."""
    out = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    return out | {"resetting_stats": list(out["resetting_stats"])}


def test_reads_every_field_in_order(tmp_path):
    cfg = load(tmp_path)
    assert [f.name for f in dataclasses.fields(PreferencePoolConfig)] == list(POOL)
    assert values(cfg) == VALID
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


@pytest.mark.parametrize("name", list(POOL))
def test_a_missing_key_is_rejected(tmp_path, name):
    path = tmp_path / "pool.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in VALID.items() if k != name}))
    with pytest.raises(ValueError):
        load_config(path)


def test_an_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, extra=1)


@pytest.mark.parametrize("name", INTEGERS)
@pytest.mark.parametrize("value", [True, False, 3.0, "3", None])
def test_integers_must_be_yaml_integers(tmp_path, name, value):
    with pytest.raises(ValueError):
        load(tmp_path, **{name: value})


@pytest.mark.parametrize(
    "name, value",
    [("head_bytes", 0), ("head_workers", 0), ("head_workers", 21), ("entries_per_shard", 0),
     ("tick_ms", 0), ("episodes", 0), ("per_episode", 0), ("n_clips", 1), ("n_clips", 0),
     ("partners", 0), ("seed", -1), ("seed", 2**32), ("workers", 0)],
)  # fmt: skip
def test_out_of_range_integers_are_rejected(tmp_path, name, value):
    with pytest.raises(ValueError):
        load(tmp_path, **{name: value})


@pytest.mark.parametrize(
    "name, value",
    [("head_bytes", 1), ("head_workers", 1), ("head_workers", 20), ("entries_per_shard", 1),
     ("tick_ms", 1), ("episodes", 1), ("per_episode", 1), ("n_clips", 2), ("partners", 1),
     ("seed", 0), ("seed", 2**32 - 1), ("workers", 1)],
)  # fmt: skip
def test_boundary_integers_are_accepted(tmp_path, name, value):
    assert getattr(load(tmp_path, **{name: value}), name) == value


@pytest.mark.parametrize("value", [True, False, "0.03", None, [0.03], math.inf, math.nan, -0.01, 1, 1.0, 1.5])  # fmt: skip
def test_tick_slack_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        load(tmp_path, tick_slack=value)


@pytest.mark.parametrize("value", [0, 0.0, 0.03, 0.7, 0.999])
def test_tick_slack_kept_as_written(tmp_path, value):
    got = load(tmp_path, tick_slack=value).tick_slack
    assert got == value and type(got) is type(value)


@pytest.mark.parametrize("value", [True, False, "0.5", None, {"a": 1}, math.inf, -math.inf, math.nan, -1, -0.5])  # fmt: skip
def test_cycles_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        load(tmp_path, cycles=value)


@pytest.mark.parametrize("value", [0, 0.0, 0.07, 3, 2.25])
def test_cycles_kept_as_written(tmp_path, value):
    got = load(tmp_path, cycles=value).cycles
    assert got == value and type(got) is type(value)


HUGE = 10**400  # a YAML integer beyond any float


@pytest.mark.parametrize(
    "name, value",
    [("tick_slack", HUGE), ("tick_slack", -HUGE), ("cycles", HUGE), ("cycles", -HUGE),
     ("seed", HUGE), ("head_workers", HUGE)],
    ids=["tick_slack", "-tick_slack", "cycles", "-cycles", "seed", "head_workers"],
)  # fmt: skip
def test_huge_integers_are_value_errors(tmp_path, name, value):
    """Notation: a violated constraint (out of range, or for cycles an integer not
    representable as a float) is a ValueError at load (no OverflowError)."""
    with pytest.raises(ValueError):
        load(tmp_path, **{name: value})


@pytest.mark.parametrize("value", [2**1024, 2**1024 + 1, 3 * 2**1023])
def test_cycles_integer_beyond_the_largest_float_is_rejected(tmp_path, value):
    """Notation: cycles representable as a float; an integer >= 2**1024 is not."""
    with pytest.raises(ValueError):
        load(tmp_path, cycles=value)


@pytest.mark.parametrize("value", [2**1023, int(sys.float_info.max)])
def test_cycles_representable_integer_kept_as_written(tmp_path, value):
    """Notation: an integer equal to a finite float is accepted, kept as the integer."""
    got = load(tmp_path, cycles=value).cycles
    assert got == value and type(got) is int


@pytest.mark.parametrize("value", ["a", {"a": 1}, None, ["a", 1], ["a", None], [["a"]], 3])  # fmt: skip
def test_resetting_stats_must_be_a_list_of_strings(tmp_path, value):
    with pytest.raises(ValueError):
        load(tmp_path, resetting_stats=value)


@pytest.mark.parametrize("value", [[], ["a"], ["a", "b c", "é"]])
def test_resetting_stats_accepted(tmp_path, value):
    assert list(load(tmp_path, resetting_stats=value).resetting_stats) == value


@pytest.mark.parametrize("name", ["perception", "out_dir"])
@pytest.mark.parametrize("value", ["", "/abs/path", "a/../b", "..", "../x", 1, None, ["out"]])  # fmt: skip
def test_paths_must_be_relative_without_dotdot(tmp_path, name, value):
    with pytest.raises(ValueError):
        load(tmp_path, **{name: value})


@pytest.mark.parametrize("name", ["perception", "out_dir"])
@pytest.mark.parametrize("value", ["x", "data/pair_clips/", "./cfg/p.yaml", "a..b/c"])
def test_paths_kept_as_written(tmp_path, name, value):
    assert getattr(load(tmp_path, **{name: value}), name) == value


def test_approved_config_has_the_starting_values(repo):
    cfg = load_config(repo / "configs/preference-pool.yaml")
    assert values(cfg) == STARTING
    assert type(cfg.tick_slack) is float and type(cfg.cycles) is float
