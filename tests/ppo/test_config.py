"""PpoConfig and load_config against specs/ppo.md Interface."""

import dataclasses

import pytest

from mcpref.ppo.config import PpoConfig, load_config

VALID = {
    "horizon": "64",
    "clip": "0.2",
    "skill_eps": "1.0e-8",
    "dual_init": "30.0",
    "dual_slack": "1.0e-3",
}
POSITIVE = ("skill_eps", "dual_init", "dual_slack")


def load(tmp_path, **changes):
    lines = {k: v for k, v in (VALID | changes).items() if v is not None}
    path = tmp_path / "ppo.yaml"
    path.write_text("".join(f"{key}: {value}\n" for key, value in lines.items()))
    return load_config(path)


def test_reads_fields_in_order_and_is_frozen(tmp_path):
    cfg = load(tmp_path)
    assert [field.name for field in dataclasses.fields(PpoConfig)] == [
        "horizon",
        "clip",
        "skill_eps",
        "dual_init",
        "dual_slack",
    ]
    assert dataclasses.astuple(cfg) == (64, 0.2, 1.0e-8, 30.0, 1.0e-3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.horizon = 1


@pytest.mark.parametrize(
    "name, text, value",
    [
        ("horizon", "1", 1),
        ("horizon", "128", 128),
        ("clip", "1.0e-300", 1e-300),
        ("clip", "0.999", 0.999),
        *((name, "1.0e-300", 1e-300) for name in POSITIVE),
    ],
)
def test_accepts_domain_boundaries(tmp_path, name, text, value):
    assert getattr(load(tmp_path, **{name: text}), name) == value


BAD = {
    **{f"missing {name}": {name: None} for name in VALID},
    "unknown key": {"discount": "0.99"},
    "horizon zero": {"horizon": "0"},
    "horizon negative": {"horizon": "-1"},
    "horizon boolean": {"horizon": "true"},
    "horizon float": {"horizon": "1.0"},
    "horizon string": {"horizon": '"1"'},
    "clip zero": {"clip": "0"},
    "clip negative": {"clip": "-0.1"},
    "clip one": {"clip": "1"},
    "clip above one": {"clip": "1.1"},
    "clip boolean": {"clip": "false"},
    "clip string": {"clip": '"0.2"'},
    "clip nan": {"clip": ".nan"},
    "clip infinity": {"clip": ".inf"},
    "clip null": {"clip": "null"},
    **{f"{name} zero": {name: "0"} for name in POSITIVE},
    **{f"{name} negative": {name: "-0.1"} for name in POSITIVE},
    **{f"{name} boolean": {name: "true"} for name in POSITIVE},
    **{f"{name} string": {name: '"1"'} for name in POSITIVE},
    **{f"{name} nan": {name: ".nan"} for name in POSITIVE},
    **{f"{name} null": {name: "null"} for name in POSITIVE},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_invalid_config_raises_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_starting_values(repo):
    assert dataclasses.asdict(load_config(repo / "configs/ppo.yaml")) == {
        "horizon": 64,
        "clip": 0.2,
        "skill_eps": 1.0e-8,
        "dual_init": 30.0,
        "dual_slack": 1.0e-3,
    }
