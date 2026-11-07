"""MctsConfig and load_config against specs/mcts.md Notation and Starting config values."""

import dataclasses

import pytest

from mcpref.mcts.config import MctsConfig, load_config

VALID = {
    "latency_calls": "2",
    "depth": "3",
    "simulations": "4",
    "min_n_sampling": "2",
    "width": "5",
    "c_puct": "1.25",
    "discount": "0.99",
    "temperature": "2",
    "min_p": "0.5",
    "epsilon": "0.05",
    "latency_ms": "150",
    "latency_fail_ms": "300",
}
FIELDS = [
    "simulations", "min_n_sampling", "depth", "width", "latency_calls", "c_puct",
    "discount", "temperature", "min_p", "epsilon", "latency_ms", "latency_fail_ms",
]  # fmt: skip


def load(tmp_path, **changes):
    lines = {k: v for k, v in (VALID | changes).items() if v is not None}
    path = tmp_path / "mcts.yaml"
    path.write_text("".join(f"{key}: {value}\n" for key, value in lines.items()))
    return load_config(path)


def test_reads_fields_in_order_and_is_frozen(tmp_path):
    cfg = load(tmp_path)
    assert [field.name for field in dataclasses.fields(MctsConfig)] == FIELDS
    assert dataclasses.astuple(cfg) == (
        4, 2, 3, 5, 2, 1.25, 0.99, 2, 0.5, 0.05, 150, 300,
    )  # fmt: skip
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.width = 8


@pytest.mark.parametrize(
    "name, text, value",
    [
        ("min_n_sampling", "1", 1),
        ("min_n_sampling", "4", 4),
        ("c_puct", "0", 0),
        ("discount", "1", 1),
        ("temperature", "1.0e-300", 1e-300),
        ("min_p", "1.0e-300", 1e-300),
        ("epsilon", "1.0e-300", 1e-300),
        ("latency_fail_ms", "0.25", 0.25),
    ],
)
def test_accepts_domain_boundaries(tmp_path, name, text, value):
    assert getattr(load(tmp_path, **{name: text}), name) == value


POSITIVE_INTS = ("simulations", "min_n_sampling", "depth", "width", "latency_calls")
NUMBERS = ("c_puct", "discount")
POSITIVES = ("temperature", "min_p", "epsilon", "latency_ms", "latency_fail_ms")
BAD = {
    "missing key": {"depth": None},
    "unknown key": {"seed": "0"},
    **{f"{key} zero": {key: "0"} for key in POSITIVE_INTS},
    **{f"{key} negative": {key: "-1"} for key in POSITIVE_INTS},
    **{f"{key} boolean": {key: "true"} for key in POSITIVE_INTS},
    **{f"{key} float": {key: "1.0"} for key in POSITIVE_INTS},
    **{f"{key} string": {key: '"1"'} for key in POSITIVE_INTS},
    **{f"{key} negative": {key: "-0.5"} for key in NUMBERS},
    **{f"{key} boolean": {key: "false"} for key in NUMBERS},
    **{f"{key} string": {key: '"1"'} for key in NUMBERS},
    **{f"{key} nan": {key: ".nan"} for key in NUMBERS},
    **{f"{key} zero": {key: "0"} for key in POSITIVES},
    **{f"{key} negative": {key: "-0.5"} for key in POSITIVES},
    **{f"{key} boolean": {key: "true"} for key in POSITIVES},
    **{f"{key} string": {key: '"1"'} for key in POSITIVES},
    **{f"{key} nan": {key: ".nan"} for key in POSITIVES},
    "discount zero": {"discount": "0"},
    "discount above one": {"discount": "1.0001"},
    "minimum above simulations": {"min_n_sampling": "5"},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_invalid_config_raises_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_starting_values_from_test_owned_config(tmp_path):
    starting = {
        "simulations": "24", "min_n_sampling": "4", "depth": "8", "width": "8",
        "latency_calls": "10", "c_puct": "1.25", "discount": "0.99", "temperature": "1.0",
        "min_p": "0.5", "epsilon": "0.05", "latency_ms": "150.0",
        "latency_fail_ms": "300.0",
    }  # fmt: skip
    assert dataclasses.asdict(load(tmp_path, **starting)) == {
        "simulations": 24,
        "min_n_sampling": 4,
        "depth": 8,
        "width": 8,
        "latency_calls": 10,
        "c_puct": 1.25,
        "discount": 0.99,
        "temperature": 1.0,
        "min_p": 0.5,
        "epsilon": 0.05,
        "latency_ms": 150.0,
        "latency_fail_ms": 300.0,
    }
