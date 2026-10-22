"""`load_config` of src/mcpref/policy/config.py against specs/policy.md (Notation: the config;
Interface: starting values): `PolicyConfig` frozen, fields in order; a missing or unknown key
and every violated constraint -> `ValueError`."""

import dataclasses

import pytest
import yaml

from mcpref.policy.config import PolicyConfig, load_config

VALID = {  # YAML lines of a valid config, keys in another order than Notation's
    "seed": "4294967295",
    "layers": "3",
    "history": "1",
    "width": "12",
    "heads": "3",
    "hidden": "5",
    "shift_hidden": "7",
    "rope_base": "10000",
    "norm_eps": "1.0e-6",
    "action_dropout": "0.3",
    "shift_weight": "2",
    "latency_ms": "0.5",
    "latency_fail_ms": "2",
    "latency_calls": "1",
}
FIELDS = ["history", "width", "heads", "hidden", "layers", "shift_hidden", "rope_base",
          "norm_eps", "action_dropout", "shift_weight", "seed", "latency_ms",
          "latency_fail_ms", "latency_calls"]  # fmt: skip


def load(tmp_path, **changes: str | None) -> PolicyConfig:
    """A config of the VALID lines with `changes` (None: the line left out)."""
    lines = {k: v for k, v in (VALID | changes).items() if v is not None}
    path = tmp_path / "c.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in lines.items()))
    return load_config(path)


def test_reads_every_field(tmp_path):
    """Fields in Notation's order; values as written; frozen."""
    cfg = load(tmp_path)
    assert [f.name for f in dataclasses.fields(PolicyConfig)] == FIELDS
    assert dataclasses.astuple(cfg) == (
        1, 12, 3, 5, 3, 7, 10000, 1e-6, 0.3, 2, 2**32 - 1, 0.5, 2, 1
    )  # fmt: skip
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.width = 4


ACCEPTED = {  # boundaries inside the domain: field, YAML, value
    "seed 0": ("seed", "0", 0),
    "one layer": ("layers", "1", 1),
    "d_h = 2": ("width", "6", 6),
    "rope_base float": ("rope_base", "0.5", 0.5),
    "norm_eps tiny": ("norm_eps", "1.0e-300", 1e-300),
    "action_dropout 0": ("action_dropout", "0", 0),
    "action_dropout 0.0": ("action_dropout", "0.0", 0.0),
    "action_dropout below 1": ("action_dropout", "0.999", 0.999),
    "shift_weight tiny": ("shift_weight", "1.0e-300", 1e-300),
    "shift_hidden 1": ("shift_hidden", "1", 1),
    "latency_fail_ms below latency_ms": ("latency_fail_ms", "0.25", 0.25),
}


@pytest.mark.parametrize("name, text, value", ACCEPTED.values(), ids=ACCEPTED)
def test_accepted_values(tmp_path, name, text, value):
    assert getattr(load(tmp_path, **{name: text}), name) == value


def test_width_equal_to_heads_times_two(tmp_path):
    cfg = load(tmp_path, width="2", heads="1")
    assert (cfg.width, cfg.heads) == (2, 1)


POSITIVE_INTS = ("history", "width", "heads", "hidden", "layers", "shift_hidden",
                 "latency_calls")  # fmt: skip
POSITIVES = ("rope_base", "norm_eps", "shift_weight", "latency_ms", "latency_fail_ms")
BAD = {
    "missing key": {"shift_hidden": None},
    "missing action_dropout": {"action_dropout": None},
    "unknown key": {"dropout": "0.0"},
    **{f"{k} 0": {k: "0"} for k in POSITIVE_INTS},
    **{f"{k} negative": {k: "-3"} for k in POSITIVE_INTS},
    **{f"{k} a boolean": {k: "true"} for k in POSITIVE_INTS},
    **{f"{k} a float": {k: "3.0"} for k in POSITIVE_INTS},
    **{f"{k} a string": {k: '"3"'} for k in POSITIVE_INTS},
    "width not divisible by heads": {"width": "10", "heads": "4"},
    "odd quotient": {"width": "9"},
    "quotient 1": {"width": "3"},
    **{f"{k} 0": {k: "0"} for k in POSITIVES},
    **{f"{k} negative": {k: "-0.5"} for k in POSITIVES},
    **{f"{k} a boolean": {k: "true"} for k in POSITIVES},
    **{f"{k} a string": {k: '"1.0"'} for k in POSITIVES},
    **{f"{k} nan": {k: ".nan"} for k in POSITIVES},
    "norm_eps 1e-6 (a string to PyYAML)": {"norm_eps": "1e-6"},
    "action_dropout 1": {"action_dropout": "1"},
    "action_dropout 1.0": {"action_dropout": "1.0"},
    "action_dropout negative": {"action_dropout": "-0.1"},
    "action_dropout a boolean": {"action_dropout": "false"},
    "action_dropout a string": {"action_dropout": '"0.3"'},
    "action_dropout nan": {"action_dropout": ".nan"},
    "action_dropout null": {"action_dropout": "null"},
    "seed 2^32": {"seed": "4294967296"},
    "seed negative": {"seed": "-1"},
    "seed a boolean": {"seed": "false"},
    "seed a float": {"seed": "0.0"},
    "seed a string": {"seed": '"0"'},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_violated_constraints_raise_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_starting_values(repo):
    """configs/policy.yaml as Interface starts it; `history` = `states` of
    configs/perception-data.yaml."""
    cfg = load_config(repo / "configs/policy.yaml")
    assert dataclasses.asdict(cfg) == {
        "history": 16, "width": 256, "heads": 4, "hidden": 768, "layers": 4,
        "shift_hidden": 512, "rope_base": 10000.0, "norm_eps": 1e-6, "action_dropout": 0.3,
        "shift_weight": 1.0, "seed": 0, "latency_ms": 3.0, "latency_fail_ms": 10.0,
        "latency_calls": 100,
    }  # fmt: skip
    data = yaml.safe_load((repo / "configs/perception-data.yaml").read_text())
    assert cfg.history == data["states"]
