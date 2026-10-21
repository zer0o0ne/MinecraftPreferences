"""The config of specs/acyclic.md (Notation): `AcyclicConfig` (frozen, fields in order) by
`load_config`; a missing or unknown key and every violated constraint -> `ValueError`;
values kept as written; the starting values of configs/acyclic.yaml (Interface)."""

import dataclasses

import pytest
import yaml

from mcpref.preferences.acyclic import AcyclicConfig, load_config

from .acyclic_world import START

REPO_CONFIG = "configs/acyclic.yaml"
BASE = {"votes_dir": "data/pair_votes", "out_dir": "data/pair_labels",
        "weights": START, "refusal_share": 0.5}  # fmt: skip


def load(tmp_path, text: str) -> AcyclicConfig:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return load_config(path)


def text(**changes) -> str:
    """BASE with changes (a value None removes the key), as YAML."""
    config = {k: v for k, v in {**BASE, **changes}.items() if v is not None}
    return yaml.safe_dump(config, sort_keys=False)


def with_raw(key: str, raw: str) -> str:
    """BASE with the YAML text `raw` as the value of key."""
    lines = text(**{key: None})
    return lines + f"{key}: {raw}\n"


def test_starting_values(repo):
    """configs/acyclic.yaml holds the starting values of Interface."""
    c = load_config(repo / REPO_CONFIG)
    assert dataclasses.astuple(c) == (
        "data/pair_votes", "data/pair_labels", START, 0.5
    )  # fmt: skip
    assert list(c.weights) == list(START)


def test_fields_in_order_and_frozen(tmp_path):
    c = load(tmp_path, text())
    assert [f.name for f in dataclasses.fields(AcyclicConfig)] == list(BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.refusal_share = 0.1


def test_values_kept_as_written(tmp_path):
    """Paths, the weights' order and numbers, refusal_share as written (no conversion)."""
    raw = (
        'votes_dir: "data/./votes"\nout_dir: "x"\n'
        "weights:\n  Z: 1\n  A: 0.30\n  R: null\n  M: 0\nrefusal_share: 0\n"
    )
    c = load(tmp_path, raw)
    assert (c.votes_dir, c.out_dir) == ("data/./votes", "x")
    assert list(c.weights.items()) == [("Z", 1), ("A", 0.3), ("R", None), ("M", 0)]
    assert [type(v) for v in c.weights.values()] == [int, float, type(None), int]
    assert c.refusal_share == 0 and type(c.refusal_share) is int
    assert type(load(tmp_path, text()).refusal_share) is float


@pytest.mark.parametrize(
    "raw",
    ["0", "0.0", "0.999", "1.0e-1"],
)
def test_refusal_share_accepted(tmp_path, raw):
    """0 <= tau < 1, a YAML integer or float."""
    assert load(
        tmp_path, with_raw("refusal_share", raw)
    ).refusal_share == yaml.safe_load(raw)


@pytest.mark.parametrize(
    "raw",
    ["{R: null}", "{X: 0, R: null, Y: 1}", "{X: 1.0, R: null}", "{R: null, X: 0.5}"],
)
def test_weights_accepted(tmp_path, raw):
    """Non-empty, values in [0, 1] or null, exactly one null (the order free)."""
    assert load(tmp_path, with_raw("weights", raw)).weights == yaml.safe_load(raw)


@pytest.mark.parametrize("key", list(BASE))
def test_missing_key(tmp_path, key):
    with pytest.raises(ValueError):
        load(tmp_path, text(**{key: None}))


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, text() + "extra: 1\n")


BAD = [
    *(("votes_dir", raw) for raw in ['""', '"/abs/votes"', '"a/../b"', '".."', "5", "null",
                                      '["a"]']),
    *(("out_dir", raw) for raw in ['""', '"/abs"', '"x/.."', "true"]),
    *(("weights", raw) for raw in [
        "{}", "[0.5, null]", '"x"', "null", "{1: 0.5, R: null}", "{true: 0.5, R: null}",
        "{X: true, R: null}", '{X: "0.5", R: null}', "{X: -0.1, R: null}",
        "{X: 1.1, R: null}", "{X: .nan, R: null}", "{X: .inf, R: null}", "{X: 0.5}",
        "{X: null, Y: null}", "{X: [0.5], R: null}", "{X: 1.0e+1, R: null}"]),
    *(("refusal_share", raw) for raw in ["true", "false", '"0.5"', "-0.1", "1", "1.0",
                                          "1.5", ".nan", ".inf", "-.inf", "null", "[0.5]"]),
]  # fmt: skip


@pytest.mark.parametrize("key, raw", BAD, ids=[f"{k}={r}" for k, r in BAD])
def test_violated_constraint(tmp_path, key, raw):
    with pytest.raises(ValueError):
        load(tmp_path, with_raw(key, raw))


HUGE = "1" + "0" * 400  # a YAML integer beyond any float
HUGE_CASES = {
    "weight": ("weights", f"{{X: {HUGE}, R: null}}"),
    "negative weight": ("weights", f"{{X: -{HUGE}, R: null}}"),
    "refusal_share": ("refusal_share", HUGE),
    "negative refusal_share": ("refusal_share", f"-{HUGE}"),
}


@pytest.mark.parametrize("key, raw", HUGE_CASES.values(), ids=HUGE_CASES)
def test_huge_integer_is_a_value_error(tmp_path, key, raw):
    """A YAML integer outside [0, 1] (weights) or [0, 1) (refusal_share), however large:
    a ValueError at load (no OverflowError)."""
    with pytest.raises(ValueError):
        load(tmp_path, with_raw(key, raw))
