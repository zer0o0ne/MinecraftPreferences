"""The config loader of specs/vlm-labels.md (Notation, Config; Interface: starting values):
`load_config` → frozen `VlmLabelsConfig`, a missing or unknown key and every violated
constraint → `ValueError` at load."""

import dataclasses

import pytest

from mcpref.preferences.labels import VlmLabelsConfig, load_config

from .config_files import FIELDS, STARTING, TINY, load, write_config

PATHS = ("pool_dir", "vlm", "out_dir")
TEXTS = FIELDS[3:12]
COUNTS = ("frames", "stride", "samples")


def test_starting_values(repo, tmp_path, monkeypatch):
    """configs/vlm-labels.yaml holds the starting values of Interface; fields in the order
    of Notation; paths kept as written; answers a list of dicts; frozen."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(repo / "configs/vlm-labels.yaml")
    assert isinstance(cfg, VlmLabelsConfig)
    assert tuple(f.name for f in dataclasses.fields(cfg)) == FIELDS
    assert dataclasses.asdict(cfg) == STARTING
    assert all(type(getattr(cfg, k)) is str for k in PATHS + TEXTS)
    assert type(cfg.answers) is list and all(type(a) is dict for a in cfg.answers)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


def test_keys_in_any_order(tmp_path):
    """The tiny config, keys in another order than Notation's, loads as written."""
    cfg = load(tmp_path)
    assert {f: getattr(cfg, f) for f in FIELDS} == TINY


@pytest.mark.parametrize("key", FIELDS)
def test_missing_key(tmp_path, key):
    mapping = {k: v for k, v in TINY.items() if k != key}
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path / "c.yaml", mapping))


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, extra=1)


@pytest.mark.parametrize("key", PATHS)
@pytest.mark.parametrize("value", ["", "/data/x", "a/../b", "..", 5, None, ["a"]])
def test_bad_path(tmp_path, key, value):
    """Paths: non-empty strings, relative, without a `..` component."""
    with pytest.raises(ValueError):
        load(tmp_path, **{key: value})


@pytest.mark.parametrize("key", TEXTS)
@pytest.mark.parametrize("value", [5, None, ["x"], {"a": "b"}])
def test_text_not_a_string(tmp_path, key, value):
    with pytest.raises(ValueError):
        load(tmp_path, **{key: value})


@pytest.mark.parametrize(
    "key, value",
    [
        ("question", "Pick: {answer}"),
        ("answer_line", "{description}"),
        ("answer_line", "{token}"),
        ("inventory_line", "Inv: {item}"),
        ("item_line", "{quantity}"),
        ("item_line", "{type}"),
    ],
)
def test_text_without_its_placeholder(tmp_path, key, value):
    """question holds `{answers}`, answer_line `{token}` and `{description}`,
    inventory_line `{items}`, item_line `{type}` and `{quantity}`."""
    with pytest.raises(ValueError):
        load(tmp_path, **{key: value})


@pytest.mark.parametrize(
    "value",
    ["Answer: `", "**Answer:** ", 'Answer: "', "Answer: '", "`", "*", '"', "'",
     "Answer:\n", "\n", "A\nB"],
)  # fmt: skip
def test_bad_prefix(tmp_path, value):
    """prefix holds none of `` ` ``, `*`, `"`, `'` and no newline ((2) removes or splits
    them)."""
    with pytest.raises(ValueError):
        load(tmp_path, prefix=value)


@pytest.mark.parametrize(
    "value",
    ["Answer: ", "", "  ", "Verdict - ", "Réponse : ", "A.", "{token}", "\t", "\r"],
)
def test_prefix_allowed(tmp_path, value):
    """Any other string, kept as written."""
    assert load(tmp_path, prefix=value).prefix == value


def answers(*tokens, **extra) -> list:
    return [{"token": t, "description": f"d{i}", **extra} for i, t in enumerate(tokens)]


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"token": "A", "description": "a"},
        ["BETTER"],
        answers("A", note="x"),
        [{"token": "A"}],
        [{"description": "a"}],
        [{"token": 5, "description": "a"}],
        [{"token": "A", "description": None}],
        answers("A", "B", "A"),
        answers(""),
        *(answers(f"A{c}B") for c in (" ", "\t", "\n", "`", "*", '"', "'", ".")),
        answers("A", "invalid"),
        answers("no_answer"),
    ],
)
def test_bad_answers(tmp_path, value):
    """A non-empty list of mappings with exactly `token` and `description`, both strings;
    tokens distinct, non-empty, without whitespace, `` ` ``, `*`, `"`, `'`, `.`, and
    neither `invalid` nor `no_answer`."""
    with pytest.raises(ValueError):
        load(tmp_path, answers=value)


def test_tokens_allowed(tmp_path):
    """Tokens the constraints allow: `{` `}` `-` `:` and other letter cases of the labels."""
    value = answers("{x}", "A-B:c", "INVALID", "No_Answer", "invalid_x")
    assert load(tmp_path, answers=value).answers == value


@pytest.mark.parametrize("key", COUNTS)
@pytest.mark.parametrize("value", [0, -1, 1.0, "2", True, False, None])
def test_bad_count(tmp_path, key, value):
    """frames, stride, samples: YAML integers, not booleans, >= 1."""
    with pytest.raises(ValueError):
        load(tmp_path, **{key: value})


def test_counts_of_one(tmp_path):
    cfg = load(tmp_path, frames=1, stride=1, samples=1)
    assert (cfg.frames, cfg.stride, cfg.samples) == (1, 1, 1)


@pytest.mark.parametrize("value", [-1, 2**32, 0.0, "0", True, False, None])
def test_bad_seed(tmp_path, value):
    """seed: a YAML integer, not a boolean, 0 <= seed < 2^32."""
    with pytest.raises(ValueError):
        load(tmp_path, seed=value)


@pytest.mark.parametrize("value", [0, 2**32 - 1])
def test_seed_bounds(tmp_path, value):
    assert load(tmp_path, seed=value).seed == value
