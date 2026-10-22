"""`load_config` of specs/perception.md (Notation: Config; Interface: starting values): the
fields, their types, their constraints (`ValueError` at load)."""

import dataclasses

import pytest
import yaml

from mcpref.perception.config import PerceptionConfig, load_config

from .world import BASE, config, values

NAMES = set(BASE)
ALPHA = [float(2**k) for k in range(10) for _ in range(4)]


def test_starting_values(repo):
    """configs/perception.yaml as Interface starts it; `cursor` resolved against the root."""
    cfg = load_config(repo / "configs/perception.yaml")
    assert {f.name for f in dataclasses.fields(PerceptionConfig)} == NAMES
    got = dataclasses.asdict(cfg)
    cursor = got.pop("cursor")
    assert repo / cursor == repo / "data/cursor/mouse_cursor_white_16x16.png"
    assert got == {
        "model": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "seed": 0, "alpha": tuple(ALPHA), "head_layers": 1, "head_width": 1024,
        "c": 1.0, "beta": 1.0, "dual_init": 30.0, "dual_slack": 1.0e-3, "tau": 1.0,
        "min_pixels": 65536, "max_pixels": 16777216,
        "rescale_factor": 0.00392156862745098,
        "image_mean": (0.5, 0.5, 0.5), "image_std": (0.5, 0.5, 0.5), "decimals": 1,
        "meta": "{inventory}Hotbar: {hotbar}\nYaw: {yaw}\nPitch: {pitch}\n"
                "Position: {x} {y} {z}\n",
        "inventory_line": "Inventory: {items}\n", "item_line": "{quantity} {type}",
        "no_items": "empty", "marker": "<|im_end|>",
        "cursor_url": "https://raw.githubusercontent.com/openai/Video-Pre-Training/"
                      "095519fbd4ee0e9281d19f19601e45629de9ac3f/cursors/"
                      "mouse_cursor_white_16x16.png",
        "cursor_sha256": "8e4ff0fbbff52c1db89a4fe4ede8f3ecc5cdbc0b35d596b60a65307b55e4f805",
        "cursor_crop": 16, "mouse_height": 720, "http_timeout": 30, "http_retries": 5,
        "retry_delay": 10,
    }  # fmt: skip
    assert len(cfg.alpha) == 40


def test_reads_the_tiny_config(tmp_path):
    """Tuples of floats for alpha, image_mean, image_std; integers as int; frozen."""
    cfg = config(tmp_path)
    assert cfg.alpha == (1.0, 2.0, 0.5, 3.0) and type(cfg.alpha) is tuple
    for field in (cfg.alpha, cfg.image_mean, cfg.image_std):
        assert type(field) is tuple and all(type(x) is float for x in field)
    assert (cfg.image_mean, cfg.image_std) == ((0.4, 0.5, 0.6), (0.2, 0.3, 0.25))
    ints = (cfg.seed, cfg.head_layers, cfg.head_width, cfg.min_pixels, cfg.max_pixels,
            cfg.decimals, cfg.cursor_crop, cfg.mouse_height, cfg.http_retries)  # fmt: skip
    assert ints == (7, 1, 8, 1024, 1048576, 1, 16, 720, 2)
    assert all(type(x) is int for x in ints)
    assert (cfg.c, cfg.beta, cfg.dual_init, cfg.dual_slack, cfg.tau) == (
        1.5, 0.7, 30.0, 1e-3, 1.0
    )  # fmt: skip
    assert str(tmp_path / cfg.cursor) == values(tmp_path)["cursor"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


ACCEPTED = {  # boundaries inside the domain: field, value
    "seed 0": ("seed", 0),
    "seed 2^32 - 1": ("seed", 2**32 - 1),
    "head_layers 0": ("head_layers", 0),
    "tau 0": ("tau", 0),
    "decimals 0": ("decimals", 0),
    "http_retries 0": ("http_retries", 0),
    "min_pixels = max_pixels": ("min_pixels", 1048576),
    "image_mean negative": ("image_mean", [-1, 0, 2.5]),
    "alpha one entry": ("alpha", [0.25]),
    "c an integer": ("c", 2),
    "dual_slack tiny": ("dual_slack", 1.0e-300),
}


@pytest.mark.parametrize("name, value", ACCEPTED.values(), ids=ACCEPTED)
def test_accepted_values(tmp_path, name, value):
    got = getattr(config(tmp_path, **{name: value}), name)
    assert got == (tuple(value) if isinstance(value, list) else value)


POSITIVE_NUMBERS = ["c", "beta", "dual_init", "dual_slack", "rescale_factor",
                    "http_timeout", "retry_delay"]  # fmt: skip
POSITIVE_INTS = [
    "head_width",
    "min_pixels",
    "max_pixels",
    "cursor_crop",
    "mouse_height",
]
NONNEG_INTS = ["head_layers", "decimals", "http_retries"]
STRINGS = ["model", "revision", "no_items", "marker", "cursor_url"]
BAD = {
    **{f"{k} 0": {k: 0} for k in POSITIVE_NUMBERS},
    **{f"{k} negative": {k: -0.5} for k in POSITIVE_NUMBERS},
    **{f"{k} a string": {k: "1"} for k in POSITIVE_NUMBERS},
    **{f"{k} a boolean": {k: True} for k in POSITIVE_NUMBERS},
    "tau negative": {"tau": -1.0e-9},
    "tau a string": {"tau": "1"},
    "tau nan": {"tau": float("nan")},
    **{f"{k} 0": {k: 0} for k in POSITIVE_INTS},
    **{f"{k} a float": {k: 16.0} for k in POSITIVE_INTS},
    **{f"{k} a boolean": {k: True} for k in POSITIVE_INTS},
    **{f"{k} negative": {k: -1} for k in NONNEG_INTS},
    **{f"{k} a float": {k: 1.0} for k in NONNEG_INTS},
    **{f"{k} a boolean": {k: False} for k in NONNEG_INTS},
    "seed negative": {"seed": -1},
    "seed 2^32": {"seed": 2**32},
    "seed a boolean": {"seed": True},
    "seed a float": {"seed": 1.0},
    **{f"{k} empty": {k: ""} for k in STRINGS},
    **{f"{k} not a string": {k: 3} for k in STRINGS},
    "alpha empty": {"alpha": []},
    "alpha a scalar": {"alpha": 1.0},
    "alpha entry 0": {"alpha": [1.0, 0]},
    "alpha entry negative": {"alpha": [-1.0]},
    "alpha entry a string": {"alpha": ["1"]},
    "alpha entry a boolean": {"alpha": [True]},
    "min_pixels above max_pixels": {"min_pixels": 1048577},
    "image_mean two entries": {"image_mean": [0.5, 0.5]},
    "image_mean four entries": {"image_mean": [0.5, 0.5, 0.5, 0.5]},
    "image_mean a string entry": {"image_mean": [0.5, "0.5", 0.5]},
    "image_std two entries": {"image_std": [0.5, 0.5]},
    "image_std entry 0": {"image_std": [0.5, 0.0, 0.5]},
    "image_std entry negative": {"image_std": [0.5, 0.5, -0.5]},
    **{
        f"meta without {k}": {"meta": BASE["meta"].replace("{" + k + "}", "")}
        for k in ("inventory", "hotbar", "yaw", "pitch", "x", "y", "z")
    },
    "meta not a string": {"meta": ["{inventory}"]},
    "inventory_line without {items}": {"inventory_line": "Inv: \n"},
    "item_line without {type}": {"item_line": "{quantity}"},
    "item_line without {quantity}": {"item_line": "{type}"},
    "cursor not a string": {"cursor": 1},
    "cursor null": {"cursor": None},
    "cursor_sha256 uppercase": {"cursor_sha256": "A" * 64},
    "cursor_sha256 63 characters": {"cursor_sha256": "a" * 63},
    "cursor_sha256 65 characters": {"cursor_sha256": "a" * 65},
    "cursor_sha256 not hex": {"cursor_sha256": "g" * 64},
    "cursor_sha256 not a string": {"cursor_sha256": 1},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_violated_constraints_raise_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        config(tmp_path, **changes)


@pytest.mark.parametrize("name", sorted(NAMES))
def test_missing_key(tmp_path, name):
    path = tmp_path / "perception.yaml"
    doc = {k: v for k, v in values(tmp_path).items() if k != name}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        config(tmp_path, D=4)
