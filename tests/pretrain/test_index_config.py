"""`load_config` of specs/train-index.md (Notation: Config; Interface: starting values):
the fields, their constraints (`ValueError` at load), `validation` kept as written."""

import dataclasses

import pytest

from mcpref.pretrain.index import TrainIndexConfig, load_config

NAMES = {"clips_dir", "pair_clips_dir", "pair_labels_dir", "out_dir", "validation",
         "seed", "items_per_shard"}  # fmt: skip
VALID = {  # YAML lines of a valid config
    "clips_dir": "c/clips",
    "pair_clips_dir": "./q",
    "pair_labels_dir": "l/x/",
    "out_dir": "o",
    "validation": '"1e-2"',
    "seed": "4294967295",
    "items_per_shard": "1",
}


def load(tmp_path, **changes: str | None) -> TrainIndexConfig:
    """A config of the VALID lines with `changes` (None: the line left out)."""
    lines = {k: v for k, v in (VALID | changes).items() if v is not None}
    path = tmp_path / "c.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in lines.items()))
    return load_config(path)


def test_reads_every_field(tmp_path):
    """The strings as written (validation too), the integers; frozen."""
    cfg = load(tmp_path)
    assert {f.name for f in dataclasses.fields(TrainIndexConfig)} == NAMES
    assert (cfg.clips_dir, cfg.pair_clips_dir, cfg.pair_labels_dir, cfg.out_dir) == (
        "c/clips",
        "./q",
        "l/x/",
        "o",
    )
    assert cfg.validation == "1e-2" and type(cfg.validation) is str
    assert (cfg.seed, cfg.items_per_shard) == (2**32 - 1, 1)
    assert type(cfg.seed) is int and type(cfg.items_per_shard) is int
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


def test_without_pairs(tmp_path):
    """Both pair dirs null: no pairs."""
    cfg = load(tmp_path, pair_clips_dir="null", pair_labels_dir="null")
    assert cfg.pair_clips_dir is None and cfg.pair_labels_dir is None


@pytest.mark.parametrize("v", ['"0"', '"0.05"', '"0.999"', "' 0.5'"])
def test_validation_accepted(tmp_path, v):
    """Strings float() reads to 0 <= v < 1, kept as written (a leading space too)."""
    assert load(tmp_path, validation=v).validation == v[1:-1]


BAD = {
    "missing key": {"seed": None},
    "unknown key": {"extra": "1"},
    "clips_dir null": {"clips_dir": "null"},
    "out_dir null": {"out_dir": "null"},
    "clips_dir empty": {"clips_dir": '""'},
    "out_dir absolute": {"out_dir": "/abs/o"},
    "clips_dir with ..": {"clips_dir": "c/../d"},
    "pair_clips_dir ..": {"pair_clips_dir": ".."},
    "pair_labels_dir ../x": {"pair_labels_dir": "../x"},
    "out_dir not a string": {"out_dir": "3"},
    "clips_dir a list": {"clips_dir": "[c]"},
    "pair_clips_dir alone null": {"pair_clips_dir": "null"},
    "pair_labels_dir alone null": {"pair_labels_dir": "null"},
    "pair_labels_dir empty": {"pair_labels_dir": '""'},
    "validation a float": {"validation": "0.05"},
    "validation an integer": {"validation": "0"},
    "validation 1": {"validation": '"1"'},
    "validation negative": {"validation": '"-0.1"'},
    "validation nan": {"validation": '"nan"'},
    "validation inf": {"validation": '"inf"'},
    "validation not a number": {"validation": '"abc"'},
    "validation null": {"validation": "null"},
    "seed negative": {"seed": "-1"},
    "seed 2**32": {"seed": "4294967296"},
    "seed a boolean": {"seed": "false"},
    "seed a float": {"seed": "1.0"},
    "seed a string": {"seed": '"1"'},
    "items_per_shard 0": {"items_per_shard": "0"},
    "items_per_shard a boolean": {"items_per_shard": "true"},
    "items_per_shard a float": {"items_per_shard": "2.0"},
    "items_per_shard a string": {"items_per_shard": '"2"'},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_invalid_config(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_starting_values(repo):
    """configs/train-index.yaml: the starting values of the Interface."""
    cfg = load_config(repo / "configs/train-index.yaml")
    assert cfg == TrainIndexConfig(
        clips_dir="data/perception",
        pair_clips_dir="data/pair_clips",
        pair_labels_dir="data/pair_labels",
        out_dir="data/train_index",
        validation="0.05",
        seed=0,
        items_per_shard=256,
    )
