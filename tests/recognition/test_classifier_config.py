"""`load_config` of src/mcpref/recognition/classifier.py against specs/gui-classifier.md
(Notation: Config; Interface: starting values)."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from mcpref.recognition.classifier import GuiClassifierConfig, load_config

VALID = {"masks_dir": "data/elsewhere", "criterion": "macro_recall"}
NAMES = "['agreement', 'macro_recall']"  # list(CRITERIA) of gui-masks


def write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "gui-classifier.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def test_reads_every_field(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert isinstance(cfg, GuiClassifierConfig)
    assert [f.name for f in dataclasses.fields(cfg)] == ["masks_dir", "criterion"]
    assert dataclasses.asdict(cfg) == VALID  # masks_dir as written
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.criterion = "agreement"


@pytest.mark.parametrize("criterion", ["agreement", "macro_recall"])
def test_every_criterion_is_accepted(tmp_path, criterion):
    assert load_config(write(tmp_path, VALID | {"criterion": criterion})).criterion == criterion  # fmt: skip


@pytest.mark.parametrize("doc", [{"masks_dir": "d"}, {"criterion": "agreement"}, VALID | {"tau": 3}])  # fmt: skip
def test_missing_or_unknown_key(tmp_path, doc):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, doc))


@pytest.mark.parametrize("value", [1, None, True, ["data/gui_masks"], {"a": "b"}])
def test_masks_dir_must_be_a_string(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, VALID | {"masks_dir": value}))


@pytest.mark.parametrize(
    "value", ["f1", "Agreement", "agreement ", 1, None, ["agreement"]]
)
def test_criterion_outside_criteria(tmp_path, value):
    """ValueError(f"criterion {value!r} is not one of {list(CRITERIA)}")."""
    with pytest.raises(ValueError) as info:
        load_config(write(tmp_path, VALID | {"criterion": value}))
    assert str(info.value) == f"criterion {value!r} is not one of {NAMES}"


def test_starting_values(repo):
    cfg = load_config(repo / "configs/gui-classifier.yaml")
    assert cfg == GuiClassifierConfig(masks_dir="data/gui_masks", criterion="agreement")
    masks = yaml.safe_load((repo / "configs/gui-masks.yaml").read_text())["out_dir"]
    assert Path(repo, cfg.masks_dir).resolve() == Path(repo, masks).resolve()
