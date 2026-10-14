"""`load_config` of src/mcpref/stacking/config.py against specs/gui-stack.md (Notation: the
config; Interface: starting values, pyproject)."""

import dataclasses
from pathlib import Path

import pytest
import tomli
import torch

from mcpref.stacking.config import GuiStackConfig, load_config

NAMES = {"masks_dir", "hidden", "l2", "lr", "betas", "adam_eps", "steps", "seed", "device",
         "out_dir"}  # fmt: skip
VALID = {  # YAML lines of a valid config
    "masks_dir": '"m/features"', "hidden": "3", "l2": "0", "lr": "0.5",
    "betas": "[0, 0.999]", "adam_eps": "1e-8", "steps": "1", "seed": "18446744073709551615",
    "device": '"cpu"', "out_dir": '"o/stack"',
}  # fmt: skip


def load(tmp_path: Path, **changes: str) -> GuiStackConfig:
    path = tmp_path / "c.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in (VALID | changes).items()))
    return load_config(path)


def test_reads_every_field(tmp_path, repo, monkeypatch):
    """`1e-8` (a string to PyYAML) and the integer 0 of l2 read with float(); the betas two
    floats; paths resolved against the repository root, not the working directory."""
    monkeypatch.chdir(tmp_path)
    cfg = load(tmp_path)
    assert {f.name for f in dataclasses.fields(GuiStackConfig)} == NAMES
    assert (cfg.hidden, cfg.steps, cfg.seed) == (3, 1, 2**64 - 1)
    assert all(type(v) is int for v in (cfg.hidden, cfg.steps, cfg.seed))
    assert (cfg.l2, cfg.lr, cfg.adam_eps) == (0.0, 0.5, 1e-8)
    assert all(type(v) is float for v in (cfg.l2, cfg.lr, cfg.adam_eps, *cfg.betas))
    assert list(cfg.betas) == [0.0, 0.999]
    assert torch.device(cfg.device) == torch.device("cpu")
    assert Path(repo, cfg.masks_dir) == repo / "m/features"
    assert Path(repo, cfg.out_dir) == repo / "o/stack"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.steps = 2


def test_floats_read_from_strings(tmp_path):
    """l2, lr, adam_eps and the betas are read with float()."""
    cfg = load(
        tmp_path, l2='"1e-4"', lr='"2"', adam_eps="1.0e-30", betas='["0.5", "0"]'
    )
    assert (cfg.l2, cfg.lr, cfg.adam_eps, list(cfg.betas)) == (
        1e-4,
        2.0,
        1e-30,
        [0.5, 0.0],
    )


def test_device_of_another_machine_accepted(tmp_path):
    """Any string torch.device accepts; loading needs no GPU."""
    assert torch.device(load(tmp_path, device='"cuda:1"').device) == torch.device(
        "cuda:1"
    )


BAD = {
    "hidden 0": {"hidden": "0"},
    "hidden a boolean": {"hidden": "true"},
    "hidden a float": {"hidden": "3.0"},
    "hidden a string": {"hidden": '"3"'},
    "steps 0": {"steps": "0"},
    "steps a boolean": {"steps": "true"},
    "seed negative": {"seed": "-1"},
    "seed 2^64": {"seed": "18446744073709551616"},
    "seed a boolean": {"seed": "false"},
    "seed a float": {"seed": "1.0"},
    "l2 negative": {"l2": "-1.0e-9"},
    "lr 0": {"lr": "0"},
    "lr not a number": {"lr": '"fast"'},
    "adam_eps 0": {"adam_eps": "0"},
    "one beta": {"betas": "[0.9]"},
    "three betas": {"betas": "[0.9, 0.99, 0.999]"},
    "betas a scalar": {"betas": "0.9"},
    "beta 1": {"betas": "[0.9, 1]"},
    "beta negative": {"betas": "[-0.1, 0.9]"},
    "bad device": {"device": '"gpu"'},
    "device not a string": {"device": "0"},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_violated_constraints_raise_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def test_approved_config_has_starting_values(repo):
    cfg = load_config(repo / "configs/gui-stack.yaml")
    got = {
        k: getattr(cfg, k) for k in ("hidden", "l2", "lr", "adam_eps", "steps", "seed")
    }
    assert got == {"hidden": 64, "l2": 1e-4, "lr": 0.01, "adam_eps": 1e-8, "steps": 2000,
                   "seed": 0}  # fmt: skip
    assert list(cfg.betas) == [0.9, 0.999]
    assert torch.device(cfg.device) == torch.device("cpu")
    assert Path(repo, cfg.masks_dir) == repo / "data/gui_masks"
    assert Path(repo, cfg.out_dir) == repo / "data/gui_stack"


def test_pyproject_pins_torch(repo):
    project = tomli.loads((repo / "pyproject.toml").read_text())["project"]
    assert "torch==2.14.1" in project["dependencies"]
