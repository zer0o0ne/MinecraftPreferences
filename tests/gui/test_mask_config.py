"""`load_config` of specs/action-mask.md (Notation: Config; Interface: starting values) and
`load_masks` reading the configured catalogue and pitch limit."""

import copy
import dataclasses
import os

import pytest
import yaml
from mask_oracle import WORLD, allowed, names

from mcpref.gui.mask import ActionMaskConfig, load_config, load_masks, state_mask

VALID = {"screens": "data/elsewhere/screens.yaml", "pitch_limit": 90}


def write(tmp_path, text: str) -> os.PathLike:
    path = tmp_path / "action-mask.yaml"
    path.write_text(text)
    return path


def dump(tmp_path, doc: dict) -> os.PathLike:
    return write(tmp_path, yaml.safe_dump(doc, sort_keys=False))


def test_approved_config(repo):
    cfg = load_config(repo / "configs/action-mask.yaml")
    assert cfg == ActionMaskConfig(screens="configs/gui-screens.yaml", pitch_limit=90)
    assert [f.name for f in dataclasses.fields(cfg)] == ["screens", "pitch_limit"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.pitch_limit = 1


def test_screens_kept_as_written(tmp_path):
    assert load_config(dump(tmp_path, VALID)).screens == "data/elsewhere/screens.yaml"


@pytest.mark.parametrize(
    "value", [1, None, True, ["configs/gui-screens.yaml"], {"a": "b"}]
)
def test_screens_must_be_a_string(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(dump(tmp_path, VALID | {"screens": value}))


@pytest.mark.parametrize("text, value", [("90", 90), ("0.25", 0.25), ("1.0e+3", 1000)])
def test_pitch_limit_accepted(tmp_path, text, value):
    cfg = load_config(write(tmp_path, f"screens: x.yaml\npitch_limit: {text}\n"))
    assert cfg.pitch_limit == value


@pytest.mark.parametrize(
    "text", ["0", "0.0", "-1", "-0.5", "true", "false", "'90'", "null", ".nan", "[90]"]
)
def test_pitch_limit_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, f"screens: x.yaml\npitch_limit: {text}\n"))


@pytest.mark.parametrize(
    "doc", [{"screens": "x.yaml"}, {"pitch_limit": 90}, VALID | {"turn_deg": 10}]
)
def test_missing_or_unknown_key(tmp_path, doc):
    with pytest.raises(ValueError):
        load_config(dump(tmp_path, doc))


def full_screen(*names: str) -> dict:
    """A class whose buttons 37, 38, ... are 200 x 20 rectangles at y = 200, 230, ..."""
    elements = [{"index": 37 + j, "name": name, "kind": "button",
                 "hit": [220, 200 + 30 * j, 200, 20], "click": [320.0, 210.0 + 30 * j]}
                for j, name in enumerate(names)]  # fmt: skip
    member = {"blits": [], "dynamic": [], "labels": []}
    return {"members": {"only": member}, "description": "Full screen.",
            "window": [0, 0, 640, 360], "elements": elements}  # fmt: skip


def test_load_masks_reads_the_configured_catalogue(
    tmp_path, repo, catalogue, write_config, monkeypatch
):
    """Tiny catalogue (n_max 38): `alpha` slots 1, 2 and button 37 `c`; `beta` button 37
    renamed `option_7`; `death` with `respawn` at 38; `pause`. Paths relative to the
    repository root, cwd elsewhere; pitch_limit 45.5."""
    classes = copy.deepcopy(catalogue["classes"])
    classes["beta"]["elements"][0]["name"] = "option_7"
    classes["death"] = full_screen("title_screen", "respawn")
    classes["pause"] = full_screen("back_to_game")
    path = write_config({**catalogue, "classes": classes})
    screens = os.path.relpath(path, repo)
    cfg = load_config(dump(tmp_path, {"screens": screens, "pitch_limit": 45.5}))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    masks = load_masks(cfg)
    a = names(38)
    assert masks.names == a and len(a) == 187 and masks.pitch_limit == 45.5
    assert set(masks.screens) == {"alpha", "beta", "death"}
    clicks = {f"{k}_{i}" for k in ("left_click", "right_click", "shift_click", "double_click")
              for i in (1, 2)}  # fmt: skip
    alpha = {"close", "left_click_37", "shift_click_37", *clicks}
    assert allowed(state_mask(masks, True, "alpha", 0.0, 0), a) == alpha
    beta = {"close", "left_click_37", "right_click_37", "shift_click_37"}
    assert allowed(state_mask(masks, True, "beta", 0.0, 0), a) == beta
    death = {"left_click_38", "shift_click_38"}
    assert allowed(state_mask(masks, True, "death", 0.0, 0), a) == death
    with pytest.raises(ValueError):
        state_mask(masks, True, "pause", 0.0, 0)
    world = set(WORLD) - {"hotbar_1"}
    assert allowed(state_mask(masks, False, None, 45.0, 0), a) == world
    assert allowed(state_mask(masks, False, None, -45.0, 0), a) == world
    assert allowed(state_mask(masks, False, None, 45.5, 0), a) == world - {"turn_down"}
    assert allowed(state_mask(masks, False, None, -45.5, 0), a) == world - {"turn_up"}
