"""Interface of specs/action-mask.md: the mask of A per state and per clip."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from mcpref import config_mapping, number, repo_path
from mcpref.gui.actions import action_names
from mcpref.gui.screens import Element, ScreenClass, load_screens, right_handled
from mcpref.vpt.actions import WORLD_ACTIONS

NOT_D = frozenset({"death", "pause"})
"""Notation: the classes of K not in D."""


@dataclass(frozen=True)
class ActionMaskConfig:
    """Notation: configs/action-mask.yaml; screens as written, resolved against the
    repository root where read."""

    screens: str
    pitch_limit: float


@dataclass(frozen=True)
class Masks:
    """Interface: A, the base of (1), the masks of (2) per class of D and (2') of death."""

    names: tuple[str, ...]
    world: np.ndarray
    screens: dict[str, np.ndarray]
    pitch_limit: float


def load_config(path: Path) -> ActionMaskConfig:
    """Notation: the YAML file loaded into ActionMaskConfig; violated constraints raise."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, ActionMaskConfig)
    repo_path("screens", raw["screens"])
    pitch_limit = number("pitch_limit", raw["pitch_limit"])
    if not pitch_limit > 0:
        raise ValueError(f"pitch_limit {raw['pitch_limit']!r} is not > 0")
    return ActionMaskConfig(raw["screens"], pitch_limit)


def allowing(pos: Mapping[str, int], allowed: Iterable[str]) -> np.ndarray:
    """(1), (2), (2'): the mask over A with exactly the actions `allowed` true."""
    mask = np.zeros(len(pos), dtype=bool)
    mask[[pos[a] for a in allowed]] = True
    return mask


def clicks(e: Element) -> list[str]:
    """(2): left and shift on e; right on a slot or a right-handled button; double on a
    slot."""
    kinds = ["left_click", "shift_click"]
    kinds += ["right_click"] if e.kind == "slot" or right_handled(e) else []
    kinds += ["double_click"] if e.kind == "slot" else []
    return [f"{c}_{e.index}" for c in kinds]


def screen_base(pos: Mapping[str, int], cls: ScreenClass) -> np.ndarray:
    """(2): close and the clicks of the elements of E_k."""
    return allowing(pos, ["close", *(a for e in cls.elements for a in clicks(e))])


def death_base(pos: Mapping[str, int], cls: ScreenClass) -> np.ndarray:
    """(2'): the left and shift clicks of the element named respawn."""
    respawn = [e.index for e in cls.elements if e.name == "respawn"]
    return allowing(
        pos, [f"{c}_{i}" for i in respawn for c in ("left_click", "shift_click")]
    )


def load_masks(cfg: ActionMaskConfig) -> Masks:
    """Interface: A of the catalogue, the bases of (1), (2), (2')."""
    catalogue = load_screens(repo_path("screens", cfg.screens))
    names = action_names(catalogue)
    pos = {a: i for i, a in enumerate(names)}
    screens = {k: screen_base(pos, c) for k, c in catalogue.items() if k not in NOT_D}
    screens["death"] = death_base(pos, catalogue["death"])
    return Masks(names, allowing(pos, WORLD_ACTIONS), screens, cfg.pitch_limit)


def world_mask(masks: Masks, pitch: float, hotbar: int) -> np.ndarray:
    """(1): the base without turn_up at theta <= -theta_max, turn_down at theta >=
    theta_max, and hotbar_(h + 1)."""
    mask = masks.world.copy()
    mask[masks.names.index("turn_up")] = pitch > -masks.pitch_limit
    mask[masks.names.index("turn_down")] = pitch < masks.pitch_limit
    mask[masks.names.index(f"hotbar_{hotbar + 1}")] = False
    return mask


def screen_mask(masks: Masks, cls: str) -> np.ndarray:
    """(2), (2'), (3): a copy of the class's base; ValueError for pause and other."""
    if cls not in masks.screens:
        raise ValueError(f"no mask for the screen {cls}")
    return masks.screens[cls].copy()


def state_mask(
    masks: Masks, gui: bool, cls: str | None, pitch: float, hotbar: int
) -> np.ndarray:
    """Interface: (1)-(3), a new array of shape (|A|,)."""
    return screen_mask(masks, cls) if gui else world_mask(masks, pitch, hotbar)


def clip_masks(masks: Masks, info: dict, lines: list[dict]) -> np.ndarray:
    """(4): the mask of every state s of a clip, shape (L, |A|)."""
    states = info["states"]
    out = np.zeros((len(states), len(masks.names)), dtype=bool)
    for s, t in enumerate(states):
        h = lines[t - states[0]]["hotbar"]
        g, k, theta = info["gui"][s], info["classes"][s], info["pitch"][s]
        out[s] = state_mask(masks, g, k, theta, h)
    return out
