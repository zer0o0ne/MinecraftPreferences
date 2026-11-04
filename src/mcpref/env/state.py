"""(3), (4) of specs/env.md: raw MineStudio results converted to EnvState."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from mcpref.gui.mask import Masks, state_mask


@dataclass(frozen=True)
class EnvState:
    """Interface: frame, perception line, GUI class, hotbar and action mask."""

    frame: np.ndarray
    line: Mapping
    gui: bool
    cls: str | None
    hotbar: int
    mask: np.ndarray


def inventory(info: Mapping) -> list[dict]:
    """(3): non-air slot entries in slot-index order, without the minecraft prefix."""
    from minestudio.simulator.minerl.herobraine.hero.mc import strip_item_prefix

    out = []
    for slot in sorted(info["inventory"]):
        item = info["inventory"][slot]
        kind = strip_item_prefix(item["type"])
        if kind != "air":
            out.append({"type": kind, "quantity": item["quantity"]})
    return out


def perception_line(info: Mapping, hotbar: int) -> Mapping:
    """(3): inventory, selected slot and the five player_pos values as floats."""
    pos = info["player_pos"]
    return {
        "inventory": inventory(info),
        "hotbar": hotbar,
        "yaw": float(pos["yaw"]),
        "pitch": float(pos["pitch"]),
        "xpos": float(pos["x"]),
        "ypos": float(pos["y"]),
        "zpos": float(pos["z"]),
    }


def observation(
    obs: Mapping,
    info: Mapping,
    hotbar: int,
    previous_gui: bool,
    cause: str | None,
    gamma: Callable[[list[np.ndarray]], list[str]],
) -> tuple[np.ndarray, Mapping, bool, str | None]:
    """(3), (4): the frame, line and GUI flag with the container first-line rule."""
    raw_gui = bool(info["isGuiOpen"])
    container = raw_gui and not previous_gui and cause == "use"
    gui = raw_gui and not container
    cls = gamma([obs["image"]])[0] if gui else None
    return obs["image"], perception_line(info, hotbar), gui, cls


def convert(
    parts: tuple[np.ndarray, Mapping, bool, str | None], masks: Masks, hotbar: int
) -> EnvState:
    """(5): an EnvState whose mask is constructed after automatic GUI closing."""
    frame, line, gui, cls = parts
    mask = state_mask(masks, gui, cls, line["pitch"], hotbar)
    return EnvState(frame, line, gui, cls, hotbar, mask)


def check_episode(result: tuple) -> None:
    """(4): every terminated, truncated or info-error result is unrecoverable."""
    _, _, terminated, truncated, info = result
    if terminated or truncated or "error" in info:
        raise RuntimeError("episode ended")
