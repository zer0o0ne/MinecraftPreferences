"""(6)-(9) of specs/env.md: low-level inputs, completion and GUI schedules."""

from collections.abc import Mapping

import numpy as np

from mcpref.datasets.vpt.video.channels import DAMAGE, JUMP, MOVES
from mcpref.gui.screens import ScreenClass

MOVE_KEYS = {name: key for key, name in MOVES.items()}
LONG = frozenset((*MOVE_KEYS, "sneak", "attack", "use"))
TURNS = {
    "turn_left": (0.0, -1.0),
    "turn_right": (0.0, 1.0),
    "turn_up": (-1.0, 0.0),
    "turn_down": (1.0, 0.0),
}


def hold(q: dict, action: str) -> None:
    """(6), (9): movement, sneak, attack or use keys for one long-action tick."""
    if action in MOVE_KEYS:
        forward, right, jump = MOVE_KEYS[action]
        q["forward"], q["back"] = max(forward, 0), max(-forward, 0)
        q["right"], q["left"] = max(right, 0), max(-right, 0)
        q["jump"] = jump
    else:
        q[action] = 1


def immediate(q: dict, action: str, turn_deg: float) -> None:
    """(7), (9): one camera delta, hotbar press or inventory press."""
    if action in TURNS:
        pitch, yaw = (x * turn_deg for x in TURNS[action])
        q["camera"] = np.array((pitch, yaw), dtype=np.float32)
    elif action == "open_inventory":
        q["inventory"] = 1
    else:
        q[f"hotbar.{action.removeprefix('hotbar_')}"] = 1


def increased(after: Mapping, before: Mapping, key: str) -> bool:
    """(6): a cumulative statistic increased, with absent values equal to zero."""
    return after.get(key, 0) > before.get(key, 0)


def opened(before: Mapping, after: Mapping) -> bool:
    """(6): raw isGuiOpen changed from false to true."""
    return bool(after["isGuiOpen"]) and not bool(before["isGuiOpen"])


def completed(action: str, before: Mapping, after: Mapping) -> bool:
    """(6): the completion signal of a long world action."""
    if action == "jump" or action.startswith("jump_"):
        return increased(after["custom"], before["custom"], JUMP)
    if action == "attack":
        if any(
            increased(after["mine_block"], before["mine_block"], k)
            for k in after["mine_block"]
        ):
            return True
        return any(increased(after["custom"], before["custom"], k) for k in DAMAGE)
    if action == "use":
        stats = after["use_item"]
        return any(increased(stats, before["use_item"], k) for k in stats) or opened(
            before, after
        )
    return False


def click(
    q: dict,
    action: str,
    tick: int,
    cls: ScreenClass,
    cursor: tuple[float, float],
    cursor_scale: float,
) -> tuple[float, float]:
    """(8), (9): the cursor delta and press/release schedule of a canonical click."""
    kind, index = action.rsplit("_", 1)
    element = next(e for e in cls.elements if e.index == int(index))
    x, y = element.click
    if tick == 0:
        u, v = cursor
        q["camera"] = np.array(
            ((y - v) / cursor_scale, (x - u) / cursor_scale), dtype=np.float32
        )
    button = "use" if kind == "right_click" else "attack"
    presses = {1, 3} if kind == "double_click" else {1}
    releases = {2, 4} if kind == "double_click" else {2}
    if tick in presses | releases:
        q[button] = int(tick in presses)
    if kind == "shift_click" and tick in {1, 2}:
        q["sneak"] = 1
    return (x, y) if tick == 0 else cursor


def escape(q: dict, pressed: bool) -> None:
    """(4), (8), (9): Escape press or release on one tick."""
    q["ESC"] = int(pressed)
