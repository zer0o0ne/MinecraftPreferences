"""Consecutive ineffective actions and semantic progress; runtime-stagnation.md."""

from copy import deepcopy

import numpy as np

from mcpref.runtime.env.actions import increased
from mcpref.runtime.env.progress import action_progress, advanced

STATS = ("mine_block", "use_item", "craft_item", "pickup", "break_item", "kill_entity")
CUSTOM = ("jump", "damage_dealt", "damage_dealt_absorbed", "damage_dealt_resisted")


def item(value: dict) -> tuple:
    """Normalize empty slots without losing the identity of occupied slots."""
    kind = str(value["type"]).removeprefix("minecraft:")
    quantity = int(value["quantity"])
    return (kind, quantity) if kind not in {"none", "air"} and quantity > 0 else ()


def snapshot(info: dict) -> dict:
    """Detach the baseline from mutable MineStudio dictionaries and numpy scalars."""
    pos = info["player_pos"]
    facts = {
        "position": tuple(float(pos[k]) for k in ("x", "y", "z", "pitch"))
        + (float(pos["yaw"]) % 360,),
        "inventory": tuple(
            (int(slot), item(value))
            for slot, value in sorted(info["inventory"].items())
            if item(value)
        ),
        "equipment": tuple(
            (slot, str(value["type"]), int(value["damage"]), int(value["maxDamage"]))
            for slot, value in sorted(info.get("equipped_items", {}).items())
        ),
        "gui": bool(info["isGuiOpen"]),
    }
    stats = {
        group: {k: int(v) for k, v in info.get(group, {}).items()} for group in STATS
    }
    stats["custom"] = {key: int(info["custom"].get(key, 0)) for key in CUSTOM}
    return {"facts": facts, "stats": stats, "progress": action_progress(info)}


def progressed(before: dict, after: dict) -> bool:
    """Ignore clocks, image animation and stat decreases."""
    return (
        before["facts"] != after["facts"]
        or advanced(before["progress"], after["progress"])
        or any(
            increased(values, before["stats"][group], key)
            for group, values in after["stats"].items()
            for key in values
        )
    )


def copy_state(value: dict) -> dict:
    """Require complete continuation state, including the observation baseline."""
    baseline = value["baseline"]
    for key in ("facts", "stats", "progress"):
        baseline[key]
    return deepcopy(
        {"last": value["last"], "count": value["count"], "baseline": baseline}
    )


class Stagnation:
    """Track raw ticks within each macro action and mask only its ineffective run."""

    def __init__(self, limit: int):
        self.limit = limit
        self.last, self.count, self.baseline = None, 0, None
        self.changed = False

    def observe(self, info: dict) -> None:
        current = snapshot(info)
        if self.baseline is not None:
            self.changed |= progressed(self.baseline, current)
        self.baseline = current

    def finish(self, action: int | None, changed: bool) -> None:
        if self.changed or changed:
            self.count = 0
        elif action is not None:
            self.count = self.count + 1 if action == self.last else 1
        if action is not None:
            self.last = action
        self.changed = False

    def mask(self, mask: np.ndarray) -> None:
        if (
            self.count > self.limit
            and self.last is not None
            and np.count_nonzero(mask) > 1
        ):
            mask[self.last] = False

    def save(self) -> dict:
        return deepcopy(
            {"last": self.last, "count": self.count, "baseline": self.baseline}
        )

    def restore(self, value: dict) -> None:
        state = copy_state(value)
        self.last, self.count, self.baseline = (
            state["last"],
            state["count"],
            state["baseline"],
        )
        self.changed = False
