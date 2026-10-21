"""Naive transcriptions of specs/action-mask.md (Notation: A, D, right-handled; Math (1), (2),
(2')) on the plain YAML elements of a catalogue; a mask as the set of its allowed names."""

import re

import numpy as np

MOVES = ("forward", "back", "left", "right", "forward_left", "forward_right",
         "back_left", "back_right")  # fmt: skip
WORLD = (*MOVES, "jump", *(f"jump_{m}" for m in MOVES), "sneak", "turn_left",
         "turn_right", "turn_up", "turn_down", "attack", "use",
         *(f"hotbar_{i}" for i in range(1, 10)), "open_inventory")  # fmt: skip
KINDS = ("left_click", "right_click", "shift_click", "double_click")
RIGHT = r"(book_recipe|option|recipe|pattern)_[0-9]+"
LIMIT = 90.0  # pitch_limit of configs/action-mask.yaml
# D = K without `death`, `pause` (specs/gui-screens.md (1)).
D = ("inventory", "inventory_book", "crafting", "crafting_book", "furnace", "furnace_book",
     "chest3", "chest6", "shulker_box", "dispenser", "hopper", "brewing_stand",
     "enchanting_table", "grindstone", "stonecutter", "loom", "cartography_table",
     "merchant", "beacon", "lectern", "anvil", "horse", "horse_chest3", "horse_chest6",
     "horse_chest9", "horse_chest12", "horse_chest15", "sleep")  # fmt: skip


def names(n_max: int) -> tuple[str, ...]:
    """A: the 34 world actions, `close`, then each click kind for i = 1..n_max."""
    clicks = (f"{k}_{i}" for k in KINDS for i in range(1, n_max + 1))
    return (*WORLD, "close", *clicks)


def allowed(mask: np.ndarray, a: tuple[str, ...]) -> set[str]:
    """The names a bool mask over `a` allows."""
    assert mask.dtype == np.bool_ and mask.shape == (len(a),)
    return {n for n, ok in zip(a, mask.tolist(), strict=True) if ok}


def world_allowed(theta: float, h: int, limit: float) -> set[str]:
    """(1)."""
    out = set(WORLD) - {f"hotbar_{h + 1}"}
    if theta <= -limit:
        out.discard("turn_up")
    if theta >= limit:
        out.discard("turn_down")
    return out


def right_handled(e: dict) -> bool:
    """Notation: a button whose name fully matches RIGHT."""
    return e["kind"] == "button" and re.fullmatch(RIGHT, e["name"]) is not None


def gui_allowed(elements: list[dict]) -> set[str]:
    """(2) for the YAML elements of a class of D."""
    out = {"close"}
    for e in elements:
        i, slot = e["index"], e["kind"] == "slot"
        out |= {f"left_click_{i}", f"shift_click_{i}"}
        out |= {f"right_click_{i}"} if slot or right_handled(e) else set()
        out |= {f"double_click_{i}"} if slot else set()
    return out


def death_allowed(elements: list[dict]) -> set[str]:
    """(2'): the left and shift clicks of `respawn`."""
    (i,) = [e["index"] for e in elements if e["name"] == "respawn"]
    return {f"left_click_{i}", f"shift_click_{i}"}
