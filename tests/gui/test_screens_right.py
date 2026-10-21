"""`right_handled` of src/mcpref/gui/screens.py against specs/gui-screens.md (Interface) and
specs/action-mask.md (Notation: right-handled button)."""

import pytest

from mcpref.gui.screens import Element, Rect, right_handled


def element(name: str, kind: str) -> Element:
    return Element(41, name, kind, Rect(name, 0, 0, 18, 18), (9.0, 9.0))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("recipe_book_button", False),
        ("recipe_0", True),
        ("book_recipe_12", True),
        ("option_2", True),
        ("pattern_15", True),
        ("book_tab_0", False),
        ("recipe_", False),  # [0-9]+: at least one digit
        ("pattern_1x", False),  # full match
        ("xoption_2", False),
        ("option_2\n", False),
        ("Option_2", False),
        ("recipe_-1", False),
        ("recipe_١", False),  # [0-9], not any Unicode digit
        ("book_recipe_0_1", False),
        ("trade_0", False),
        ("respawn", False),
    ],
)
def test_right_handled_buttons(name, expected):
    assert right_handled(element(name, "button")) is expected


@pytest.mark.parametrize("name", ["s1", "recipe_0", "option_2", "pattern_15"])
def test_no_slot_is_right_handled(name):
    assert right_handled(element(name, "slot")) is False


def test_right_handled_on_the_approved_catalogue(approved_screens):
    """Notation of action-mask: the recipes of the three recipe-book classes, the options of
    `enchanting_table`, the recipes of `stonecutter`, the patterns of `loom`, nothing else."""
    groups = {"inventory_book": ("book_recipe", 20), "crafting_book": ("book_recipe", 20),
              "furnace_book": ("book_recipe", 20), "enchanting_table": ("option", 3),
              "stonecutter": ("recipe", 12), "loom": ("pattern", 16)}  # fmt: skip
    expected = {(k, f"{p}_{j}") for k, (p, n) in groups.items() for j in range(n)}
    found = {
        (k, e.name)
        for k, cls in approved_screens.items()
        for e in cls.elements
        if right_handled(e)
    }
    assert found == expected and len(found) == 91
