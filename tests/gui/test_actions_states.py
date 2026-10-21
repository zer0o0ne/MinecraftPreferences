"""`opens_container` and (8) `excluded_states` of specs/vpt-gui-actions.md (the
first-line rule for states); hand cases."""

import pytest
from gui_action_lines import box, episode, inv, world

from mcpref.gui.actions import excluded_states, opens_container

fs = frozenset


@pytest.mark.parametrize(
    "k, expected",
    [
        (0, False),  # k = 0
        (1, False),  # second line of the span
        (3, True),  # first line of a container span
        (4, False),
        (6, False),  # first line of a player inventory span
        (7, False),  # container after the player inventory, one span
        (9, False),
    ],
)
def test_opens_container(k, expected):
    lines = episode(box(), box(), world(), box(), box(), world(), inv(), box(), world(),
                    inv())  # fmt: skip
    assert opens_container(lines, k) is expected


def test_excluded_states():
    lines = episode(
        world(), box(), box(), box(), world(),  # 1: first line of a container
        inv(), inv(), world(),  # 5: first line of the player inventory
        inv(), inv(), world(),  # [8, 9]: in `cut`
    )  # fmt: skip
    classes = {0: "other", 1: "other", 2: "box", 3: "pause", 5: "other", 6: "death",
               8: "other", 9: "inv", 10: "other"}  # fmt: skip
    found = excluded_states(lines, classes, {8, 9})
    assert found == fs({3, 5, 6}) and type(found) is frozenset  # 6: `death`, revision


def test_excluded_states_from_line_0_and_only_over_classes():
    lines = episode(box(), box(), box(), world())
    assert excluded_states(lines, {0: "other", 1: "box"}, ()) == fs({0})
    assert excluded_states(lines, {1: "box", 2: "pause"}, []) == fs({2})
    assert excluded_states(lines, {1: "box", 2: "pause"}, {0, 1, 2}) == fs()
