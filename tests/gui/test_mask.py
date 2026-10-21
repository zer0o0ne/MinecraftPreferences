"""`load_masks`, `state_mask` of specs/action-mask.md, (1)-(3) and Behaviour, on the approved
catalogue: hand-picked cases and the naive transcription of mask_oracle.py."""

import dataclasses

import numpy as np
import pytest
from mask_oracle import (
    KINDS,
    LIMIT,
    WORLD,
    D,
    allowed,
    death_allowed,
    gui_allowed,
    names,
    world_allowed,
)

from mcpref.gui.mask import Masks, load_config, load_masks, state_mask

A = names(90)
STATES = [(-LIMIT, 0), (0.0, 4), (LIMIT, 8)]  # (theta, h): the domain's ends and inside

# (class, index, flags of KINDS: left, right, shift, double); element names in comments.
HAND = [
    ("chest6", 90, "1111"),  # slot: all four clicks
    ("inventory", 47, "1010"),  # recipe_book_button
    ("crafting_book", 54, "1110"),  # book_recipe_0
    ("crafting_book", 73, "1110"),  # book_recipe_19
    ("crafting_book", 47, "1010"),  # recipe_book_button
    ("crafting_book", 48, "1010"),  # book_tab_0
    ("inventory_book", 66, "1110"),  # book_recipe_12
    ("furnace_book", 46, "1110"),  # book_recipe_0
    ("furnace_book", 67, "1010"),  # book_next_page
    ("enchanting_table", 39, "1110"),  # option_0
    ("enchanting_table", 41, "1110"),  # option_2
    ("stonecutter", 39, "1110"),  # recipe_0
    ("stonecutter", 50, "1110"),  # recipe_11
    ("loom", 41, "1110"),  # pattern_0
    ("loom", 56, "1110"),  # pattern_15
    ("merchant", 40, "1010"),  # trade_0
    ("merchant", 46, "1010"),  # trade_6
    ("beacon", 37, "1111"),  # the payment slot
    ("beacon", 38, "1010"),  # primary_speed
    ("beacon", 45, "1010"),  # confirm
    ("beacon", 46, "1010"),  # cancel
    ("lectern", 37, "1010"),  # prev_page
    ("lectern", 40, "1010"),  # take_book
    ("sleep", 37, "1010"),  # leave_bed
    ("chest3", 64, "0000"),  # absent indices
    ("chest3", 90, "0000"),
    ("inventory", 48, "0000"),
    ("enchanting_table", 42, "0000"),
    ("lectern", 1, "0000"),
    ("lectern", 41, "0000"),
    ("sleep", 38, "0000"),
]


def test_masks_record(masks):
    fields = [f.name for f in dataclasses.fields(Masks)]
    assert fields == ["names", "world", "screens", "pitch_limit"]
    assert masks.names == A and len(A) == 395
    assert set(masks.screens) == {*D, "death"}
    assert masks.pitch_limit == LIMIT
    with pytest.raises(dataclasses.FrozenInstanceError):
        masks.pitch_limit = 1.0


def test_d_is_the_catalogue_without_death_and_pause(classes):
    assert set(classes) - {"death", "pause"} == set(D) and len(D) == 28


def test_bases(masks, classes):
    # Behaviour: the world base without its pitch and hotbar terms; the bases of (2), (2')
    assert allowed(masks.world, A) == set(WORLD)
    for k in D:
        assert allowed(masks.screens[k], A) == gui_allowed(classes[k]["elements"]), k
    assert allowed(masks.screens["death"], A) == {"left_click_37", "shift_click_37"}


def test_world_mask_by_hand(masks):
    # theta = -90, h = 0: turn_up and hotbar_1 masked; theta = 90, h = 8: turn_down, hotbar_9
    low = allowed(state_mask(masks, False, None, -90.0, 0), A)
    high = allowed(state_mask(masks, False, None, 90.0, 8), A)
    assert low == set(WORLD) - {"turn_up", "hotbar_1"}
    assert high == set(WORLD) - {"turn_down", "hotbar_9"}


@pytest.mark.parametrize("h", range(9))
@pytest.mark.parametrize("theta", [-LIMIT, -89.75, 0.0, 37.5, 89.75, LIMIT])
def test_world_mask(masks, theta, h):
    found = allowed(state_mask(masks, False, None, theta, h), A)
    assert found == world_allowed(theta, h, LIMIT)


@pytest.mark.parametrize("cls", ["chest6", "death", "pause", "other"])
def test_world_ignores_the_class(masks, cls):
    found = allowed(state_mask(masks, False, cls, 37.5, 2), A)
    assert found == world_allowed(37.5, 2, LIMIT)


@pytest.mark.parametrize("cls, i, flags", HAND)
def test_gui_hand_picked(masks, cls, i, flags):
    found = allowed(state_mask(masks, True, cls, 12.5, 3), A)
    expected = {k for k, f in zip(KINDS, flags, strict=True) if f == "1"}
    assert {k for k in KINDS if f"{k}_{i}" in found} == expected


@pytest.mark.parametrize("theta, h", STATES)
def test_gui_close_allowed_and_world_masked(masks, theta, h):
    for k in D:
        found = allowed(state_mask(masks, True, k, theta, h), A)
        assert "close" in found and not found & set(WORLD), k


@pytest.mark.parametrize("theta, h", STATES)
def test_gui_mask_matches_naive_transcription(masks, classes, theta, h):
    for k in D:
        found = allowed(state_mask(masks, True, k, theta, h), A)
        assert found == gui_allowed(classes[k]["elements"]), k


@pytest.mark.parametrize("theta, h", STATES)
def test_death_mask(masks, classes, theta, h):
    # (2'): respawn is element 37 of `death` (specs/gui-screens.md (3))
    found = allowed(state_mask(masks, True, "death", theta, h), A)
    assert found == {"left_click_37", "shift_click_37"}
    assert found == death_allowed(classes["death"]["elements"])


@pytest.mark.parametrize("cls", ["pause", "other"])
def test_outside_the_domain_raises(masks, cls):
    with pytest.raises(ValueError) as info:
        state_mask(masks, True, cls, 0.0, 0)
    assert str(info.value) == f"no mask for the screen {cls}"


def test_state_mask_returns_a_new_array(repo):
    own = load_masks(load_config(repo / "configs/action-mask.yaml"))
    world, screens = own.world.copy(), {k: v.copy() for k, v in own.screens.items()}
    for gui, cls in [(False, None), (True, "chest6"), (True, "death")]:
        first = state_mask(own, gui, cls, LIMIT, 8)
        expected = first.copy()
        first[:] = ~first
        assert np.array_equal(state_mask(own, gui, cls, LIMIT, 8), expected), cls
    # the world terms of an earlier state do not stay in the base
    found = allowed(state_mask(own, False, None, -LIMIT, 0), A)
    assert found == world_allowed(-LIMIT, 0, LIMIT)
    assert np.array_equal(own.world, world)
    assert all(np.array_equal(own.screens[k], v) for k, v in screens.items())
