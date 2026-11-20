"""Black-box semantic watchdog tests with test-owned limit=3, Delta=6."""

import copy

import numpy as np
import pytest

from runtime.env.fake_simulator import FRAME, idle_progress, raw
from runtime.env.test_continuation import compare_restored


def attack(world):
    return world.index("attack")


def ticks(world, **info):
    world.fake.results.extend(raw(**info) for _ in range(6))
    return world.step("attack")


def test_strict_limit_then_different_action_releases_ban(ready):
    for _ in range(3):
        assert ready.step("attack").mask[attack(ready)]
    blocked = ready.step("attack")
    assert not blocked.mask[attack(ready)]
    assert ready.env.continuation().stagnation["count"] == 4
    assert ready.step("forward").mask[attack(ready)]
    assert ready.env.continuation().stagnation["count"] == 1


def test_changing_images_and_clocks_do_not_count(ready):
    for i in range(4):
        state = ticks(
            ready, frame=np.full_like(FRAME, i), custom={"play_one_minute": i * 6}
        )
    assert not state.mask[attack(ready)]


def test_masked_request_idles_without_increasing_count(ready):
    for _ in range(4):
        ready.step("attack")
    start = len(ready.fake.actions)
    state = ready.step("attack")
    assert not state.mask[attack(ready)]
    assert ready.env.continuation().stagnation["count"] == 4
    assert all(q["attack"] == 0 for q in ready.fake.actions[start:])
    assert ticks(ready, mine_block={"stone": 1}).mask[attack(ready)]
    assert ready.env.continuation().stagnation["count"] == 0


@pytest.mark.parametrize(
    "group,key",
    [
        ("mine_block", "stone"),
        ("use_item", "apple"),
        ("craft_item", "stick"),
        ("pickup", "stone"),
        ("break_item", "wooden_pickaxe"),
        ("kill_entity", "zombie"),
        ("custom", "jump"),
        ("custom", "damage_dealt"),
        ("custom", "damage_dealt_absorbed"),
        ("custom", "damage_dealt_resisted"),
    ],
)
def test_semantic_event_resets_run(ready, group, key):
    for _ in range(3):
        ready.step("attack")
    state = ticks(ready, **{group: {key: 1}})
    assert state.mask[attack(ready)]
    assert ready.env.continuation().stagnation["count"] == 0


def test_transient_changes_count_even_if_final_state_matches(ready):
    for _ in range(3):
        ready.step("attack")
    ready.fake.results.extend(
        [
            raw(inventory={0: {"type": "stone", "quantity": 1}}),
            *[raw() for _ in range(5)],
        ]
    )
    state = ready.step("attack")
    assert state.mask[attack(ready)]
    assert ready.env.continuation().stagnation["count"] == 0


def test_inventory_slot_transfer_counts_but_empty_representations_do_not(ready):
    ticks(ready, inventory={0: {"type": "stone", "quantity": 1}})
    for _ in range(3):
        ready.step("attack")
    assert ticks(ready, inventory={1: {"type": "stone", "quantity": 1}}).mask[
        attack(ready)
    ]
    ticks(ready, inventory={0: {"type": "none", "quantity": 0}})
    for _ in range(4):
        state = ticks(ready, inventory={7: {"type": "air", "quantity": 0}})
    assert not state.mask[attack(ready)]


def test_movement_resets_but_whole_yaw_turn_does_not(ready):
    for _ in range(3):
        ready.step("attack")
    pos = {"x": 1.000001, "y": 2, "z": 3, "yaw": 4, "pitch": 5}
    assert ticks(ready, pos=pos).mask[attack(ready)]
    for _ in range(3):
        ready.step("attack")
    assert not ticks(ready, pos=pos | {"yaw": 364}).mask[attack(ready)]


def test_last_allowed_action_never_removed(ready, monkeypatch):
    for _ in range(3):
        ready.step("attack")
    only_attack = np.zeros(len(ready.env.masks.names), dtype=bool)
    only_attack[attack(ready)] = True
    monkeypatch.setattr(
        "mcpref.runtime.env.state.state_mask", lambda *args: only_attack.copy()
    )
    state = ready.step("attack")
    assert state.mask.sum() == 1 and state.mask[attack(ready)]


@pytest.mark.parametrize("at", [3, 4])
def test_restore_preserves_streak_and_exact_next_mask(ready, at):
    for _ in range(at):
        state = ready.step("attack")
    saved = ready.env.continuation()
    compare_restored(ready, state, saved, "attack")
    assert saved.stagnation["count"] == at
    saved.stagnation["baseline"]["facts"].clear()
    assert ready.env.continuation().stagnation["baseline"]["facts"]


def test_slow_mining_advances_for_longer_than_limit(ready):
    for step in range(10):
        for tick in range(6):
            progress = idle_progress()
            progress["mining"] = {
                "active": True,
                "target": [1, 2, 3],
                "progress": (step * 6 + tick + 1) / 1000,
            }
            ready.fake.results.append(raw(action_progress=progress))
        assert ready.step("attack").mask[attack(ready)]
        assert ready.env.continuation().stagnation["count"] == 0


def test_unbreakable_target_and_changing_target_do_not_count(ready):
    for step in range(4):
        progress = idle_progress()
        progress["mining"] = {"active": True, "target": [1, 2, step], "progress": 0.0}
        state = ticks(ready, action_progress=progress)
    assert not state.mask[attack(ready)]


def test_decreasing_or_constant_mining_progress_does_not_count(ready):
    progress = idle_progress()
    progress["mining"] = {"active": True, "target": [1, 2, 3], "progress": 0.5}
    ticks(ready, action_progress=progress)
    for p in (0.4, 0.3, 0.3, 0.3):
        value = copy.deepcopy(progress)
        value["mining"]["progress"] = p
        state = ticks(ready, action_progress=value)
    assert not state.mask[attack(ready)]
