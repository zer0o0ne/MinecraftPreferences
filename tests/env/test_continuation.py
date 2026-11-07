"""Continuation snapshots and exact fresh/active restoration of specs/env.md (11)."""

import copy
import dataclasses
import inspect

import numpy as np
import pytest

from mcpref.env.config import EnvConfig
from mcpref.env.state import EnvContinuation, EnvState
from mcpref.env.wrapper import Env

from .fake_simulator import DAMAGE, JUMP, MINE, USE, assert_actions, expected, raw

IDLE = expected()


def fresh_env(world):
    return Env(
        EnvConfig(
            "configs/engine.yaml",
            "configs/gui-classifier.yaml",
            "configs/action-mask.yaml",
            "configs/clips.yaml",
            20 / 3,
        )
    )


def clear_trace(fake):
    fake.actions.clear()
    fake.original_actions.clear()
    fake.action_ids.clear()
    fake.camera_ids.clear()


def assert_same_state(got: EnvState, wanted: EnvState):
    np.testing.assert_array_equal(got.frame, wanted.frame)
    assert got.line == wanted.line
    assert (got.gui, got.cls, got.hotbar) == (wanted.gui, wanted.cls, wanted.hotbar)
    np.testing.assert_array_equal(got.mask, wanted.mask)


def compare_restored(world, state, continuation, action, results=(), fresh=True):
    fake = world.fake
    boundary = fake.current
    clear_trace(fake)
    fake.results.extend(results)
    control_state = world.step(action)
    control_actions = copy.deepcopy(fake.actions)

    fake.current = boundary
    fake.results.clear()
    clear_trace(fake)
    target = fresh_env(world) if fresh else world.env
    before = (
        fake.reset_calls,
        list(fake.commands),
        len(fake.actions),
        len(world.classifier.calls),
        fake.noop_calls,
    )
    target.restore(state, continuation)
    after = (
        fake.reset_calls,
        list(fake.commands),
        len(fake.actions),
        len(world.classifier.calls),
        fake.noop_calls,
    )
    assert after == before

    fake.results.extend(results)
    restored_state = target.step(world.index(action))
    assert_actions(fake, control_actions)
    assert_same_state(restored_state, control_state)
    return restored_state


def test_continuation_public_interface():
    assert [field.name for field in dataclasses.fields(EnvContinuation)] == [
        "info",
        "cursor",
        "held_action",
    ]
    assert list(inspect.signature(Env.continuation).parameters) == ["self"]
    assert list(inspect.signature(Env.restore).parameters) == [
        "self",
        "state",
        "continuation",
    ]
    assert (
        inspect.get_annotations(Env.continuation, eval_str=True)["return"]
        is EnvContinuation
    )
    assert inspect.get_annotations(Env.restore, eval_str=True)["return"] is None


def test_continuation_is_exact_and_independent_deep_copy(world):
    stats = {
        "custom": {JUMP: 4},
        "mine_block": {MINE: 2},
        "use_item": {USE: 3},
    }
    world.fake.reset_raw = raw(gui=True, **stats)
    world.reset()
    world.step("left_click_1")
    first = world.env.continuation()
    assert set(first.info) == {"isGuiOpen", "custom", "mine_block", "use_item"}
    assert first.info == {"isGuiOpen": True, **stats}
    assert first.held_action is None

    wanted = copy.deepcopy(stats)
    first.info["custom"][JUMP] = 99
    first.info["mine_block"].clear()
    first.info["use_item"]["extra"] = 8
    second = world.env.continuation()
    assert second.info == {"isGuiOpen": True, **wanted}
    assert world.fake.current[4]["custom"] == wanted["custom"]
    assert second.info is not first.info
    assert second.info["custom"] is not first.info["custom"]


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize(
    "next_action,key", [("attack", "attack"), ("forward", "forward")]
)
def test_unfinished_hold_restore_matches_control(ready, fresh, next_action, key):
    state = ready.step("attack")
    continuation = ready.env.continuation()
    assert continuation.held_action == ready.index("attack")
    compare_restored(ready, state, continuation, next_action, fresh=fresh)
    assert_actions(ready.fake, [expected(**{key: 1})] * 6)


@pytest.mark.parametrize(
    "action,key", [("jump", "jump"), ("attack", "attack"), ("use", "use")]
)
def test_restored_nonzero_cumulative_stats_do_not_complete(world, action, key):
    stats = {
        "custom": {JUMP: 4, DAMAGE: 5},
        "mine_block": {MINE: 2},
        "use_item": {USE: 3},
    }
    world.fake.reset_raw = raw(**stats)
    state = world.reset()
    continuation = world.env.continuation()
    results = [raw(**stats) for _ in range(6)]
    compare_restored(world, state, continuation, action, results)
    assert_actions(world.fake, [expected(**{key: 1})] * 6)


def test_restored_hotbar_keeps_current_selection_masked(ready):
    state = ready.step("hotbar_3")
    continuation = ready.env.continuation()
    restored = compare_restored(ready, state, continuation, "hotbar_3")
    assert restored.hotbar == 2 and restored.line["hotbar"] == 2
    assert_actions(ready.fake, [IDLE] * 6)


def test_restored_repeated_click_has_zero_camera_delta(world):
    world.fake.reset_raw = raw(gui=True)
    world.reset()
    state = world.step("left_click_1")
    continuation = world.env.continuation()
    assert continuation.cursor == (248, 247)
    compare_restored(world, state, continuation, "left_click_1")
    zero = np.zeros(2, dtype=np.float32)
    wanted = [expected(camera=zero), expected(attack=1), *([IDLE] * 4)]
    assert_actions(world.fake, wanted)


def test_restore_preserves_raw_gui_true_behind_hidden_state(ready):
    ready.fake.results.extend([raw()] * 5 + [raw(gui=True)])
    state = ready.step("use")
    continuation = ready.env.continuation()
    assert state.gui is False
    assert continuation.info["isGuiOpen"] is True

    restored = compare_restored(
        ready, state, continuation, "open_inventory", [raw(gui=True)]
    )
    assert (restored.gui, restored.cls) == (True, "inventory")
    assert_actions(ready.fake, [expected(inventory=1), *([IDLE] * 5)])
