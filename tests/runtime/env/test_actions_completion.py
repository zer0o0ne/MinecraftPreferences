"""Completion, strict stat comparison and hold boundaries of specs/env.md (5), (6)."""

import pytest

from runtime.env.fake_simulator import (
    DAMAGE,
    JUMP,
    MINE,
    USE,
    assert_actions,
    expected,
    raw,
)

IDLE = expected()
ATTACK_STATS = [
    ("mine_block", MINE),
    ("mine_block", "minecraft.mine_block:minecraft.dirt"),
    ("custom", DAMAGE),
    ("custom", DAMAGE + "_absorbed"),
    ("custom", DAMAGE + "_resisted"),
]


def stat_result(group, key, value=1):
    return raw(**{group: {key: value}})


def completed_actions(key, value, tick):
    held = expected(**key)
    return [held] * tick + [IDLE] * (6 - tick)


@pytest.mark.parametrize("tick", [1, 3, 6])
def test_jump_completes_on_increment_then_idles(ready, tick):
    ready.fake.results.extend(raw() for _ in range(tick - 1))
    ready.fake.results.append(stat_result("custom", JUMP))
    ready.step("jump")
    assert_actions(ready.fake, completed_actions({"jump": 1}, 1, tick))


@pytest.mark.parametrize("tick", [1, 3, 6])
@pytest.mark.parametrize("group, key", ATTACK_STATS)
def test_attack_completes_on_every_mine_and_damage_signal(ready, tick, group, key):
    ready.fake.results.extend(raw() for _ in range(tick - 1))
    ready.fake.results.append(stat_result(group, key))
    ready.step("attack")
    assert_actions(ready.fake, completed_actions({"attack": 1}, 1, tick))


@pytest.mark.parametrize("tick", [1, 3, 6])
@pytest.mark.parametrize("key", [USE, "minecraft.use_item:minecraft.bread"])
def test_use_completes_on_any_use_item_increment(ready, tick, key):
    ready.fake.results.extend(raw() for _ in range(tick - 1))
    ready.fake.results.append(stat_result("use_item", key))
    ready.step("use")
    assert_actions(ready.fake, completed_actions({"use": 1}, 1, tick))


def test_use_completes_when_gui_opens_without_use_item(ready):
    ready.fake.results.append(raw(gui=True))
    ready.step("use")
    assert_actions(ready.fake, [expected(use=1), *([IDLE] * 5)])


def test_missing_stat_is_zero_and_equal_values_do_not_complete(ready):
    ready.fake.reset_raw = raw(custom={JUMP: 4})
    ready.reset()
    ready.fake.results.extend(
        [
            raw(custom={JUMP: 4}),
            raw(),
            raw(custom={JUMP: 0}),
            raw(),
            raw(custom={JUMP: 0}),
            raw(),
        ]
    )
    ready.step("jump")
    assert_actions(ready.fake, [expected(jump=1)] * 6)


def test_decreased_stat_does_not_complete(ready):
    ready.fake.reset_raw = raw(custom={JUMP: 4})
    ready.reset()
    ready.fake.results.extend(raw(custom={JUMP: 3}) for _ in range(6))
    ready.step("jump")
    assert_actions(ready.fake, [expected(jump=1)] * 6)


def test_comparison_is_to_previous_raw_not_session_baseline(ready):
    ready.fake.reset_raw = raw(custom={JUMP: 4})
    ready.reset()
    ready.fake.results.extend([raw(), raw(custom={JUMP: 1})])
    ready.step("jump")
    assert_actions(ready.fake, [expected(jump=1)] * 2 + [IDLE] * 4)


def test_irrelevant_stat_does_not_complete_action(ready):
    ready.fake.results.append(stat_result("custom", JUMP))
    ready.step("forward")
    assert_actions(ready.fake, [expected(forward=1)] * 6)


def test_unfinished_same_action_continues_without_release(ready):
    ready.step("attack")
    ready.step("attack")
    assert_actions(ready.fake, [expected(attack=1)] * 12)


def test_unfinished_same_jump_movement_continues_without_release(ready):
    ready.step("jump_forward_left")
    ready.step("jump_forward_left")
    assert_actions(ready.fake, [expected(forward=1, left=1, jump=1)] * 12)


def test_same_action_after_completion_starts_a_new_hold(ready):
    ready.fake.results.append(stat_result("mine_block", MINE))
    ready.step("attack")
    ready.step("attack")
    assert_actions(
        ready.fake, [expected(attack=1), *([IDLE] * 5), *([expected(attack=1)] * 6)]
    )


def test_completion_on_last_tick_has_no_idle_but_resets_completion(ready):
    ready.fake.results.extend([raw() for _ in range(5)] + [stat_result("custom", JUMP)])
    ready.step("jump")
    ready.step("jump")
    assert_actions(ready.fake, [expected(jump=1)] * 12)


def test_other_action_releases_previous_hold_on_first_tick(ready):
    ready.step("attack")
    ready.step("forward")
    assert_actions(ready.fake, [expected(attack=1)] * 6 + [expected(forward=1)] * 6)


def test_immediate_action_releases_previous_hold_on_first_tick(ready):
    ready.step("use")
    ready.step("hotbar_2")
    assert_actions(
        ready.fake, [expected(use=1)] * 6 + [expected(**{"hotbar.2": 1}), *([IDLE] * 5)]
    )


def test_masked_action_releases_hold_and_idles(ready):
    ready.step("attack")
    ready.step("close")
    assert_actions(ready.fake, [expected(attack=1)] * 6 + [IDLE] * 6)
