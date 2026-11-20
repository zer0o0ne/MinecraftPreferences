"""The engine observation and finite progress phases are not elapsed-time guards."""

import pytest

from mcpref.runtime.env.progress import action_progress
from runtime.env.fake_simulator import idle_progress, raw


def use_signal(action, elapsed, duration, charge):
    signal = idle_progress()
    signal["item_use"] = {
        "active": True,
        "action": action,
        "elapsed": elapsed,
        "remaining": duration - elapsed,
        "duration": duration,
        "charge_duration": charge,
    }
    return signal


def test_eating_beyond_four_actions_is_not_interrupted(ready):
    use = ready.index("use")
    for start in range(0, 30, 6):
        for elapsed in range(start + 1, start + 7):
            ready.fake.results.append(
                raw(action_progress=use_signal("EAT", elapsed, 32, 32))
            )
        assert ready.step("use").mask[use]
    ready.fake.results.extend(
        [
            raw(action_progress=use_signal("EAT", 31, 32, 32)),
            raw(use_item={"apple": 1}),
        ]
    )
    assert ready.step("use").mask[use]
    assert ready.env.continuation().stagnation["count"] == 0


def test_charged_bow_elapsed_counter_does_not_postpone_ban(ready):
    use = ready.index("use")
    for start in range(0, 48, 6):
        for elapsed in range(start + 1, start + 7):
            ready.fake.results.append(
                raw(action_progress=use_signal("BOW", elapsed, 72000, 20))
            )
        state = ready.step("use")
        assert bool(state.mask[use]) is (start < 42)
    assert ready.env.continuation().stagnation["count"] == 4


def test_shield_hold_elapsed_is_not_charge_progress(ready):
    for start in range(0, 24, 6):
        for elapsed in range(start + 1, start + 7):
            ready.fake.results.append(
                raw(action_progress=use_signal("BLOCK", elapsed, 72000, 0))
            )
        state = ready.step("use")
    assert not state.mask[ready.index("use")]


@pytest.mark.parametrize(
    "missing", ["action_progress", "version", "mining", "item_use"]
)
def test_missing_telemetry_fails_clearly(missing):
    info = {"action_progress": idle_progress()}
    (info if missing == "action_progress" else info["action_progress"]).pop(missing)
    with pytest.raises(RuntimeError, match="telemetry missing or invalid"):
        action_progress(info)


def test_handler_survives_real_minestudio_processing_without_ticks():
    import json
    from types import SimpleNamespace

    from minestudio.simulator.minerl.env._multiagent import _MultiAgentEnv

    from mcpref.runtime.env.minestudio import EndlessHumanSurvival

    task = EndlessHumanSurvival(resolution=(2, 2))
    task.reset()
    env = SimpleNamespace(task=task, _last_pov={}, _last_obs={})
    info = {
        "action_progress": idle_progress(),
        "xpos": 0,
        "ypos": 0,
        "zpos": 0,
        "yaw": 0,
        "pitch": 0,
        "isGuiOpen": False,
        "inventory": [],
    }
    import numpy as np

    obs, _ = _MultiAgentEnv._process_observation(
        env, "agent", np.zeros((2, 2, 3), dtype=np.uint8), json.dumps(info)
    )
    assert obs["action_progress"] == action_progress(info)


def test_client_remaining_can_be_negative_before_server_completion():
    signal = use_signal("EAT", elapsed=33, duration=32, charge=32)
    parsed = action_progress({"action_progress": signal})
    assert parsed["item_use"]["remaining"] == -1
    assert parsed["item_use"]["elapsed"] == 33
