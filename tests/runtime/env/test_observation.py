"""GUI first-line rule, automatic Escape, masks and unrecoverable ends of specs/env.md (4), (10)."""

import numpy as np
import pytest

from mcpref.gui.mask import state_mask
from runtime.env.conftest import patch_public
from runtime.env.fake_simulator import assert_actions, expected, raw

IDLE = expected()


def test_use_first_raw_opening_is_hidden_when_it_is_the_final_state(ready):
    ready.fake.results.extend([raw()] * 5 + [raw(gui=True)])
    state = ready.step("use")
    assert ready.fake.current[4]["isGuiOpen"] is True
    assert (state.gui, state.cls) == (False, None)
    assert ready.classifier.calls == []
    assert state.mask[ready.index("attack")]
    assert not state.mask[ready.index("close")]


def test_any_later_raw_gui_state_is_classified_and_returned(ready):
    ready.fake.results.append(raw(gui=True))
    state = ready.step("use")
    assert (state.gui, state.cls) == (True, "inventory")
    assert len(ready.classifier.calls) == 1
    assert state.mask[ready.index("close")]
    assert not state.mask[ready.index("attack")]


def test_classifier_precedes_every_gui_mask(world, monkeypatch):
    world.fake.reset_raw = raw(gui=True)
    calls = []

    def checked(*args):
        calls.append((len(world.classifier.calls), args[1:]))
        return state_mask(*args)

    patch_public(monkeypatch, "state_mask", checked)
    world.reset()
    assert calls == [(1, (True, "inventory", 5.0, 0))]


@pytest.mark.parametrize("cls", ["other", "pause"])
def test_automatic_escape_extends_beyond_delta_and_mask_waits(
    ready, cls, monkeypatch, capsys
):
    ready.classifier.labels.extend([cls] * 4)
    ready.fake.results.extend(
        [
            *([raw()] * 5),
            raw(gui=True),
            raw(gui=True),
            raw(gui=True),
            raw(gui=True),
            raw(),
        ]
    )
    masks = []

    def checked(*args):
        masks.append(args[1:3])
        return state_mask(*args)

    patch_public(monkeypatch, "state_mask", checked)
    capsys.readouterr()
    camera = np.array([0, 10], dtype=np.float32)
    state = ready.step("turn_right")
    ordinary = [expected(camera=camera), *([IDLE] * 5)]
    escape = [expected(ESC=1), IDLE, expected(ESC=1), IDLE]
    assert_actions(ready.fake, [*ordinary, *escape])
    assert (state.gui, state.cls) == (False, None)
    assert masks and set(masks) == {(False, None)}
    assert ready.classifier.calls
    assert capsys.readouterr() == ("", "")


def test_automatic_escape_on_initial_conversion(world):
    world.fake.reset_raw = raw(gui=True)
    world.classifier.labels.append("other")
    world.fake.results.extend([raw(gui=True), raw()])
    state = world.reset()
    assert_actions(world.fake, [expected(ESC=1), IDLE])
    assert (state.gui, state.cls) == (False, None)


@pytest.mark.parametrize(
    "ending", [{"terminated": True}, {"truncated": True}, {"error": "boom"}]
)
def test_every_episode_end_aborts(ready, ending):
    ready.fake.results.append(raw(**ending))
    with pytest.raises(RuntimeError) as error:
        ready.step("forward")
    assert str(error.value) == "episode ended"


def test_reset_info_error_aborts(world):
    world.fake.reset_raw = raw(error="boom")
    with pytest.raises(RuntimeError) as error:
        world.reset()
    assert str(error.value) == "episode ended"


def test_no_real_time_sleep(ready, monkeypatch):
    monkeypatch.setattr(
        "time.sleep", lambda _: pytest.fail("unexpected real-time sleep")
    )
    ready.step("forward")
    assert len(ready.fake.actions) == 6
