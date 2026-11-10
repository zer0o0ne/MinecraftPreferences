"""Fake-engine smoke main of specs/env.md Interface and Tests: sequence, errors, finally, bars."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from progress_bars import Bar, assert_aborted, assert_progress, screen, states

import mcpref.runtime.env.check as check_module
from mcpref.runtime.env.check import main
from mcpref.runtime.env.state import EnvState
from runtime.env.conftest import ACTIONS
from runtime.env.fake_simulator import FRAME, FakeSimulator

SEQUENCE = [
    "open_inventory", "close", "open_inventory", "left_click_1", "right_click_1",
    "shift_click_1", "double_click_1", "close", "attack", "use", "hotbar_1",
]  # fmt: skip


class FakeEnv:
    def __init__(self, cfg, events, fail):
        self.cfg, self.events, self.fail = cfg, events, fail
        self.sim = FakeSimulator()
        self.masks = SimpleNamespace(names=ACTIONS)
        self.actions = ACTIONS
        self.hotbar = 0

    def state(self):
        mask = np.ones(395, dtype=np.bool_)
        mask[ACTIONS.index("hotbar_1")] = False
        return EnvState(FRAME, {"hotbar": 0}, False, None, 0, mask)

    def reset(self):
        self.events.append("reset")
        if self.fail == "reset":
            raise RuntimeError("boom")
        return self.state()

    def step(self, action):
        self.events.append(action)
        if self.fail == len([e for e in self.events if isinstance(e, int)]):
            raise RuntimeError("boom")
        return self.state()

    def _command(self, command):
        self.events.append(command)
        if self.fail == "kill":
            raise RuntimeError("boom")
        return self.state()

    def close(self):
        self.events.append("close")
        if self.fail == "close":
            raise RuntimeError("boom")


@pytest.fixture
def smoke(monkeypatch, capsys):
    def run(fail=None):
        events = []
        holder = {}

        def construct(cfg):
            holder["env"] = FakeEnv(cfg, events, fail)
            return holder["env"]

        monkeypatch.setattr(check_module, "Env", construct)
        monkeypatch.setattr(check_module, "load_config", lambda path: ("cfg", path))
        code = main(["configs/runtime/env.yaml"])
        out, err = capsys.readouterr()
        return code, out, err, events

    return run


def expected_events():
    return ["reset", *(ACTIONS.index(a) for a in SEQUENCE[:10]), "/kill",
            ACTIONS.index("hotbar_1"), "close"]  # fmt: skip


def test_smoke_sequence_and_output(smoke):
    code, out, err, events = smoke()
    assert (code, out) == (0, "env: ok\n")
    assert events == expected_events()
    assert_progress(err)
    assert states(err, "env check")[0] == Bar("env check", 0, 12, "action")
    assert screen(err) == [Bar("env check", 12, 12, "action")]


@pytest.mark.parametrize("fail", ["reset", 1, 5, 10, 11, "kill"])
def test_every_failure_reports_and_closes(smoke, fail):
    code, out, err, events = smoke(fail)
    assert out == "" and events[-1] == "close"
    assert_aborted(code, err)
    assert err.splitlines()[-1] == "mcpref.runtime.env.check: RuntimeError: boom"
    assert "env: ok" not in out


def test_close_failure_is_reported(smoke):
    code, out, err, events = smoke("close")
    assert (code, out, events[-1]) == (1, "", "close")
    assert err.splitlines()[-1] == "mcpref.runtime.env.check: RuntimeError: boom"


@pytest.mark.parametrize("argv", [[], ["a", "b"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert "usage: python -m mcpref.runtime.env.check" in capsys.readouterr().err


def test_main_public_signature():
    from collections.abc import Sequence

    signature = inspect.signature(main)
    assert list(signature.parameters) == ["argv"]
    hints = inspect.get_annotations(main, eval_str=True)
    assert hints["argv"] == Sequence[str] and hints["return"] is int
