"""Construction, initial timeline, Escape and public signatures of specs/env.md (1), (2)."""

import inspect

from progress_bars import Bar, assert_progress, screen, states

from mcpref.runtime.env.config import EnvConfig
from mcpref.runtime.env.minestudio import EndlessHumanSurvival
from mcpref.runtime.env.state import EnvState
from mcpref.runtime.env.wrapper import Env
from runtime.env.conftest import COMMANDS
from runtime.env.fake_simulator import raw


def test_constructor_arguments_and_engine_order(world):
    kwargs = {
        "action_type": "env",
        "obs_size": (640, 360),
        "render_size": (640, 360),
        "seed": 7,
    }
    assert world.fake.init_kwargs == kwargs
    names = [event[0] for event in world.events]
    assert names.index("use_engine") < names.index("MinecraftSim")
    args = world.classifier.load_calls[0]
    assert args[1:] == ("configs/gui/screens.yaml", 640, 360)
    assert world.fake.commands == [] and world.fake.reset_calls == 0


def test_initial_options_commands_and_no_later_reset(world, ready):
    assert ready.fake.reset_calls == 1
    assert ready.fake.commands == COMMANDS
    options = ready.root / "ms/runtime/options.txt"
    assert options.read_bytes() == b"autoJump:true\n"
    assert list(options.parent.iterdir()) == [options]
    ready.step("turn_left")
    assert ready.fake.reset_calls == 1 and ready.fake.commands == COMMANDS


def test_initial_state_is_last_command_result(world):
    final = raw(pos={"x": 9, "y": 8, "z": 7, "yaw": 6, "pitch": 5})
    world.fake.reset_raw = final
    state = world.reset()
    assert state.line["xpos"] == 9.0 and state.frame is final[0]["image"]


def test_env_init_bar(world, capsys):
    world.reset()
    out, err = capsys.readouterr()
    assert out == ""
    assert_progress(err)
    assert states(err, "env init")[0] == Bar("env init", 0, 5, "command")
    assert screen(err) == [Bar("env init", 5, 5, "command")]


def test_close_delegates_once(ready):
    assert ready.env.close() is None
    assert ready.fake.close_calls == 1


def test_no_jar_mutation(ready):
    jar = ready.root / "ms/engine/build/libs/mcprec-6.13.jar"
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"sentinel jar")
    before = jar.read_bytes()
    ready.step("attack")
    ready.env.close()
    second = Env(
        EnvConfig(
            "configs/runtime/engine.yaml",
            "configs/gui/recognition/classifier.yaml",
            "configs/gui/action-mask.yaml",
            "configs/datasets/clips.yaml",
            20 / 3,
        )
    )
    try:
        assert jar.read_bytes() == before
    finally:
        second.close()


def test_escape_actionable_and_no_quit_producers(monkeypatch):
    from minestudio.simulator.entry import HumanSurvival
    from minestudio.simulator.minerl.herobraine.hero import handlers

    monkeypatch.setattr(HumanSurvival, "create_actionables", lambda self: ["ordinary"])
    survival = object.__new__(EndlessHumanSurvival)
    actions = survival.create_actionables()
    assert actions[0] == "ordinary" and len(actions) == 2
    assert isinstance(actions[1], handlers.KeybasedCommandAction)
    assert actions[1].to_string() == "ESC"
    assert survival.create_server_quit_producers() == []


def test_human_survival_replaced_only_during_construction(world):
    from minestudio.simulator.entry import HumanSurvival

    assert world.humans == [EndlessHumanSurvival]
    assert HumanSurvival is not EndlessHumanSurvival
    assert issubclass(EndlessHumanSurvival, HumanSurvival)


def test_env_public_signatures():
    assert list(inspect.signature(Env).parameters) == ["cfg"]
    assert list(inspect.signature(Env.__init__).parameters) == ["self", "cfg"]
    assert list(inspect.signature(Env.reset).parameters) == ["self"]
    assert list(inspect.signature(Env.step).parameters) == ["self", "action"]
    assert list(inspect.signature(Env.close).parameters) == ["self"]
    hints = inspect.get_annotations(Env.step, eval_str=True)
    assert hints["action"] is int and hints["return"] is EnvState
    assert inspect.get_annotations(Env.reset, eval_str=True)["return"] is EnvState
    assert inspect.get_annotations(Env.close, eval_str=True)["return"] is None
    assert inspect.get_annotations(Env.__init__, eval_str=True)["cfg"] is EnvConfig
