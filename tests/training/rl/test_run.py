"""CLI contract: construct one Env, run exactly one cycle, and close it in finally."""

import mcpref.training.rl.run as run_module
from training.rl.learning.helpers import config_path


class Env:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


def test_main_runs_one_cycle_and_closes_cli_owned_env(tmp_path, monkeypatch):
    env = Env()
    calls = []
    monkeypatch.setattr(run_module, "Env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        run_module,
        "run_cycle",
        lambda cfg, path, given: calls.append((cfg, path, given)),
    )
    path = config_path(tmp_path / "rl.yaml")
    assert run_module.main([str(path)]) == 0
    assert (
        len(calls) == 1 and calls[0][2] is env and calls[0][1] == tmp_path / "rl.yaml"
    )
    assert env.closed


def test_main_closes_env_when_the_cycle_raises(tmp_path, monkeypatch):
    env = Env()
    monkeypatch.setattr(run_module, "Env", lambda *args, **kwargs: env)

    def fail(*args):
        raise RuntimeError("stop")

    monkeypatch.setattr(run_module, "run_cycle", fail)
    path = config_path(tmp_path / "rl.yaml")
    assert run_module.main([str(path)]) == 1
    assert env.closed
