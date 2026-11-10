"""Fresh-process real-Env recovery and unfinished holds of env-sampling.md Tests."""

import copy
import multiprocessing
import shutil
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from env.conftest import ACTIONS, FakeClassifier, patch_public
from env.fake_simulator import FakeSimulator
from pretrain.index_run import repo_root

from mcpref.env.config import load_config as load_env_config
from mcpref.env.wrapper import Env
from mcpref.sampling.runner import Sampler
from mcpref.sampling.shards import load_state, load_step

from .helpers import modules
from .test_recovery import assert_record, assert_value


def assert_state(actual, expected):
    np.testing.assert_array_equal(actual.frame, expected.frame)
    assert actual.line == expected.line
    assert (actual.gui, actual.cls, actual.hotbar) == (
        expected.gui,
        expected.cls,
        expected.hotbar,
    )
    np.testing.assert_array_equal(actual.mask, expected.mask)


def fresh_simulator(boundary):
    fake = FakeSimulator()
    fake.current = copy.deepcopy(boundary.current)
    fake.results = copy.deepcopy(boundary.results)
    return fake


def recover_in_child(root, cfg, fake, fingerprint, offset, action, connection):
    try:
        with repo_root(root), pytest.MonkeyPatch.context() as monkeypatch:
            classifier = FakeClassifier()

            def construct(**kwargs):
                return fake

            def load_classifier(*args):
                return classifier, "fake-classifier"

            import minestudio.simulator
            from minestudio.simulator import entry

            monkeypatch.setattr(entry, "MinecraftSim", construct)
            monkeypatch.setattr(minestudio.simulator, "MinecraftSim", construct)
            patch_public(monkeypatch, "MinecraftSim", construct)
            patch_public(monkeypatch, "load_classifier", load_classifier)
            env = Env(load_env_config(root / cfg.env))
            sampler = Sampler(
                cfg,
                env,
                modules(
                    fingerprint,
                    offset=offset,
                    actions=len(ACTIONS),
                    preferred=action,
                ),
            )
            sampler.reset()
            restore_counters = (
                fake.reset_calls,
                list(fake.commands),
                len(fake.actions),
                fake.noop_calls,
                len(classifier.calls),
            )
            sampler.sample()
            connection.send(
                {
                    "restore_counters": restore_counters,
                    "trace": copy.deepcopy(fake.actions),
                }
            )
            env.close()
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        connection.close()


@pytest.fixture
def real_world(site, monkeypatch):
    fake, classifier = FakeSimulator(), FakeClassifier()
    patch_public(monkeypatch, "use_engine", lambda cfg: None)
    patch_public(
        monkeypatch,
        "load_classifier",
        lambda *args: (classifier, "fake-classifier"),
    )
    import minestudio.simulator
    from minestudio.simulator import entry

    def construct(**kwargs):
        return fake

    monkeypatch.setattr(entry, "MinecraftSim", construct)
    monkeypatch.setattr(minestudio.simulator, "MinecraftSim", construct)
    patch_public(monkeypatch, "MinecraftSim", construct)
    env = Env(load_env_config(site.root / site.cfg.env))
    yield SimpleNamespace(env=env, fake=fake, classifier=classifier)
    env.close()


@pytest.mark.parametrize("changed", [False, True], ids=["unchanged", "changed"])
@pytest.mark.parametrize("next_name", ["attack", "forward"], ids=["repeat", "change"])
def test_fresh_process_real_env_recovery_exact_with_unfinished_hold(
    site, real_world, changed, next_name
):
    cfg = site.config(steps=1)
    mcts_path = site.root / cfg.mcts
    mcts = yaml.safe_load(mcts_path.read_text())
    mcts.update(simulations=2, min_n_sampling=2)
    mcts_path.write_text(yaml.safe_dump(mcts, sort_keys=False))
    attack, next_action = ACTIONS.index("attack"), ACTIONS.index(next_name)
    initial_modules = modules(actions=len(ACTIONS), preferred=attack)
    sampler = Sampler(cfg, real_world.env, initial_modules)
    sampler.reset()
    sampler.sample()
    previous = load_step(site.out, 0)
    boundary = load_state(site.out, initial_modules.phi, 2, torch.device("cpu"))
    assert boundary.env.held_action == attack
    assert boundary.observation.line == previous.after.line
    complete = site.out / "shards/000000"
    complete_bytes = {
        path.relative_to(complete): path.read_bytes()
        for path in complete.rglob("*")
        if path.is_file()
    }
    run_bytes = (site.out / "run.json").read_bytes()
    child_fake = fresh_simulator(real_world.fake)

    fingerprint = "changed-model" if changed else "fake-model"
    offset = 100.0 if changed else 0.0
    control = Sampler(
        cfg,
        real_world.env,
        modules(
            fingerprint,
            offset=offset,
            actions=len(ACTIONS),
            preferred=next_action,
        ),
    )
    control.reset()
    trace_start = len(real_world.fake.actions)
    control.sample()
    expected_record = load_step(site.out, 1)
    expected_state = torch.load(
        site.out / "shards/000001/state.pt", map_location="cpu", weights_only=True
    )
    expected_trace = copy.deepcopy(real_world.fake.actions[trace_start:])
    shutil.rmtree(site.out / "shards/000001")
    shutil.rmtree(site.root / "data/diagnostics")
    (site.out / "run.json").write_bytes(run_bytes)

    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=recover_in_child,
        args=(
            site.root,
            cfg,
            child_fake,
            fingerprint,
            offset,
            next_action,
            send,
        ),
    )
    process.start()
    send.close()
    payload = receive.recv()
    process.join(timeout=10)
    assert process.exitcode == 0, payload.get("error")
    assert "error" not in payload, payload["error"]
    assert payload["restore_counters"] == (0, [], 0, 0, 0)

    actual = load_step(site.out, 1)
    assert_state(actual.before, previous.after)
    assert_record(actual, expected_record)
    actual_state = torch.load(
        site.out / "shards/000001/state.pt", map_location="cpu", weights_only=True
    )
    assert_value(actual_state, expected_state)
    assert_value(payload["trace"], expected_trace)
    after_bytes = {
        path.relative_to(complete): path.read_bytes()
        for path in complete.rglob("*")
        if path.is_file()
    }
    assert after_bytes == complete_bytes
