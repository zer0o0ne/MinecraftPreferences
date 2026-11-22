"""Tiny differentiable recurrence through the real sampler and RL orchestration."""

from types import SimpleNamespace

import torch
import yaml
from runtime.sampling.continuous_world import setup
from runtime.sampling.helpers import FakeEnv
from torch import nn
from transformers import PretrainedConfig

from mcpref.config import read_yaml
from mcpref.models.perception.continuous.state import ContinuousState, GDNState
from training.rl.cycle_world import prepare
from training.rl.learning.loss_fakes import modules


def build_modules(*args):
    built = modules()
    built["phi"].backbone = nn.Module()
    built["phi"].backbone.config = PretrainedConfig(layer_types=["linear_attention"])
    built["pi"].history, built["P"].history = 2, 5
    built["pi"].embed = SimpleNamespace(weight=built["pi"].read_weight)
    built["P"].embed = SimpleNamespace(weight=built["P"].weight)
    return built


def observation(phi, frame, line, state=None):
    old = 0 if state is None else state.layers[0].recurrent
    value = 0.9 * old + phi.scale * (line["id"] + 1)
    z = torch.stack((value, phi.scale * (0.5 * line["id"] - 1)))
    origin = dict(line) if state is None else state.origin
    return z, ContinuousState((GDNState(value.reshape(1), value),), origin)


def world(tmp_path, repo, monkeypatch, *, retrain):
    root, path = prepare(
        tmp_path,
        repo,
        retrain=retrain,
        continuous={"gradient_history": 2, "replay_stride": 2},
    )
    built = build_modules()
    site = SimpleNamespace(
        root=root,
        cfg=SimpleNamespace(pretrain="configs/training/pretrain/config.yaml"),
        modules=SimpleNamespace(**built),
    )
    setup(site, monkeypatch)
    mcts = root / "configs/runtime/mcts.yaml"
    mcts.write_text(
        yaml.safe_dump(
            read_yaml(mcts)
            | {
                "simulations": 1,
                "min_n_sampling": 1,
                "depth": 1,
                "width": 3,
                "latency_calls": 1,
            }
        )
    )
    live = []

    def sampled(phi, frame, line, state=None):
        live.append((line["id"], float(phi.scale.detach())))
        return observation(phi, frame, line, state)

    monkeypatch.setattr("mcpref.training.rl.cycle.build_rl_modules", build_modules)
    monkeypatch.setattr(
        "mcpref.models.perception.continuous.observation.observation_step", sampled
    )
    monkeypatch.setattr(
        "mcpref.training.rl.continuous.replay.observation_step", observation
    )
    return root, path, FakeEnv(), live


def saved(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert a.equal(b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b
