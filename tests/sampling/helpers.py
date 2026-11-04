"""Temporary repository and deterministic fake env/networks for env-sampling tests."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from mcts.helpers import FakePolicy, FakePreference
from torch import nn
from transformers import DynamicCache, PretrainedConfig

from mcpref.env.state import EnvState
from mcpref.sampling.config import SamplingConfig, load_config
from mcpref.sampling.models import SamplingModules

ACTIONS = 3


def state(index: int) -> EnvState:
    return EnvState(
        frame=np.full((1, 2, 3), index, dtype=np.uint8),
        line={"id": index},
        gui=False,
        cls=None,
        hotbar=0,
        mask=np.ones(ACTIONS, dtype=bool),
    )


class FakeEnv:
    def __init__(self, events=None):
        self.index = 0
        self.resets = 0
        self.events = [] if events is None else events

    def reset(self):
        self.resets += 1
        self.events.append(("reset", self.index))
        return state(self.index)

    def step(self, action: int):
        self.events.append(("step", action))
        self.index += 1
        return state(self.index)


class FakePhi(nn.Module):
    def __init__(self, offset=0.0, events=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.head = nn.Sequential(nn.Linear(2, 2, dtype=torch.float64))
        self.backbone = nn.Module()
        self.backbone.config = PretrainedConfig(
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=4,
        )
        self.offset = offset
        self.events = [] if events is None else events
        self.grad_modes = []

    def latent(self, index: int) -> torch.Tensor:
        return torch.tensor(
            [index + self.offset, index + 0.5 + self.offset],
            dtype=self.weight.dtype,
            device=self.weight.device,
        )


class FakeStream:
    def __init__(self, phi: FakePhi, states: int):
        self.phi, self.states = phi, states
        self.ids, self.grids = [], []
        self.cache = DynamicCache(config=phi.backbone.config)
        self.first = None

    def step(self, frame: np.ndarray, line: dict) -> torch.Tensor:
        index = line["id"]
        if len(self.grids) == self.states:
            self.ids, self.grids = [], []
            self.cache = DynamicCache(config=self.phi.backbone.config)
            self.first = None
        if self.first is None:
            self.first = line
        self.phi.events.append(("stream", index))
        with torch.no_grad():
            self.phi.grad_modes.append(torch.is_grad_enabled())
            self.ids.append(index)
            self.grids.append(torch.tensor([[1, 1, 1]]))
            return self.phi.latent(index)


class Policy(nn.Module, FakePolicy):
    def __init__(self, events=None):
        nn.Module.__init__(self)
        FakePolicy.__init__(
            self,
            logits=[0.0] * ACTIONS,
            shifts=torch.tensor(
                [[0.01, 0.02], [0.1, 0.2], [0.2, -0.1]], dtype=torch.float64
            ),
        )
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.names = [f"action_{index}" for index in range(ACTIONS)]
        self.events = [] if events is None else events

    def readout(self, Y, C):
        self.events.append(("readout", Y[0, -1, 0].item()))
        return super().readout(Y, C)


class Preference(FakePreference, nn.Module):
    def __init__(self, events=None):
        nn.Module.__init__(self)
        FakePreference.__init__(
            self,
            logits_fn=lambda Ya, Yb: (
                Ya[0, -1, 0] - Yb[0, -1, 0],
                Ya[0, -1, 1] + Yb[0, -1, 1],
            ),
        )
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.events = [] if events is None else events

    def __call__(self, Ya, Yb):
        self.events.append(("preference", Ya[0, -1, 0].item(), Yb[0, -1, 0].item()))
        return super().__call__(Ya, Yb)


def modules(fingerprint="fake-model", events=None, offset=0.0) -> SamplingModules:
    return SamplingModules(
        phi=FakePhi(offset, events),
        pi=Policy(events),
        P=Preference(events),
        fingerprint=fingerprint,
    )


class Site:
    def __init__(self, tmp: Path, repo: Path, steps=2, states=2):
        self.root = tmp / "repo"
        shutil.copytree(repo / "configs", self.root / "configs")
        self._edit("clips.yaml", delta=5)
        self._edit("perception.yaml", alpha=[1.0, 2.0])
        self._edit("perception-data.yaml", states=states)
        self._edit("pretrain.yaml", device="cpu")
        self._edit(
            "mcts.yaml",
            simulations=1,
            min_n_sampling=1,
            depth=1,
            width=3,
            latency_calls=1,
        )
        self.mapping = {
            "pretrain": "configs/pretrain.yaml",
            "env": "configs/env.yaml",
            "mcts": "configs/mcts.yaml",
            "perception_data": "configs/perception-data.yaml",
            "out_dir": "out",
            "steps": steps,
            "seed": 7,
            "safety_limit": 25.0,
        }
        self.path = tmp / "env-sampling.yaml"
        self.path.write_text(yaml.safe_dump(self.mapping, sort_keys=False))
        self.cfg = load_config(self.path)
        self.out = self.root / "out"
        pretrain = self.root / "data/pretrain"
        pretrain.mkdir(parents=True)
        (pretrain / "run.json").write_text(json.dumps({"complete": True}))
        torch.save({"phi": {}, "pi": {}, "P": {}}, pretrain / "model.pt")
        self.events = []
        self.env = FakeEnv(self.events)
        self.modules = modules(events=self.events)

    def _edit(self, name: str, **changes) -> None:
        path = self.root / "configs" / name
        value = yaml.safe_load(path.read_text()) | changes
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    def config(self, **changes) -> SamplingConfig:
        values = dict(self.mapping) | changes
        return SamplingConfig(**values)
