"""Temporary repository, fake classifier and fake simulator for the black-box Env tests."""

import shutil
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from pretrain.index_run import repo_root
from progress_bars import tqdm_defaults

from mcpref.env.config import EnvConfig
from mcpref.env.wrapper import Env
from mcpref.vpt.actions import WORLD_ACTIONS

from .fake_simulator import FakeSimulator

ACTIONS = (
    *WORLD_ACTIONS,
    "close",
    *(f"{kind}_{i}" for kind in ("left_click", "right_click", "shift_click", "double_click")
      for i in range(1, 91)),
)  # fmt: skip
COMMANDS = [
    "/gamerule doImmediateRespawn true",
    "/gamerule sendCommandFeedback false",
    "/difficulty medium",
    "/gamerule keepInventory false",
    "/gamerule doDaylightCycle true",
]


class FakeClassifier:
    def __init__(self):
        self.labels = deque()
        self.calls = []
        self.load_calls = []

    def __call__(self, frames):
        self.calls.append(frames)
        label = self.labels.popleft() if self.labels else "inventory"
        return [label] * len(frames)


@dataclass
class World:
    env: Env
    fake: FakeSimulator
    classifier: FakeClassifier
    root: Path
    events: list
    humans: list = field(default_factory=list)

    def reset(self):
        return self.env.reset()

    def index(self, name):
        return ACTIONS.index(name)

    def step(self, name):
        return self.env.step(self.index(name))


def patch_public(monkeypatch, name, value) -> None:
    modules = [
        "mcpref.engine.config",
        "mcpref.gui.mask",
        "mcpref.recognition.classifier",
        "mcpref.env.minestudio",
        "mcpref.env.state",
        "mcpref.env.wrapper",
    ]
    for module_name in modules:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, name):
            monkeypatch.setattr(module, name, value)


def make_repository(root: Path, repo: Path, delta: int = 6) -> None:
    (root / "configs").mkdir(parents=True)
    for name in (
        "action-mask.yaml",
        "gui-classifier.yaml",
        "gui-screens.yaml",
        "vpt-actions.yaml",
    ):
        shutil.copy2(repo / "configs" / name, root / "configs" / name)
    engine = yaml.safe_load((repo / "configs/engine.yaml").read_text())
    engine.update(minestudio_dir="ms", render_size=[640, 360], seed=7)
    (root / "configs/engine.yaml").write_text(yaml.safe_dump(engine, sort_keys=False))
    (root / "configs/clips.yaml").write_text(
        yaml.safe_dump({"delta": delta, "actions": "configs/vpt-actions.yaml"})
    )


@pytest.fixture
def env_root(tmp_path, monkeypatch, repo):
    root, cwd = tmp_path / "repo", tmp_path / "cwd"
    cwd.mkdir()
    make_repository(root, repo)
    monkeypatch.chdir(cwd)
    tqdm_defaults(monkeypatch)
    with repo_root(root):
        yield root


@pytest.fixture
def world(env_root, monkeypatch):
    root = env_root
    fake, classifier, events, humans = FakeSimulator(), FakeClassifier(), [], []

    def use_engine(cfg):
        events.append(("use_engine", cfg))

    def load_classifier(*args):
        classifier.load_calls.append(args)
        events.append(("load_classifier", args))
        return classifier, "fake-classifier"

    with repo_root(root):
        patch_public(monkeypatch, "use_engine", use_engine)
        patch_public(monkeypatch, "load_classifier", load_classifier)
        import minestudio.simulator
        from minestudio.simulator import entry

        def construct(**kwargs):
            events.append(("MinecraftSim", kwargs))
            humans.append(entry.HumanSurvival)
            fake.init_kwargs = kwargs
            return fake

        monkeypatch.setattr(entry, "MinecraftSim", construct)
        monkeypatch.setattr(minestudio.simulator, "MinecraftSim", construct)
        patch_public(monkeypatch, "MinecraftSim", construct)
        cfg = EnvConfig(
            "configs/engine.yaml",
            "configs/gui-classifier.yaml",
            "configs/action-mask.yaml",
            "configs/clips.yaml",
            20 / 3,
        )
        env = Env(cfg)
        yield World(env, fake, classifier, root, events, humans)


@pytest.fixture
def ready(world):
    world.reset()
    return world
