"""Fake-CPU R1 environment/sampler around the real RL cycle and artifact producers."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from perception.world import line
from sampling.helpers import state

from mcpref.sampling.shards import StepRecord, write_step

from .helpers import config_path, repository, write_json
from .loss_fakes import modules


def env_state(index):
    raw = line(xpos=float(index), ypos=64.0, zpos=-float(index)) | {"id": index}
    return dataclasses.replace(
        state(index), frame=np.full((1, 2, 3), index + 1, dtype=np.uint8), line=raw
    )


def record(index):
    return StepRecord(
        index=index,
        before=env_state(index),
        after=env_state(index + 1),
        action=index % 3,
        probabilities=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        delta=torch.full((2,), torch.nan, dtype=torch.float64),
        predicted=torch.full((2,), torch.nan, dtype=torch.float64),
        elapsed_ms=0.0,
        preference_close={"succ": 0.0, "inc": 0.0},
        preference_start={"succ": 0.0, "inc": 0.0},
        tree={},
    )


class Env:
    def __init__(self, events):
        self.events = events
        self.index = 0

    def reset(self):
        self.events.append(("env-reset", self.index))
        return env_state(self.index)

    def step(self, action):
        self.events.append(("env-step", self.index, action))
        self.index += 1
        return env_state(self.index)


class Sampler:
    def __init__(self, cfg, env, supplied=None):
        self.cfg, self.env = cfg, env
        self.supplied = supplied
        self.root = Path(cfg.out_dir)
        if not self.root.is_absolute():
            import mcpref

            self.root = mcpref.REPO / self.root
        self.events = env.events
        fingerprint = getattr(supplied, "fingerprint", None)
        self.events.append(("sampler-init", fingerprint))

    def reset(self):
        self.events.append(("sampler-reset", self.env.index))

    def sample(self):
        cycle = self.env.index // self.cfg.steps
        shard = self.root / "shards" / f"{cycle:06d}"
        shard.mkdir(parents=True, exist_ok=True)
        for j in range(self.cfg.steps):
            index = cycle * self.cfg.steps + j
            write_step(shard, j, record(index))
            self.env.step(index % 3)
        torch.save({"next_step": (cycle + 1) * self.cfg.steps}, shard / "state.pt")
        write_json(
            self.root / "run.json",
            {
                "config": {},
                "sha256": {},
                "counts": {
                    "trajectories": cycle + 1,
                    "steps": (cycle + 1) * self.cfg.steps,
                },
                "models": {},
            },
        )
        self.events.append(("sampled", cycle))
        return shard


def prepare(tmp_path, repo, **changes):
    root = tmp_path / "repo"
    repository(root, repo)
    initial = modules()
    model = {name: initial[name].state_dict() for name in ["phi", "pi", "P"]}
    pretrain = root / "data/pretrain-scaled-init"
    pretrain.mkdir(parents=True)
    write_json(pretrain / "run.json", {"complete": True})
    torch.save(model, pretrain / "model.pt")
    values = {"batch": 2, "items_per_shard": 3, "epochs": 1,
              "checkpoint_steps": 1, "keep_checkpoints": 2} | changes  # fmt: skip
    path = config_path(tmp_path / "rl.yaml", **values)
    return root, path


def build_modules(*args, **kwargs):
    return modules()


def install(monkeypatch, events, batches):
    monkeypatch.setattr("mcpref.rl.cycle.build_modules", build_modules)
    monkeypatch.setattr("mcpref.rl.cycle.Sampler", Sampler)

    def gradients(items, modules, **kwargs):
        batches.append([dict(item) for item in items])
        draw = torch.rand((), dtype=torch.float64)
        modules["phi"].scale.grad = draw.clone()
        value = float(draw)
        return {"ppo": value, "shift": value, "metra": value, "P": value, "loss": value}

    monkeypatch.setattr("mcpref.rl.loop.step_gradients", gradients)


def tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file() and path.suffix != ".part"}  # fmt: skip


def run(root, path, env):
    from pretrain.index_run import repo_root

    from mcpref.rl.config import load_config
    from mcpref.rl.cycle import run_cycle

    with repo_root(root):
        return run_cycle(load_config(path), path, env)


def read(path):
    return json.loads(path.read_text())
