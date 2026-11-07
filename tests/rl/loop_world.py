"""A five-item, two-epoch RL training world with deterministic fake gradients."""

import contextlib
import io

import torch
from pretrain.index_run import repo_root

from mcpref.rl.loop import train

from .helpers import config, repository, write_json


def write_world(tmp_path, repo, **changes):
    root = tmp_path / "repo"
    repository(root, repo)
    values = {"epochs": 2, "batch": 2, "items_per_shard": 3,
              "checkpoint_steps": 1, "keep_checkpoints": 20} | changes  # fmt: skip
    cfg = config(tmp_path / "rl.yaml", **values)
    index = root / "data/rl_cycle/000000/index"
    shards = [
        [
            {"kind": "fresh", "index": 0},
            {"kind": "preference", "index": 10},
            {"kind": "fresh", "index": 1},
        ],
        [{"kind": "pending", "index": 2}, {"kind": "preference", "index": 11}],
    ]
    for k, items in enumerate(shards):
        write_json(index / "shards" / f"{k:06d}" / "items.json", {"items": items})
    write_json(
        index / "run.json",
        {
            "complete": True,
            "counts": {"fresh": 2, "pending": 1, "ppo": 2, "preference": 2, "items": 5},
            "batches": 3,
            "steps": 3 * cfg.epochs,
            "sha256": "fake",
        },
    )
    return root, cfg, index


def write_unit_world(tmp_path, repo, cycle=0, epochs=1, **changes):
    root = tmp_path / "repo"
    if not (root / "configs").exists():
        repository(root, repo)
    values = {"epochs": epochs, "batch": 1, "items_per_shard": 1,
              "checkpoint_steps": 1, "keep_checkpoints": 20} | changes  # fmt: skip
    cfg = config(tmp_path / f"rl-{cycle}.yaml", **values)
    index = root / f"data/rl_cycle/{cycle:06d}/index"
    write_json(
        index / "shards/000000/items.json",
        {"items": [{"kind": "fresh", "index": 2 * cycle}]},
    )
    write_json(
        index / "run.json",
        {
            "complete": True,
            "counts": {"fresh": 1, "pending": 0, "ppo": 1, "preference": 0, "items": 1},
            "batches": 1,
            "steps": epochs,
            "sha256": "fake",
        },
    )
    return root, cfg, index


def install_steps(monkeypatch, calls, updates=None, all_parameters=False):
    def fake(items, modules, **kwargs):
        draw = torch.rand((), dtype=torch.float64)
        calls.append(([dict(item) for item in items], float(draw)))
        targets = [
            parameter
            for module in modules.values()
            for parameter in module.parameters()
        ]
        if not all_parameters:
            targets = [modules["phi"].scale]
        for k, parameter in enumerate(targets):
            parameter.grad = torch.full_like(parameter, draw + 0.01 * k)
        value = float(draw)
        return {"ppo": value, "shift": value + 1, "metra": value + 2,
                "P": value + 3, "loss": value + 4}  # fmt: skip

    monkeypatch.setattr("mcpref.rl.loop.step_gradients", fake)
    if updates is not None:
        from mcpref.pretrain.update import update

        def counted(*args, **kwargs):
            updates.append(len(updates) + 1)
            return update(*args, **kwargs)

        monkeypatch.setattr("mcpref.rl.loop.update", counted)


def run_train(root, cfg, index, mods, initial_optimizer=None):
    err, out = io.StringIO(), io.StringIO()
    with (
        repo_root(root),
        contextlib.redirect_stderr(err),
        contextlib.redirect_stdout(out),
    ):
        train(cfg, mods, index, initial_optimizer)
    return err.getvalue(), out.getvalue()


def model(index):
    return torch.load(index.parent / "model.pt", map_location="cpu", weights_only=True)


def checkpoints(index):
    return sorted((index.parent / "checkpoints").glob("*/checkpoint.pt"))


def losses(index):
    import json

    return [json.loads(path.read_text())
            for path in sorted(index.parent.glob("checkpoints/*/losses.json"))]  # fmt: skip
