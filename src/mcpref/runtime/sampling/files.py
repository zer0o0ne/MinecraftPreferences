"""Run fingerprints and crash-repaired counts of specs/env-sampling.md."""

from pathlib import Path

import torch
import yaml

from mcpref.config import REPO, config_sha256
from mcpref.datasets.clips.config import load_config as load_clips_config
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.infrastructure.runs.files import check_fingerprints, sha256, write_json
from mcpref.infrastructure.shards.store import complete_shards, name, shard_path
from mcpref.runtime.env.config import load_config as load_env_config
from mcpref.runtime.sampling.config import SamplingConfig
from mcpref.training.pretrain.config import load_config as load_pretrain_config


def config_paths(cfg: SamplingConfig, path: Path) -> dict[str, Path]:
    """The ordered transitive config/model identities of env-sampling."""
    pretrain = load_pretrain_config(REPO / cfg.pretrain)
    env = load_env_config(REPO / cfg.env)
    clips = load_clips_config(REPO / env.clips)
    mask = load_mask_config(REPO / pretrain.action_mask)
    return {
        "config": path,
        "pretrain_config": REPO / cfg.pretrain,
        "pretrain_model": REPO / pretrain.out_dir / "model.pt",
        "env": REPO / cfg.env,
        "mcts": REPO / cfg.mcts,
        "perception_data": REPO / cfg.perception_data,
        "perception": REPO / pretrain.perception,
        "action_mask": REPO / pretrain.action_mask,
        "policy": REPO / pretrain.policy,
        "preference": REPO / pretrain.preference,
        "engine": REPO / env.engine,
        "classifier": REPO / env.classifier,
        "clips": REPO / env.clips,
        "vpt_actions": clips.actions,
        "gui_screens": REPO / mask.screens,
    }


class SamplingRun:
    """Datasets: the ordered static fingerprints and per-shard model mapping."""

    def __init__(self, cfg: SamplingConfig, path: Path) -> None:
        self.path = REPO / cfg.out_dir / "run.json"
        hashes = {
            key: sha256(value) if key == "pretrain_model" else config_sha256(value)
            for key, value in config_paths(cfg, path).items()
        }
        check_fingerprints(self.path, hashes, cfg.out_dir)
        models = {}
        shards = REPO / cfg.out_dir / "shards"
        count = complete_shards(shards)
        for k in range(count):
            state = torch.load(
                shard_path(shards, k) / "state.pt",
                map_location="cpu",
                weights_only=True,
            )
            models[name(k)] = state["fingerprint"]
        self.value = {
            "config": yaml.safe_load(path.read_text()),
            "sha256": hashes,
            "models": models,
            "counts": {"trajectories": count, "steps": count * cfg.steps},
        }
        write_json(self.path, self.value)

    def published(self, k: int, next_step: int, fingerprint: str) -> None:
        """Datasets: add one published shard and rewrite the run atomically."""
        self.value["models"][name(k)] = fingerprint
        self.value["counts"] = {"trajectories": k + 1, "steps": next_step}
        write_json(self.path, self.value)
