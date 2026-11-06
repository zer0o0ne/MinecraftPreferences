"""Temporary repositories and tiny configurations for specs/rl-cycle.md."""

import json
import shutil
from pathlib import Path

import yaml

from mcpref.rl.config import load_config

GROUPS = {
    "phi_vision": {"lr": 1.0e-5, "weight_decay": 0.01},
    "phi_vision_no_decay": {"lr": 1.0e-5, "weight_decay": 0.0},
    "phi_llm": {"lr": 1.0e-5, "weight_decay": 0.01},
    "phi_llm_no_decay": {"lr": 1.0e-5, "weight_decay": 0.0},
    "phi_head": {"lr": 1.0e-4, "weight_decay": 0.01},
    "phi_head_no_decay": {"lr": 1.0e-4, "weight_decay": 0.0},
    "metra_dual": {"lr": 1.0e-4, "weight_decay": 0.0},
    "pi": {"lr": 3.0e-4, "weight_decay": 0.01},
    "pi_no_decay": {"lr": 3.0e-4, "weight_decay": 0.0},
    "P": {"lr": 3.0e-4, "weight_decay": 0.01},
    "P_no_decay": {"lr": 3.0e-4, "weight_decay": 0.0},
}

BASE = {
    "sampling": "configs/env-sampling.yaml",
    "ppo": "configs/ppo.yaml",
    "rl_pairs": "configs/rl-preference-pairs.yaml",
    "out_dir": "data/rl_cycle",
    "seed": 0,
    "epochs": 1,
    "batch": 2,
    "micro_batch": 1,
    "items_per_shard": 3,
    "checkpoint_steps": 1,
    "keep_checkpoints": 2,
    "w_ppo": 1.0,
    "w_delta": 1.0,
    "w_metra": 1.0,
    "w_P": 1.0,
    "adam_eps": 1.0e-8,
    "grad_clip": 1.0,
    "groups": GROUPS,
    "betas": [0.9, 0.999],
    "warmup_steps": 0,
    "lr_floor": 0.1,
    "device": "cpu",
}


def repository(root: Path, repo: Path, steps=2, horizon=2, pairs=2) -> None:
    """Copy approved configs, changing only the tiny scenario's explicit values."""
    shutil.copytree(repo / "configs", root / "configs")
    edits = {
        "env-sampling.yaml": {"steps": steps, "out_dir": "data/sampling"},
        "ppo.yaml": {"horizon": horizon},
        "rl-preference-pairs.yaml": {
            "out_dir": "data/rl_preference_pairs",
            "history_dir": "data/rl_preference_history",
            "sample_states": pairs,
            "temporal_per_state": 1,
            "nearest_per_state": 1,
            "branch": 1,
            "beam_width": 1,
            "beam_depth": 1,
            "items_per_shard": 2,
            "history_per_shard": 2,
        },
        "perception-data.yaml": {"states": 3},
        "perception.yaml": {"alpha": [1.5, 0.5]},
        "pretrain.yaml": {"device": "cpu"},
    }
    for name, changes in edits.items():
        path = root / "configs" / name
        value = yaml.safe_load(path.read_text()) | changes
        path.write_text(yaml.safe_dump(value, sort_keys=False))


def config_path(path: Path, **changes) -> Path:
    path.write_text(yaml.safe_dump(BASE | changes, sort_keys=False))
    return path


def config(path: Path, **changes):
    return load_config(config_path(path, **changes))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_pairs(directory: Path, records: list[dict], shard_size=2) -> None:
    for start in range(0, len(records), shard_size):
        write_json(directory / "shards" / f"{start // shard_size:06d}" / "items.json",
                   records[start : start + shard_size])  # fmt: skip
    if not records:
        write_json(directory / "shards/000000/items.json", [])


def preference(index: int) -> dict:
    def ref(suffix):
        return {"episode": f"data/sampling#00000{suffix}", "index": index}

    return {"kind": "preference", "a": ref(0), "b": ref(1),
            "labels": {"p": 0.75, "inc": 0}}  # fmt: skip


def pair_items(count: int) -> list[dict]:
    return [
        {k: v for k, v in preference(i).items() if k != "kind"} for i in range(count)
    ]


def read_json(path: Path):
    return json.loads(path.read_text())


def shard_items(directory: Path, filename="items.json") -> list[dict]:
    return [
        read_json(path / filename)
        for path in sorted(directory.iterdir())
        if path.is_dir()
    ]
