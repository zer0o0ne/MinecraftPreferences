"""Materialize effective configs without changing their source files."""

import os
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from mcpref.config import REPO, read_yaml
from mcpref.diagnostics.core.config import load_config as load_diagnostics
from mcpref.infrastructure.shards.store import write_file
from mcpref.runtime.play.checkpoints import snapshot
from mcpref.runtime.play.config import PlayConfig
from mcpref.runtime.sampling.checkpoints import checkpoint_path
from mcpref.runtime.sampling.config import load_config as load_sampling
from mcpref.training.pretrain.modes import load_selected_config as load_pretrain
from mcpref.training.pretrain.modes import selected_config
from mcpref.training.rl.config import load_config as load_rl
from mcpref.training.rl.context import load_context
from mcpref.training.rl.pairs.config import load_config as load_pairs


def create_directory(cfg: PlayConfig) -> Path:
    path = REPO / cfg.out_dir
    if os.path.lexists(path):
        raise FileExistsError(f"{path} already exists; choose a new gameplay out_dir")
    root = path.resolve()
    if not root.is_relative_to(REPO.resolve()):
        raise ValueError("gameplay out_dir must be inside the repository")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _save(root: Path, name: str, raw: dict) -> str:
    path = root / "configs" / f"{name}.yaml"
    write_file(path, yaml.safe_dump(raw, sort_keys=False).encode())
    return path.relative_to(REPO.resolve()).as_posix()


def _rl(cfg: PlayConfig, root: Path, sampling: str) -> str:
    raw = read_yaml(REPO / cfg.rl_cycle)
    pairs = read_yaml(REPO / raw["rl_pairs"])
    base = root.relative_to(REPO.resolve()).as_posix()
    pairs |= {"out_dir": f"{base}/pairs", "history_dir": f"{base}/history"}
    pairs_path = _save(root, "pairs", pairs)
    load_pairs(REPO / pairs_path)
    raw |= {
        "out_dir": f"{base}/rl",
        "sampling": sampling,
        "rl_pairs": pairs_path,
        "device": cfg.device,
    }
    path = _save(root, "rl", raw)
    load_context(load_rl(REPO / path), load_sampling(REPO / sampling))
    return path


def materialize(cfg: PlayConfig, root: Path) -> PlayConfig:
    sampling = read_yaml(REPO / cfg.sampling)
    source = checkpoint_path(load_sampling(REPO / cfg.sampling))
    _, selected = selected_config(REPO / sampling["pretrain"])
    pretrain = read_yaml(selected)
    diagnostics = read_yaml(REPO / pretrain["diagnostics"])
    base = root.relative_to(REPO.resolve()).as_posix()
    diagnostics["out_dir"] = f"{base}/diagnostics"
    diag_path = _save(root, "diagnostics", diagnostics)
    load_diagnostics(REPO / diag_path)
    pretrain |= {"diagnostics": diag_path, "device": cfg.device}
    pretrain_path = _save(root, "pretrain", pretrain)
    load_pretrain(REPO / pretrain_path)
    snapshot(source, root / "checkpoint")
    sampling["checkpoint_dir"] = f"{base}/checkpoint"
    sampling |= {"out_dir": f"{base}/sampling", "pretrain": pretrain_path}
    sampling_path = _save(root, "sampling", sampling)
    load_sampling(REPO / sampling_path)
    rl = _rl(cfg, root, sampling_path) if cfg.mode == "rl" else cfg.rl_cycle
    effective = replace(cfg, out_dir=base, sampling=sampling_path, rl_cycle=rl)
    _save(root, "play", asdict(effective))
    return effective
