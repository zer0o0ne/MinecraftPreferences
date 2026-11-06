"""Run selection, static identities and content fingerprints of RL-cycle artifacts."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

from mcpref import REPO
from mcpref.pretrain.config import load_config as load_pretrain_config
from mcpref.rl.config import RLCycleConfig
from mcpref.rl_pairs.config import load_config as load_pairs_config
from mcpref.rl_pairs.files import write_json_shard
from mcpref.runs.files import check_fingerprints, sha256, write_json
from mcpref.sampling.config import SamplingConfig
from mcpref.sampling.files import config_paths
from mcpref.shards.store import complete_shards, shard_path


def select_cycle(cfg: RLCycleConfig, trajectories: int) -> int:
    """Resume the highest numeric unfinished run, or start after the highest final one."""
    root = REPO / cfg.out_dir
    runs = []
    if root.exists():
        runs = [
            int(path.name)
            for path in root.iterdir()
            if path.is_dir()
            and path.name.isascii()
            and path.name.isdigit()
            and (path / "run.json").exists()
        ]
    if not runs:
        return trajectories
    latest = max(runs)
    run = json.loads((root / f"{latest:06d}" / "run.json").read_text())
    return latest + 1 if run.get("complete") else latest


def initial_model(cfg: RLCycleConfig, sampling: SamplingConfig, cycle: int) -> Path:
    """Pretrain weights for cycle 0, otherwise the previous cycle's final weights."""
    if cycle:
        return REPO / cfg.out_dir / f"{cycle - 1:06d}" / "model.pt"
    pretrain = load_pretrain_config(REPO / sampling.pretrain)
    return REPO / pretrain.out_dir / "model.pt"


def fingerprints(
    cfg: RLCycleConfig, path: Path, sampling: SamplingConfig, model: Path
) -> dict:
    """Static RL identities followed by sampling's ordered config identities."""
    source = getattr(sampling, "_source_path", REPO / cfg.sampling)
    result = {
        "rl_config": sha256(path),
        "ppo": sha256(REPO / cfg.ppo),
        "rl_pairs": sha256(REPO / cfg.rl_pairs),
    }
    result |= {
        key: sha256(value)
        for key, value in config_paths(sampling, source).items()
        if key != "pretrain_model"
    }
    result["initial_model"] = sha256(model)
    return result


def start_run(
    cfg: RLCycleConfig, path: Path, cycle: int, u_before: int, hashes: dict
) -> tuple[Path, dict]:
    """Create the cycle run before R1, or enforce its static identity on resume."""
    root = REPO / cfg.out_dir / f"{cycle:06d}"
    run_path = root / "run.json"
    check_fingerprints(run_path, hashes, cfg.out_dir)
    if run_path.exists():
        run = json.loads(run_path.read_text())
        if run["cycle"] != cycle or run["u_before"] != u_before:
            raise RuntimeError(
                f"cycle position changed since the run started; delete {root}"
            )
        return run_path, run
    run = {
        "cycle": cycle,
        "u_before": u_before,
        "phase": None,
        "complete": False,
        "config": yaml.safe_load(path.read_text()),
        "sha256": hashes,
    }
    write_json(run_path, run)
    return run_path, run


def advance(
    path: Path, run: dict, phase: str, complete: bool = False, **hashes
) -> dict:
    """Atomically record a satisfied phase and its immutable artifact fingerprints."""
    for key, value in hashes.items():
        if key in run["sha256"] and run["sha256"][key] != value:
            raise RuntimeError(
                f"{key} changed since the run started; delete {path.parent}"
            )
    run = run | {"phase": phase, "complete": complete}
    run["sha256"] = run["sha256"] | hashes
    write_json(path, run)
    return run


def pair_rng(cfg: RLCycleConfig, cycle: int) -> np.random.Generator:
    """Fresh independent pair RNG at cycle 0, then continuous across cycles."""
    if cycle == 0:
        return np.random.Generator(np.random.PCG64([cfg.seed, 4]))
    pairs = load_pairs_config(REPO / cfg.rl_pairs)
    previous = REPO / pairs.out_dir / f"{cycle - 1:06d}" / "run.json"
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = json.loads(previous.read_text())["rng"]
    return rng


def incoming_tail(root: Path, cycle: int, u_before: int) -> list[int]:
    """Read and validate the previous cycle's complete bounded tail."""
    if cycle == 0:
        return []
    path = shard_path(root.parent / f"{cycle - 1:06d}" / "tail" / "shards", 0)
    tail = json.loads((path / "items.json").read_text())
    indices = tail["indices"]
    if tail["next_step"] != u_before or any(t < 0 or t >= u_before for t in indices):
        raise RuntimeError("the incoming tail is not continuous with this cycle")
    if any(b <= a for a, b in pairwise(indices)):
        raise RuntimeError("the incoming tail is not strictly increasing and unique")
    return indices


def write_tail(root: Path, u_after: int, horizon: int) -> None:
    """Write the sole next-cycle tail shard once, including an empty index list."""
    directory = root / "tail"
    value = {
        "next_step": u_after,
        "indices": list(range(max(0, u_after - horizon + 1), u_after)),
    }
    if not complete_shards(directory / "shards"):
        write_json_shard(directory / "shards", 0, "items.json", value)
    run = directory / "run.json"
    if not run.exists():
        write_json(
            run,
            {
                "complete": True,
                "sha256": shards_digest(directory / "shards", "items.json"),
            },
        )


def shards_digest(directory: Path, filename: str) -> str:
    """SHA-256 of complete shards' filename contents in shard order."""
    digest = hashlib.sha256()
    for k in range(complete_shards(directory)):
        digest.update((shard_path(directory, k) / filename).read_bytes())
    return digest.hexdigest()


def tree_digest(directory: Path) -> str:
    """Recursive path-ordered content digest of one immutable trajectory shard."""
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
