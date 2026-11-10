"""Resumable R3 pair production and deferred R5 History publication."""

import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from mcpref import REPO
from mcpref.preferences.network import Preference
from mcpref.rl_pairs.config import RLPreferencePairsConfig
from mcpref.rl_pairs.history import History
from mcpref.rl_pairs.label import correct
from mcpref.rl_pairs.sample import read_work, select
from mcpref.rl_pairs.scores import memoized
from mcpref.rl_pairs.types import Encoder, StateRef, current_comparison, ref_dict
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards, shard_path


def fingerprint(value: object) -> str:
    """SHA-256 of a config/input JSON value."""
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def initial_run(
    cfg: RLPreferencePairsConfig,
    cycle: int,
    states: Sequence[StateRef],
    rng: np.random.Generator,
) -> dict:
    """The config/input-fingerprinted run file before any cycle phase."""
    config = asdict(cfg)
    return {
        "config": config,
        "cycle": cycle,
        "episode": states[0].episode if states else None,
        "phase": None,
        "rng": rng.bit_generator.state,
        "complete": False,
        "sha256": {
            "config": fingerprint(config),
            "input": fingerprint([ref_dict(ref) for ref in states]),
        },
    }


def check_run(path: Path, run: dict) -> dict:
    """Keep an existing cycle only when its config and input fingerprints match."""
    if not path.exists():
        write_json(path, run)
        return run
    old = json.loads(path.read_text())
    for key, value in run["sha256"].items():
        if old["sha256"][key] != value:
            raise RuntimeError(
                f"{key} changed since the run started; delete {path.parent}"
            )
    if old["cycle"] != run["cycle"] or old["episode"] != run["episode"]:
        raise RuntimeError(f"input changed since the run started; delete {path.parent}")
    return old


def main_run(run: dict, phase: str, rng: np.random.Generator) -> dict:
    """The cycle run file after a completed pair-production phase."""
    return run | {"phase": phase, "rng": rng.bit_generator.state, "complete": False}


def already_done(published: bool = False) -> None:
    """Report completed phases without replaying their work."""
    phases = ["seed scoring", "pair sampling", "pair correction"]
    if published:
        phases += ["memory", "pair cleanup"]
    for desc in phases:
        print(f"rl {desc}: already done", file=sys.stderr)


def sample_pairs(
    cfg: RLPreferencePairsConfig,
    cycle: int,
    states: Sequence[StateRef],
    encode: Encoder,
    P: Preference,
    alpha: Tensor,
    rng: np.random.Generator,
) -> Path:
    """Run (1)-(9), leaving the pending History unpublished for joint training."""
    out_dir = REPO / cfg.out_dir / f"{cycle:06d}"
    run = check_run(out_dir / "run.json", initial_run(cfg, cycle, states, rng))
    rng.bit_generator.state = run["rng"]
    if run.get("complete"):
        already_done(published=True)
        return out_dir
    if run.get("phase") in ("correction", "history"):
        already_done()
        return out_dir
    work_dir = out_dir / "work"
    if run.get("phase") != "sampling":
        history = History.load(
            REPO / cfg.history_dir, cfg.history_per_shard, asdict(cfg)
        )
        select(states, history, encode, P, alpha, cfg, rng, work_dir)
        run = main_run(run, "sampling", rng)
        write_json(out_dir / "run.json", run)
    pairs = correct(work_dir, out_dir, cfg)
    run = main_run(run, "correction", rng) | {"pairs": pairs}
    write_json(out_dir / "run.json", run)
    return out_dir


def pending_history(cfg: RLPreferencePairsConfig, cycle: int) -> History:
    """Reconstruct unpublished seeds and symmetric updates from complete raw work."""
    history = History.load(REPO / cfg.history_dir, cfg.history_per_shard, asdict(cfg))
    work = REPO / cfg.out_dir / f"{cycle:06d}" / "work"
    count = complete_shards(work)
    for k in range(count):
        result = read_work(shard_path(work, k))
        history.stage_seed(StateRef(**result["seed"]), ())
    for k in range(count):
        result = read_work(shard_path(work, k))
        for value in result["comparisons"]:
            history.stage_comparison(current_comparison(value))
    return history


def published_history(cfg: RLPreferencePairsConfig, cycle: int) -> Path | None:
    """The complete generation c, if its atomic publication already happened."""
    path = REPO / cfg.history_dir / f"{cycle:06d}"
    run = path / "run.json"
    if run.exists() and json.loads(run.read_text()).get("complete"):
        return path
    return None


def final_keys(history: History, work: Path, encode: Encoder) -> None:
    """Eq. (12): re-encode only fresh selected seeds by the final phi."""
    count = complete_shards(work)
    if count == 0:
        print("rl memory: nothing to do", file=sys.stderr)
        return
    encode = memoized(encode)
    with torch.no_grad(), tqdm(total=count, desc="rl memory", unit="seed") as bar:
        for k in range(count):
            result = read_work(shard_path(work, k))
            ref = StateRef(**result["seed"])
            Y = encode([ref])[0]
            history.replace_key(ref, Y[0, -1].detach().cpu().tolist())
            bar.update(1)


def publish_memory(
    cfg: RLPreferencePairsConfig,
    cycle: int,
    encode: Encoder,
    observe: Callable[[History], None] | None = None,
) -> Path:
    """Run (12) after training and atomically publish the pending History."""
    out_dir = REPO / cfg.out_dir / f"{cycle:06d}"
    run = json.loads((out_dir / "run.json").read_text())
    if run["cycle"] != cycle or run["config"] != asdict(cfg):
        raise RuntimeError(
            f"pair config changed since the run started; delete {out_dir}"
        )
    if run.get("phase") not in ("correction", "history"):
        raise RuntimeError("memory publication requires completed pair correction")
    published = published_history(cfg, cycle)
    if published is not None:
        History.load(REPO / cfg.history_dir, cfg.history_per_shard, asdict(cfg))
        write_json(out_dir / "run.json", run | {"phase": "history", "complete": True})
        already_done(published=True)
        return published
    history = pending_history(cfg, cycle)
    final_keys(history, out_dir / "work", encode)
    history.age_unused()
    history.cleanup(cfg.capacity, out_dir / "cleanup.json")
    if observe is not None:
        observe(history)
    published = history.publish(cycle)
    write_json(out_dir / "run.json", run | {"phase": "history", "complete": True})
    return published
