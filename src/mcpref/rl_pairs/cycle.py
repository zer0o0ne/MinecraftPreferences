"""The complete resumable R3 pair-production cycle."""

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from torch import Tensor

from mcpref import REPO
from mcpref.preferences.network import Preference
from mcpref.rl_pairs.config import RLPreferencePairsConfig
from mcpref.rl_pairs.history import History
from mcpref.rl_pairs.label import correct
from mcpref.rl_pairs.sample import select
from mcpref.rl_pairs.types import Encoder, StateRef, ref_dict
from mcpref.runs.files import write_json


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
    """The cycle run file after a completed phase."""
    return run | {
        "phase": phase,
        "rng": rng.bit_generator.state,
        "complete": phase == "history",
    }


def already_done() -> None:
    """Report all four completed phases without replaying their work."""
    for desc in ("seed scoring", "pair sampling", "pair correction", "pair cleanup"):
        print(f"rl {desc}: already done", file=sys.stderr)


def finish_published(
    out_dir: Path, history_dir: Path, run: dict, rng: np.random.Generator
) -> Path:
    """Finish a rerun whose complete History generation was already published."""
    graph = json.loads((out_dir / "graph.json").read_text())
    History.load(history_dir, run["config"]["history_per_shard"], run["config"])
    rng.bit_generator.state = run["rng"]
    finished = main_run(run, "history", rng) | {"pairs": len(graph["comparisons"])}
    write_json(out_dir / "run.json", finished)
    already_done()
    return out_dir


def sample_pairs(
    cfg: RLPreferencePairsConfig,
    cycle: int,
    states: Sequence[StateRef],
    encode: Encoder,
    P: Preference,
    alpha: Tensor,
    rng: np.random.Generator,
) -> Path:
    """Run one complete cycle and return its immutable Q_c directory."""
    out_dir = REPO / cfg.out_dir / f"{cycle:06d}"
    history_dir = REPO / cfg.history_dir
    run = check_run(out_dir / "run.json", initial_run(cfg, cycle, states, rng))
    if run.get("complete"):
        already_done()
        return out_dir
    published = history_dir / f"{cycle:06d}" / "run.json"
    if published.exists() and json.loads(published.read_text()).get("complete"):
        return finish_published(out_dir, history_dir, run, rng)
    history = History.load(history_dir, cfg.history_per_shard, asdict(cfg))
    work_dir = out_dir / "work"
    select(states, history, encode, P, alpha, cfg, rng, work_dir)
    run = main_run(run, "sampling", rng)
    write_json(out_dir / "run.json", run)
    pairs = correct(work_dir, out_dir, cfg)
    run = main_run(run, "correction", rng) | {"pairs": pairs}
    write_json(out_dir / "run.json", run)
    history.age_unused()
    history.cleanup(cfg.capacity, out_dir / "cleanup.json")
    history.publish(cycle)
    finished = main_run(run, "history", rng) | {"pairs": pairs}
    write_json(out_dir / "run.json", finished)
    return out_dir
