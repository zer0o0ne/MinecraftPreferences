"""(3') of specs/vpt-gui-frames.md: the plan (3) in shards of K draws, continued after
its complete shards."""

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref.shards.store import complete_shards, shard_path, write_shard
from mcpref.vpt.config import GuiFramesConfig
from mcpref.vpt.episodes import Examined, Fetch, draw, examine_gui, index, take

SHARD = "shard.json"


@dataclass
class Plan:
    """(3'): the state needed to continue: the generator, the pools (indices[s] =
    (basedir, pool)), the counts over all shards, h complete shards."""

    rng: np.random.Generator
    indices: dict[str, tuple[str, list[str]]]
    planned: int
    reasons: Counter
    h: int


@dataclass
class Shard:
    """(3'): the shard k in work: its draws' episode records and frames, its draws."""

    k: int
    episodes: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    draws: int = 0


def read_shard(directory: Path, k: int) -> dict:
    """Interface Produces: plan/<name(k)>/shard.json."""
    return json.loads((shard_path(directory, k) / SHARD).read_text())


def write_plan_shard(directory: Path, shard: Shard, rng: np.random.Generator) -> None:
    """(3'): shard k, its records and frames and the generator state after its last draw."""
    data = {"episodes": shard.episodes, "frames": shard.frames}
    data["rng"] = rng.bit_generator.state
    with write_shard(directory, shard.k) as staged:
        (staged / SHARD).write_text(json.dumps(data))


def generator(seed: int, state: dict | None) -> np.random.Generator:
    """(3'): Generator(PCG64(seed)) with h = 0, else set to the state of shard h - 1."""
    if state is None:
        return np.random.Generator(np.random.PCG64(seed))
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state
    return rng


def read_complete(
    directory: Path, h: int
) -> tuple[int, Counter, dict[str, set[str]], dict | None]:
    """(3'): the h complete shards read one at a time, bar `resume`: their frames planned,
    episodes per reason, relpaths drawn per subset, the generator state of the last."""
    planned, reasons, drawn, state = 0, Counter(), defaultdict(set), None
    if h == 0:
        print("resume: nothing to do", file=sys.stderr)
        return planned, reasons, drawn, state
    with tqdm(total=h, desc="resume", unit="shard") as bar:
        for k in range(h):
            shard = read_shard(directory, k)
            for e in shard["episodes"]:
                reasons[e["reason"]] += 1
                drawn[e["subset"]].add(e["relpath"])
            planned, state = planned + len(shard["frames"]), shard["rng"]
            bar.update(1)
    return planned, reasons, drawn, state


def resume(cfg: GuiFramesConfig, fetch: Fetch, directory: Path) -> Plan:
    """(3'): the pools of (3) without the relpaths drawn in the complete shards, order
    kept; the counts and the generator state of the complete shards."""
    indices = {s: index(fetch(url)) for s, url in cfg.index_urls.items()}
    h = complete_shards(directory)
    planned, reasons, drawn, state = read_complete(directory, h)
    for s, (basedir, pool) in indices.items():
        indices[s] = (basedir, [r for r in pool if r not in drawn[s]])
    return Plan(generator(cfg.seed, state), indices, planned, reasons, h)


def add(
    cfg: GuiFramesConfig, p: Plan, shard: Shard, s: str, relpath: str, ex: Examined
) -> int:
    """(3): a drawn episode's record and, if valid, its r frames added to the shard; the
    draw ends with a valid episode or an empty pool; returns r."""
    reason, _, options, _, lines = ex
    record = {"subset": s, "relpath": relpath, "reason": reason, "options": options}
    shard.episodes.append(record)
    p.reasons[reason] += 1
    left = cfg.n_frames - p.planned
    taken = [] if reason else take(cfg, p.rng, s, relpath, lines, left)
    shard.frames += taken
    p.planned += len(taken)
    shard.draws += reason is None or not p.indices[s][1]
    return len(taken)


def extend(cfg: GuiFramesConfig, fetch: Fetch, directory: Path, p: Plan) -> None:
    """(3), (3'): draws until N frames are planned or every pool is empty, each K draws
    and the last ones written as the next shard; bar `plan`."""
    drawn = draw(p.rng, p.indices, partial(examine_gui, cfg, fetch))
    shard = Shard(p.h)
    with tqdm(total=cfg.n_frames, initial=p.planned, desc="plan", unit="frame") as bar:
        bar.set_postfix_str(f"{p.reasons.total()} episodes")
        while p.planned < cfg.n_frames and (e := next(drawn, None)) is not None:
            bar.update(add(cfg, p, shard, *e))
            bar.set_postfix_str(f"{p.reasons.total()} episodes")
            if shard.draws == cfg.plan_shard_draws:
                write_plan_shard(directory, shard, p.rng)
                shard = Shard(shard.k + 1)
        if shard.draws:
            write_plan_shard(directory, shard, p.rng)


def plan(cfg: GuiFramesConfig, fetch: Fetch, directory: Path) -> tuple[int, Counter]:
    """Interface, (3), (3'): the plan continued after the complete shards of `directory`
    and written there; frames planned and episodes per reason over all shards."""
    p = resume(cfg, fetch, directory)
    if p.planned >= cfg.n_frames or not any(pool for _, pool in p.indices.values()):
        print(f"plan: {'already done' if p.h else 'nothing to do'}", file=sys.stderr)
    else:
        extend(cfg, fetch, directory, p)
    return p.planned, p.reasons
