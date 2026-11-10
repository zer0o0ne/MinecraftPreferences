"""Eqs. (1)-(7): seed selection, raw work and current comparisons."""

import contextlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.models.preference.network import Preference
from mcpref.training.rl.pairs.alternatives import beam, nearest, polarity, temporal
from mcpref.training.rl.pairs.config import RLPreferencePairsConfig
from mcpref.training.rl.pairs.files import write_json_shard
from mcpref.training.rl.pairs.history import History
from mcpref.training.rl.pairs.scores import (
    choose_seeds,
    evaluating,
    memoized,
    predict,
    seed_scores,
)
from mcpref.training.rl.pairs.types import (
    CurrentComparison,
    Encoder,
    StateRef,
    comparison_dict,
    current_comparison,
    ref_dict,
)

__all__ = [
    "beam",
    "choose_seeds",
    "nearest",
    "polarity",
    "seed_scores",
    "select",
    "temporal",
]


def pair_key(a: StateRef, b: StateRef) -> tuple[StateRef, StateRef]:
    """The unordered deduplication key of a current-cycle pair."""
    return (a, b) if a < b else (b, a)


def read_work(path: Path) -> dict:
    """Read one complete per-seed raw-work shard."""
    return json.loads((path / "result.json").read_text())


def write_work(
    work_dir: Path,
    ordinal: int,
    seed: StateRef,
    alternatives: list[StateRef],
    comparisons: list[CurrentComparison],
    rng: np.random.Generator,
    observed: dict,
) -> None:
    """Publish a seed's alternatives, raw results and post-seed RNG state."""
    result = {
        "seed": ref_dict(seed),
        "alternatives": [ref_dict(ref) for ref in alternatives],
        "comparisons": [comparison_dict(c) for c in comparisons],
        "rng": rng.bit_generator.state,
        "diagnostics": observed,
    }
    write_json_shard(work_dir, ordinal, "result.json", result)


def insert_seeds(
    states: Sequence[StateRef], seeds: list[int], history: History, encode: Encoder
) -> None:
    """Insert every selected seed before any alternative discovery."""
    for index in seeds:
        Y = encode([states[index]])[0]
        history.stage_seed(states[index], Y[0, -1].detach().cpu().tolist())


def restore_work(
    work_dir: Path, history: History, rng: np.random.Generator
) -> tuple[int, set[tuple[StateRef, StateRef]]]:
    """Replay complete raw work exactly once into staged History and deduplication."""
    completed = complete_shards(work_dir)
    seen = set()
    last_rng = None
    for k in range(completed):
        result = read_work(shard_path(work_dir, k))
        last_rng = result["rng"]
        for value in result["comparisons"]:
            c = current_comparison(value)
            seen.add(pair_key(c.a, c.b))
            history.stage_comparison(c)
    if last_rng is not None:
        rng.bit_generator.state = last_rng
    return completed, seen


def discover(
    states: Sequence[StateRef],
    seeds: list[int],
    index: int,
    history: History,
    alpha: torch.Tensor,
    cfg: RLPreferencePairsConfig,
    rng: np.random.Generator,
    observed: dict | None = None,
) -> list[StateRef]:
    """The seed's ordered temporal, nearest and beam alternatives."""
    seed = states[index]
    chosen = temporal(seeds, index, cfg.temporal_per_state, rng)
    alternatives = [states[t] for t in chosen]
    close = nearest(seed, history, alpha, cfg.nearest_per_state)
    alternatives.extend(close)
    expanded = beam(seed, close, history, cfg)
    alternatives.extend(expanded)
    if observed is not None:
        observed.update(
            temporal=len(chosen),
            nearest=len(close),
            beam=len(expanded),
            empty_candidates=int(not alternatives),
            repeated=0,
        )
    return alternatives


def compare(
    seed: StateRef,
    alternatives: list[StateRef],
    seen: set[tuple[StateRef, StateRef]],
    history: History,
    encode: Encoder,
    P: Preference,
    observed: dict | None = None,
) -> list[CurrentComparison]:
    """Compare each newly discovered unordered pair and stage its symmetric means."""
    comparisons = []
    for alternative in alternatives:
        if pair_key(seed, alternative) in seen:
            continue
        if observed is not None:
            observed["repeated"] += int(history.older_comparison(seed, alternative))
        p, inc = predict([seed, alternative], encode, P)
        c = CurrentComparison(seed, alternative, p, inc)
        seen.add(pair_key(c.a, c.b))
        comparisons.append(c)
        history.stage_comparison(c)
    return comparisons


@contextlib.contextmanager
def sampling_bar(total: int, completed: int) -> Iterator[tqdm | None]:
    """The seed-sampling bar or its no-work/already-done line."""
    if total == 0:
        print("rl pair sampling: nothing to do", file=sys.stderr)
        yield None
    elif completed == total:
        print("rl pair sampling: already done", file=sys.stderr)
        yield None
    else:
        with tqdm(
            total=total, initial=completed, desc="rl pair sampling", unit="seed"
        ) as bar:
            yield bar


def select(
    states: Sequence[StateRef],
    history: History,
    encode: Encoder,
    P: Preference,
    alpha: torch.Tensor,
    cfg: RLPreferencePairsConfig,
    rng: np.random.Generator,
    work_dir: Path,
) -> Path:
    """Eqs. (1)-(7), with all seeds inserted before any alternative selection."""
    encode = memoized(encode)
    scores_dir = work_dir.parent / "scores"
    seed_scores(states, encode, P, scores_dir)
    seeds = choose_seeds(scores_dir, cfg.sample_states, rng)
    with evaluating(P):
        insert_seeds(states, seeds, history, encode)
        completed, seen = restore_work(work_dir, history, rng)
        with sampling_bar(len(seeds), completed) as bar:
            for ordinal in range(completed, len(seeds)):
                seed = states[seeds[ordinal]]
                observed = {}
                alternatives = discover(
                    states, seeds, seeds[ordinal], history, alpha, cfg, rng, observed
                )
                comparisons = compare(
                    seed, alternatives, seen, history, encode, P, observed
                )
                write_work(
                    work_dir, ordinal, seed, alternatives, comparisons, rng, observed
                )
                if bar is not None:
                    bar.update(1)
    return work_dir
