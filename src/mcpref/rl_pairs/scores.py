"""Eqs. (1)-(2): auxiliary seed scores and seed draws."""

import contextlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcpref.preferences.network import Preference
from mcpref.rl_pairs.files import (
    read_json_shard,
    shard_items,
    write_json_shard,
)
from mcpref.rl_pairs.types import Encoder, StateRef
from mcpref.shards.store import complete_shards


@contextlib.contextmanager
def evaluating(P: Preference) -> Iterator[None]:
    """Start-of-cycle P in eval mode under no_grad, with its caller mode restored."""
    training = P.training
    P.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        P.train(training)


def predict(
    refs: Sequence[StateRef], encode: Encoder, P: Preference
) -> tuple[float, float]:
    """One P comparison, returning sigmoid preference and incomparability."""
    histories = encode(refs)
    preference, incomparable = P(histories[0], histories[1])
    return torch.sigmoid(preference).item(), torch.sigmoid(incomparable).item()


def choice(
    rng: np.random.Generator, values: np.ndarray, size: int, p: np.ndarray | None
) -> list[int]:
    """Eqs. (2)-(3)'s explicit no-replacement draw in its returned order."""
    return rng.choice(
        values, size=size, replace=False, p=p, axis=0, shuffle=True
    ).tolist()


def shard_size(scores_dir: Path) -> int:
    """The configured score-shard size, or one for the standalone public operation."""
    run = scores_dir.parent / "run.json"
    if run.exists():
        return json.loads(run.read_text())["config"]["items_per_shard"]
    return 1


def score_remaining(
    states: Sequence[StateRef],
    encode: Encoder,
    P: Preference,
    scores_dir: Path,
    per_shard: int,
    shards: int,
    done: int,
    bar: tqdm | None,
) -> None:
    """Score and persist the unfinished suffix under eval/no_grad."""
    with evaluating(P):
        scores: list[float] = []
        for t in range(done, len(states)):
            if t == 0:
                score = 0.0
            else:
                p, q = predict(states[t - 1 : t + 1], encode, P)
                score = max(q, (1 - q) * abs(2 * p - 1))
            scores.append(score)
            if len(scores) == per_shard:
                write_json_shard(scores_dir, shards, "scores.json", scores)
                shards += 1
                scores = []
            if t > 0 and bar is not None:
                bar.update(1)
        if scores:
            write_json_shard(scores_dir, shards, "scores.json", scores)


def seed_scores(
    states: Sequence[StateRef], encode: Encoder, P: Preference, scores_dir: Path
) -> Path:
    """Eq. (1), persisted in episode order and resumed after complete shards."""
    per_shard = shard_size(scores_dir)
    shards = complete_shards(scores_dir)
    done = shard_items(scores_dir, "scores.json", shards, per_shard)
    total = max(len(states) - 1, 0)
    initial = max(done - 1, 0)
    if not states:
        print("rl seed scoring: nothing to do", file=sys.stderr)
        return scores_dir
    if done == len(states):
        print("rl seed scoring: already done", file=sys.stderr)
        return scores_dir
    bar = None
    if initial < total:
        bar = tqdm(total=total, initial=initial, desc="rl seed scoring", unit="state")
    else:
        print("rl seed scoring: nothing to do", file=sys.stderr)
    try:
        score_remaining(states, encode, P, scores_dir, per_shard, shards, done, bar)
    finally:
        if bar is not None:
            bar.close()
    return scores_dir


def read_scores(scores_dir: Path) -> list[float]:
    """Read complete score shards in order for the sampling distribution."""
    scores = []
    for k in range(complete_shards(scores_dir)):
        scores.extend(read_json_shard(scores_dir, k, "scores.json"))
    return scores


def choose_seeds(scores_dir: Path, count: int, rng: np.random.Generator) -> list[int]:
    """Eq. (2): positive-score weighted draws, then uniform zero-score filling."""
    scores = read_scores(scores_dir)
    n = min(count, len(scores))
    positive = np.asarray([t for t, score in enumerate(scores) if score > 0])
    zero = np.asarray([t for t, score in enumerate(scores) if score == 0], dtype=int)
    n_positive = min(n, len(positive))
    chosen = []
    if n_positive:
        probabilities = np.asarray([scores[t] for t in positive])
        probabilities = probabilities / probabilities.sum()
        chosen = choice(rng, positive, n_positive, probabilities)
    if n_positive < n:
        chosen.extend(choice(rng, zero, n - n_positive, None))
    return chosen
