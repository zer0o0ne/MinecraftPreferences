"""Eqs. (3)-(6): temporal, nearest and beam alternatives."""

from collections.abc import Sequence

import numpy as np
import torch

from mcpref.rl_pairs.config import RLPreferencePairsConfig
from mcpref.rl_pairs.history import History
from mcpref.rl_pairs.scores import choice
from mcpref.rl_pairs.types import StateRef


def temporal(
    seeds: Sequence[int], seed: int, count: int, rng: np.random.Generator
) -> list[int]:
    """Eq. (3): other fresh seeds weighted by temporal distance."""
    others = [t for t in seeds if t != seed]
    n = min(count, len(others))
    if n == 0:
        return []
    weights = np.asarray([abs(seed - t) for t in others], dtype=float)
    probabilities = weights / weights.sum()
    return choice(rng, np.asarray(others), n, probabilities)


def nearest(
    seed: StateRef, history: History, alpha: torch.Tensor, count: int
) -> list[StateRef]:
    """Eq. (4): the nearest stale keys outside the seed's episode."""
    if count == 0:
        return []
    vertex = history.get(seed)
    if vertex is None:
        raise KeyError("the seed must be inserted before nearest alternatives")
    key = torch.tensor(vertex.key, dtype=alpha.dtype, device=alpha.device)
    found: list[tuple[float, StateRef]] = []
    for other in history.outside(seed.episode):
        old = torch.tensor(other.key, dtype=key.dtype, device=key.device)
        distance = torch.linalg.vector_norm(alpha * (key - old), ord=2, dim=-1).item()
        found.append((distance, other.ref))
        found.sort(key=lambda item: (item[0], item[1]))
        del found[count:]
    return [ref for _, ref in found]


def polarity(p: float, inc: float) -> float:
    """Eq. (5): orientation-invariant comparison polarity J."""
    return max(inc, (1 - inc) * abs(2 * p - 1))


def expand(
    seed: StateRef,
    current: list[StateRef],
    visited: set[StateRef],
    history: History,
    branch: int,
) -> dict[StateRef, float]:
    """One Eq. (6) frontier's best branch discoveries before global beam ranking."""
    discovered: dict[StateRef, float] = {}
    for ref in current:
        vertex = history.get(ref)
        if vertex is None:
            continue
        records = []
        for c in vertex.comparisons:
            if c.ref.episode == seed.episode or c.ref in visited:
                continue
            records.append((polarity(c.p, c.inc), c.ref))
        records.sort(key=lambda item: (-item[0], item[1]))
        for value, partner in records[:branch]:
            discovered[partner] = max(value, discovered.get(partner, 0.0))
    return discovered


def beam(
    seed: StateRef,
    frontier: Sequence[StateRef],
    history: History,
    cfg: RLPreferencePairsConfig,
) -> list[StateRef]:
    """Eq. (6): best-polarity recursive expansion from ordered nearest vertices."""
    visited = {seed, *frontier}
    alternatives = list(frontier)
    current = list(frontier)
    for _ in range(cfg.beam_depth):
        discovered = expand(seed, current, visited, history, cfg.branch)
        ranked = sorted(discovered, key=lambda ref: (-discovered[ref], ref))
        current = ranked[: cfg.beam_width]
        if not current:
            break
        visited.update(current)
        alternatives.extend(current)
    return alternatives
