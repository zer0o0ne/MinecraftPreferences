"""Timed action choice and delayed transition records of the spec (3)-(6)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from tqdm import tqdm

from mcpref.env.state import EnvState
from mcpref.mcts.search import Decision, decide
from mcpref.mcts.tree import LatentHistory, advance
from mcpref.sampling.codecs import _cpu, _encode_tree
from mcpref.sampling.history import encode, preferences
from mcpref.sampling.shards import SamplerState, StepRecord, write_state, write_step

if TYPE_CHECKING:
    from mcpref.sampling.runner import Sampler


@dataclass
class Pending:
    """(5): a chosen transition awaiting its factual successor and preferences."""

    index: int
    before: EnvState
    after: EnvState
    action: int
    probabilities: Tensor
    predicted: Tensor
    elapsed_ms: float
    tree: dict
    z: Tensor
    origin: LatentHistory
    start: LatentHistory


def _complete(sampler: Sampler, pending: Pending) -> StepRecord:
    """(5)-(6): finish a pending record from the already encoded H_t+1."""
    close, start = preferences(
        sampler.modules.P, sampler.history, pending.origin, pending.start
    )
    delta = _cpu(sampler.history.Y[0, -1] - pending.z)
    return StepRecord(
        pending.index,
        pending.before,
        pending.after,
        pending.action,
        pending.probabilities,
        delta,
        pending.predicted,
        pending.elapsed_ms,
        close,
        start,
        pending.tree,
    )


def _predicted(sampler: Sampler, action: int) -> Tensor:
    """Notation: the executed root child's exact shift before re-rooting."""
    child = next(c for c in sampler.tree.root.children if c.action == action)
    return _cpu(child.x - sampler.tree.root.x)


def _choose(
    sampler: Sampler, local: int
) -> tuple[EnvState, Decision, float, dict, Tensor]:
    """(3): prepare and decide within the deadline, then snapshot the tree."""
    started = time.perf_counter()
    deadline = started + (50 * sampler.delta - sampler.cfg.safety_limit) / 1000
    sampler._prepare()
    if sampler.pending is not None:
        advance(
            sampler.tree,
            sampler.pending.action,
            sampler.history,
            sampler.alpha,
            sampler.mcts.epsilon,
        )
    if local == 0:
        sampler.trajectory_start = sampler.history
    before = sampler._observation
    decision = decide(
        sampler.tree,
        before.mask,
        sampler.modules.pi,
        sampler.modules.P,
        sampler.alpha,
        sampler.mcts,
        sampler.rng,
        deadline,
    )
    elapsed = 1000 * (time.perf_counter() - started)
    snapshot = _encode_tree(sampler.tree)
    return before, decision, elapsed, snapshot, _predicted(sampler, decision.action)


def _pending(
    sampler: Sampler,
    before: EnvState,
    after: EnvState,
    decision: Decision,
    elapsed: float,
    snapshot: dict,
    predicted: Tensor,
) -> Pending:
    """(5): retain one chosen transition until its successor is encoded."""
    return Pending(
        sampler.next_step,
        before,
        after,
        decision.action,
        torch.tensor(decision.probabilities, dtype=torch.float64, device="cpu"),
        predicted,
        elapsed,
        snapshot,
        sampler.history.Y[0, -1],
        sampler.history,
        sampler.trajectory_start,
    )


def _write_previous(sampler: Sampler, staged: Path, local: int, bar: tqdm) -> None:
    """(5), (8): write the preceding delayed record and advance the bar."""
    if sampler.pending is not None:
        write_step(staged, local - 1, _complete(sampler, sampler.pending))
        bar.update(1)


def _save_state(sampler: Sampler, staged: Path, after: EnvState) -> None:
    """(7): write the complete trajectory-boundary continuation state."""
    write_state(
        staged,
        SamplerState(
            after,
            sampler.env.continuation(),
            sampler.history,
            sampler.tree,
            sampler.rng.bit_generator.state,
            sampler.next_step,
            sampler.stream,
            sampler.modules.fingerprint,
        ),
    )


def _finish(
    sampler: Sampler,
    staged: Path,
    local: int,
    pending: Pending,
    after: EnvState,
    bar: tqdm,
) -> None:
    """(5): encode and advance the final successor, then persist its step and state."""
    sampler.history = encode(
        sampler.stream,
        sampler.history,
        sampler._raw,
        sampler._preceding,
        sampler.actions,
        sampler.states,
    )
    sampler._raw, sampler._preceding = [], []
    advance(
        sampler.tree,
        pending.action,
        sampler.history,
        sampler.alpha,
        sampler.mcts.epsilon,
    )
    write_step(staged, local, _complete(sampler, pending))
    bar.update(1)
    _save_state(sampler, staged, after)
    sampler.pending = None


def step(sampler: Sampler, staged: Path, local: int, bar: tqdm) -> None:
    """(3)-(5): choose and execute one action, completing the final step if needed."""
    before, decision, elapsed, snapshot, predicted = _choose(sampler, local)
    _write_previous(sampler, staged, local, bar)
    after = sampler.env.step(decision.action)
    sampler._observation = after
    pending = _pending(sampler, before, after, decision, elapsed, snapshot, predicted)
    sampler.pending = pending
    sampler.next_step += 1
    sampler._raw, sampler._preceding = [after], [decision.action]
    if local + 1 == sampler.cfg.steps:
        _finish(sampler, staged, local, pending, after, bar)
