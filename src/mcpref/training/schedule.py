"""The mini-batch schedule of specs/gui-stack.md (4), shared with specs/pretrain.md (3):
per epoch the shards permuted, then the positions within each shard."""

from collections.abc import Iterator, Sequence

import numpy as np


def schedule(
    rng: np.random.Generator, sizes: Sequence[int], batch: int
) -> list[tuple[int, np.ndarray]]:
    """Notation, (4): one epoch: the shards in the order permutation(Q), the positions
    of each permutation(|S_q|), cut into consecutive mini-batches of `batch` frames."""
    batches = []
    for q in rng.permutation(len(sizes)).tolist():
        positions = rng.permutation(sizes[q])
        starts = range(0, sizes[q], batch)
        batches += [(q, positions[i : i + batch]) for i in starts]
    return batches


def n_batches(sizes: Sequence[int], batch: int) -> int:
    """Notation: M = sum_q ceil(|S_q| / B)."""
    return sum((n + batch - 1) // batch for n in sizes)


def remaining(
    rng: np.random.Generator, sizes: Sequence[int], batch: int, state: dict, epochs: int
) -> Iterator[tuple[dict, int, np.ndarray]]:
    """(4): the mini-batches of the schedule after the first `batch_in_epoch` of epoch
    `epoch` of `state`: (epoch, schedule_rng, batch_in_epoch after it), q, positions."""
    for e in range(state["epoch"], epochs + 1):
        before = rng.bit_generator.state
        done = state["batch_in_epoch"] if e == state["epoch"] else 0
        for i, (q, positions) in enumerate(schedule(rng, sizes, batch)):
            if i >= done:
                position = {"epoch": e, "schedule_rng": before, "batch_in_epoch": i + 1}
                yield position, q, positions
