"""`schedule` of src/mcpref/training/schedule.py against the transcription of the Notation of
specs/gui-stack.md (schedule_oracle.py); the move of specs/pretrain.md (Interface:
`schedule`, `n_batches`, `remaining` moved from `stacking/fit.py`, which imports them)."""

import math

import numpy as np
import pytest

import mcpref.stacking.fit
import mcpref.training.schedule
from mcpref.training.schedule import schedule

from .schedule_oracle import epoch, mini_batches

CASES = [([5, 2], 2), ([3], 5), ([1, 4, 2], 1), ([6, 6, 1], 4), ([7], 7)]  # sizes, B


@pytest.mark.parametrize("sizes, batch", CASES)
def test_schedule_transcription(sizes, batch):
    """Two epochs from one generator equal the transcription of the Notation, and leave the
    generator in the same state."""
    rng, ref = (np.random.Generator(np.random.PCG64(11)) for _ in range(2))
    for _ in range(2):
        got, want = schedule(rng, sizes, batch), epoch(ref, sizes, batch)
        assert [q for q, _ in got] == [q for q, _ in want]
        assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(got, want))
        assert rng.bit_generator.state == ref.bit_generator.state


@pytest.mark.parametrize("sizes, batch", CASES)
def test_schedule_partitions_each_shard(sizes, batch):
    """Every frame of S once per epoch; each shard's mini-batches consecutive, of B frames
    but its last one (shorter or equal); M mini-batches."""
    got = schedule(np.random.Generator(np.random.PCG64(5)), sizes, batch)
    order = [q for q, _ in got]
    assert len(got) == mini_batches(sizes, batch)
    for q, n in enumerate(sizes):
        at = [i for i, r in enumerate(order) if r == q]
        assert at == list(range(at[0], at[0] + math.ceil(n / batch)))
        parts = [got[i][1] for i in at]
        assert sorted(np.concatenate(parts).tolist()) == list(range(n))
        assert all(len(p) == batch for p in parts[:-1]) and 1 <= len(parts[-1]) <= batch


@pytest.mark.parametrize("name", ["schedule", "n_batches", "remaining"])
def test_moved_from_gui_stack(name):
    """The function lives in training/schedule.py; stacking/fit.py keeps no copy of it (a
    name it holds is the imported function)."""
    moved = getattr(mcpref.training.schedule, name)
    assert moved.__module__ == "mcpref.training.schedule"
    assert getattr(mcpref.stacking.fit, name, moved) is moved
