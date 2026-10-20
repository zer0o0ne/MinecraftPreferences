"""(1)-(3) of specs/train-index.md against the in-test transcription of index_oracle.py
(`Generator(PCG64([seed, 2]))`: one draw per group, then sigma_0, then sigma_1): every
items.json and run.json byte for byte; the shard cut ($B$ dividing $N_x$ and not, $B > N_x$,
$N_x = 0$)."""

import math

import pytest

from .index_run import items
from .index_world import HAND, NO_PAIRS, random_world

CASES = {  # name: (world seed, config changes)
    "hand": (None, {}),
    "seed 0": (0, {"seed": 0, "validation": "0.3", "items_per_shard": 4}),
    "seed 1": (1, {"seed": 1, "validation": "0.5", "items_per_shard": 1}),
    "seed max": (2, {"seed": 2**32 - 1, "validation": "0.25", "items_per_shard": 5}),
    "seed 3, v 1e-1": (3, {"seed": 3, "validation": "1e-1", "items_per_shard": 7}),
    "no pairs": (
        4,
        {**NO_PAIRS, "seed": 11, "validation": "0.4", "items_per_shard": 3},
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES)
def test_shards_and_run_json_as_transcribed(make_site, case):
    """The parts of the groups, the shuffles, the cut and the counts of the transcription;
    {out_dir} holds run.json and the items.json of the shards, nothing else."""
    world_seed, changes = case
    world = HAND if world_seed is None else random_world(world_seed)
    site = make_site(world=world, **changes)
    r = site.run()
    assert r.code == 0 and r.stdout == "", r.err
    assert site.tree() == site.expected_tree()


def test_both_parts_drawn(make_site):
    """The cases above split each world into two non-empty parts (else they would not
    check sigma_1 after sigma_0)."""
    for world_seed, changes in [c for c in CASES.values() if c[0] is not None]:
        site = make_site(world=random_world(world_seed), write=False, **changes)
        assert all(site.expected().shards), changes


@pytest.mark.parametrize("size", [1, 6, 7, 42, 4, 5, 43])
def test_shard_cut(make_site, size):
    """v = 0, N = 42 (30 clips, 12 pairs): ceil(N / B) shards of B items, the last one
    N - (n - 1) B; validation none."""
    world = random_world(4)
    n = 12 + sum(len(r["clips"] or ()) for s in world.clips for r in s)
    assert n == 42
    site = make_site(world=world, validation="0", items_per_shard=size)
    assert site.run().code == 0
    train = items(site.out, "train")
    assert [len(s) for s in train] == [size] * (n // size) + [n % size] * (n % size > 0)
    assert len(train) == math.ceil(n / size)
    assert items(site.out, "validation") == []
    assert site.tree() == site.expected_tree()


def test_empty_train_part(make_site):
    """v close to 1: every group in validation (u >= 1 - v for each draw), train has no
    shard; sigma_0 = permutation(0)."""
    site = make_site(
        world=random_world(6), validation="0.99999999999", items_per_shard=5
    )
    assert not site.expected().shards[0]
    assert site.run().code == 0
    assert items(site.out, "train") == [] and items(site.out, "validation")
    assert site.tree() == site.expected_tree()
