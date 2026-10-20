"""`main` of specs/train-index.md on hand-built inputs (Notation): the unlabeled clips $U$
(records with null `clips` skipped, staged shards unread), the pairs $R$ (a clip in several
pairs, a pair clip equal to an unlabeled one kept), the item paths, the episodes and their
order, the groups ((2): no pair split); without pairs."""

import json

import pytest

from .index_run import items
from .index_world import LABELS, NO_PAIRS, random_world

U = [  # U of HAND: shard 0 a_10, a_50, (x: clips null), b_0; shard 1 c_7
    {"clip": "w/clips/shards/000000/clips/a_10"},
    {"clip": "w/clips/shards/000000/clips/a_50"},
    {"clip": "w/clips/shards/000000/clips/b_0"},
    {"clip": "w/clips/shards/000001/clips/c_7"},
]
R = [  # R of HAND, its Q_c clips at w/pair_clips
    {"a": "w/pair_clips/shards/000000/clips/b_0",
     "b": "w/pair_clips/shards/000000/clips/d_3", "labels": LABELS[0]},
    {"a": "w/pair_clips/shards/000000/clips/d_3",
     "b": "w/pair_clips/shards/000001/clips/e_5", "labels": LABELS[1]},
    {"a": "w/pair_clips/shards/000000/clips/b_90",
     "b": "w/pair_clips/shards/000000/clips/b_0", "labels": LABELS[2]},
]  # fmt: skip
ZERO = {"clips": 0, "pairs": 0, "episodes": 0}


def key(item: dict) -> str:
    return json.dumps(item, sort_keys=True)


def counts(site) -> dict:
    return json.loads((site.out / "run.json").read_text())["counts"]


def test_items_of_unlabeled_clips_and_pairs(make_site):
    """v = 0, one shard: the items are U's 4 clips and R's 3 pairs, each once; the episodes
    a, b, c, d, e (x has no clip, f no pair, z only staged)."""
    site = make_site(validation="0", items_per_shard=100)
    r = site.run()
    assert r.code == 0, r.err
    [shard] = items(site.out, "train")
    assert sorted(shard, key=key) == sorted(U + R, key=key)
    assert items(site.out, "validation") == []
    assert counts(site) == {"train": {"clips": 4, "pairs": 3, "episodes": 5},
                            "validation": ZERO}  # fmt: skip


def test_items_without_pairs(make_site):
    """Pair dirs null: U alone; the episodes a, b, c; Q_c and Q_l never read (absent)."""
    site = make_site(
        write=False, **{**NO_PAIRS, "validation": "0", "items_per_shard": 9}
    )
    site.write()
    r = site.run()
    assert r.code == 0, r.err
    [shard] = items(site.out, "train")
    assert sorted(shard, key=key) == sorted(U, key=key)
    assert counts(site) == {"train": {"clips": 4, "pairs": 0, "episodes": 3},
                            "validation": ZERO}  # fmt: skip


def test_item_paths_from_the_config_strings(make_site):
    """p = str(PurePosixPath(<config path string>) / "shards" / name(k) / "clips" / id):
    './w//clips/' gives 'w/clips/shards/...'."""
    site = make_site(write=False, validation="0", items_per_shard=100)
    site.config |= {"clips_dir": "./w//clips/", "pair_clips_dir": "w/pair_clips/"}
    site.write()
    assert site.run().code == 0
    [shard] = items(site.out, "train")
    assert sorted(shard, key=key) == sorted(U + R, key=key)


def episode_parts(site, world) -> dict:
    """episode -> the set of parts of the items of its clips (unlabeled and paired)."""
    where = {}
    for x, part in enumerate(("train", "validation")):
        for shard in items(site.out, part):
            for item in shard:
                for p in [item.get("clip"), item.get("a"), item.get("b")]:
                    if p is not None:
                        where.setdefault(p, set()).add(x)
    parts = {}
    for directory, shards in (
        ("w/clips", world.clips),
        ("w/pair_clips", world.pair_clips),
    ):
        for k, records in enumerate(shards):
            for r in records:
                for i in r["clips"] or ():
                    p = f"{directory}/shards/{k:06d}/clips/{i}"
                    parts.setdefault(r["relpath"], set()).update(where.get(p, set()))
    return parts


@pytest.mark.parametrize("world_seed", [0, 1, 2])
def test_episodes_linked_by_pairs_share_a_part(make_site, tmp_path, world_seed):
    """(1), (2): every episode's items lie in one part; a pair's two episodes in the same
    one, also when pairs chain episodes (HAND: b - d - e); over seeds both parts occur."""
    world = random_world(world_seed)
    seen = set()
    for seed in range(4):
        site = make_site(world=world, seed=seed, validation="0.5", out_dir=f"w/o{seed}")
        assert site.run().code == 0
        parts = episode_parts(site, world)
        assert all(len(p) <= 1 for p in parts.values()), parts
        for shard in world.pairs:
            for p in shard:
                assert parts[p["a"]["episode"]] == parts[p["b"]["episode"]]
        seen |= set().union(*parts.values())
    assert seen == {0, 1}


def test_chained_pairs_one_group(make_site):
    """HAND, v = 0.5, seeds 0-7: b's clip and the three pairs (b - d, d - e, b - b) always
    in one part, a's two clips in one part."""
    for seed in range(8):
        site = make_site(seed=seed, validation="0.5", out_dir=f"w/o{seed}")
        assert site.run().code == 0
        where = {
            key(item): x
            for x, part in enumerate(("train", "validation"))
            for shard in items(site.out, part)
            for item in shard
        }
        assert len({where[key(i)] for i in [U[2], *R]}) == 1
        assert where[key(U[0])] == where[key(U[1])]
