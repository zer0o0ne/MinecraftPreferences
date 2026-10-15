"""(1), (6) of specs/perception-data.md: the draws and the reasons, one shard per draw
(several popped episodes, an empty pool without a candidate, a candidate ending its draw
whatever its reason), the generator state stored with each shard, the cut at N."""

import numpy as np
import pytest

from .collect_out import assert_outputs
from .collect_world import EPISODES, Ep


def names(records: list) -> list[str]:
    return [r["relpath"].split("/")[2][:-4] for r in records]


def shards(site) -> list[list[dict]]:
    """The records of each complete shard."""
    tree = site.tree("shards")
    return [tree[f"shards/{x}/records.json"] for x in site.complete()]


def test_records_follow_the_draws_with_every_reason(site):
    site.run()
    ex = site.expected()
    assert [x.records for x in ex.shards] == shards(site)
    assert site.complete() == [f"{k:06d}" for k in range(11)]
    assert (
        len(site.records()) == 13
    )  # every pool emptied: 14 relpaths, one exact repeat


def test_exact_repeats_are_drawn_once(site):
    """7xx lists data/7.1/b7.mp4 twice."""
    site.run()
    relpaths = [r["relpath"] for r in site.records()]
    assert relpaths.count("data/7.1/b7.mp4") == 1
    assert len(relpaths) == len(set(relpaths))


def pops(seed: int, n: int) -> list[int]:
    """(1) on one subset of n episodes: the pool positions popped by the first draw."""
    g = np.random.Generator(np.random.PCG64(seed))
    g.integers(1)
    return [int(g.integers(m)) for m in range(n, 0, -1)]


def test_several_popped_episodes_in_one_shard(make_site):
    """One subset [x (version), m (missing), a (valid)]: a seed whose first draw pops `a`
    last; one shard holds the three records in pop order, a's clips and options."""
    eps = [Ep("data/6.9/x.mp4"), Ep("data/7.0/m.mp4", options=None), Ep("data/7.0/a.mp4", "w" * 6)]  # fmt: skip
    seed = next(s for s in range(100) if pops(s, 3)[:2] in ([0, 0], [1, 0]))
    order = list(eps)
    popped = [order.pop(j) for j in pops(seed, 3)]
    site = make_site(episodes={"7xx": eps})
    site.run(seed=seed)
    assert site.complete() == ["000000"]
    assert names(site.records()) == [e.stem for e in popped]
    assert [r["reason"] for r in site.records()] == [
        {"x": "version", "m": "missing", "a": None}[e.stem] for e in popped
    ]
    assert_outputs(site, site.expected(seed=seed))


def test_draw_ending_with_an_empty_pool_has_no_candidate(make_site):
    eps = [Ep("data/6.9/x.mp4"), Ep("data/7.0/m.mp4", options=None), Ep("data/7.0/e.mp4", jsonl=b"")]  # fmt: skip
    site = make_site(episodes={"8xx": eps})
    site.run()
    assert site.complete() == ["000000"]
    assert sorted(r["reason"] for r in site.records()) == [
        "empty",
        "missing",
        "version",
    ]
    assert set(site.raw("shards")) == {"shards/000000/records.json", "shards/000000/state.json"}  # fmt: skip
    assert site.run_json()["clips"] == 0
    assert_outputs(site, site.expected())


def test_a_candidate_ends_its_draw_whatever_its_reason(make_site):
    """Every episode passes the checks: one draw per episode, `video` and `no_clip` too."""
    eps = [Ep("data/7.0/v.mp4", video=None), Ep("data/7.0/n.mp4", "wwww"), Ep("data/7.0/a.mp4", "w" * 6)]  # fmt: skip
    site = make_site(episodes={"9xx": eps})
    site.run()
    assert site.complete() == ["000000", "000001", "000002"]
    assert [len(x) for x in shards(site)] == [1, 1, 1]
    assert sorted(map(str, (r["reason"] for r in site.records()))) == ["None", "no_clip", "video"]  # fmt: skip
    assert_outputs(site, site.expected())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subsets_and_pools_are_drawn_as_in_the_transcription(make_site, seed):
    """Pools of different sizes, an empty subset; one clip per episode, N = 7."""
    eps = {s: [Ep(f"data/7.0/{s}{k}.mp4", "w" * 6) for k in range(n)]
           for s, n in zip(EPISODES, (3, 1, 0, 2, 4))}  # fmt: skip
    site = make_site(episodes=eps)
    changes = {"seed": seed, "n_clips": 7, "per_episode": 1}
    site.run(**changes)
    assert_outputs(site, site.expected(**changes))


def test_state_continues_the_plan_generator(site):
    """state.json of shard k: the generator after draw k; drawing from it gives draw k + 1's
    subset and first pop (the pools of (7) from the records of shards 0..k)."""
    site.run()
    ex, tree = site.expected(), site.tree("shards")
    pools = {s: list(dict.fromkeys(e.relpath for e in es)) for s, es in site.world.episodes.items()}  # fmt: skip
    for k in range(len(ex.shards) - 1):
        state = tree[f"shards/{k:06d}/state.json"]
        assert state["clips"] == ex.shards[k].clips
        for r in ex.shards[k].records:
            pools[r["subset"]].remove(r["relpath"])
        g = np.random.Generator(np.random.PCG64())
        g.bit_generator.state = state["generator"]
        active = [s for s in pools if pools[s]]
        s = active[g.integers(len(active))]
        assert ex.shards[k + 1].records[0]["relpath"] == pools[s][g.integers(len(pools[s]))]  # fmt: skip


@pytest.mark.parametrize("where", [0, 1], ids=["inside", "at_end"])
def test_cut_at_n_writes_the_first_n_clips(site, where):
    """N inside the first candidate with two clips, or at its end: its shard is the last;
    it holds the first N - c clips in the order taken."""
    full = site.expected()
    k = full.first(None, clips=2)
    n = full.shards[k - 1].clips + 1 + where
    site.run(n_clips=n)
    ex = site.expected(n_clips=n)
    assert_outputs(site, ex)
    assert len(ex.shards) == k + 1 and site.run_json()["clips"] == n
    assert (
        site.records()[-1]["clips"] == full.shards[k].records[-1]["clips"][: 1 + where]
    )
    assert [x.records for x in ex.shards[:k]] == [x.records for x in full.shards[:k]]


@pytest.mark.parametrize("seed", range(4))
def test_episode_generator_is_keyed_by_the_draw(make_site, seed):
    """Three candidates of `wwwwww` (starts 0, 1; one clip each), a `version` episode
    popped before one of them: each clip starts at the first pick of
    Generator(PCG64([seed, 1, k])) among the two starts, k its shard."""
    eps = {"6xx": [Ep("data/6.9/x.mp4"), Ep("data/7.0/a.mp4", "w" * 6)],
           "7xx": [Ep("data/7.0/b.mp4", "w" * 6)], "8xx": [Ep("data/7.0/c.mp4", "w" * 6)]}  # fmt: skip
    site = make_site(episodes=eps)
    site.run(seed=seed, per_episode=1)
    tree, valid = site.tree("shards"), 0
    for k, name in enumerate(site.complete()):
        for rec in tree[f"shards/{name}/records.json"]:
            if rec["reason"] is None:
                pick = np.random.Generator(np.random.PCG64([seed, 1, k])).integers(2)
                assert rec["clips"] == [f"7.0_{rec['relpath'][9]}_{pick}"], k
                valid += 1
    assert valid == 3
