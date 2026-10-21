"""(2) and (P) of specs/preference-pool.md against the transcription of pool_expected.py
with Generator(PCG64([seed, 3])): the subset among the active ones, the session popped from
its pool, up to e of its pieces at sorted(permutation[:e]), in piece order; the generator
state and the session key stored with each shard continue the plan."""

import numpy as np
import pytest

from .pool_site import assert_outputs
from .pool_world import chain, replace


def missing_options(pieces: list) -> list:
    """Pieces whose options file is missing: their records, no fetch beyond it."""
    return [replace(e, options=None) for e in pieces]


def draw_world() -> dict:
    """Sessions of 1, 3, 4 pieces in 6xx, none in 7xx (an orphan chain), 2 in 8xx, 5 and
    1 in 9xx, 3 in 10xx; 6xx's sessions interleaved in the index."""
    s = {name: missing_options(chain(v, name, "ab" * 6, ["w" * 6] * n, t0=100 * i))
         for i, (v, name, n) in enumerate([
             ("6.13", "p", 1), ("6.13", "q", 3), ("6.13", "r", 4), ("8.1", "u", 2),
             ("9.1", "v", 5), ("9.1", "w", 1), ("10.0", "y", 3)])}  # fmt: skip
    orphan = missing_options(chain("7.1", "o", "ab" * 6, ["w" * 6] * 2, session=False))
    six = [
        s["q"][0],
        s["p"][0],
        s["r"][0],
        s["q"][2],
        s["r"][3],
        s["q"][1],
        *s["r"][1:3],
    ]
    return {"6xx": six, "7xx": orphan, "8xx": s["u"], "9xx": s["v"] + s["w"], "10xx": s["y"]}  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("e", [1, 2, 4])
def test_draws_follow_the_transcription(make_site, seed, e):
    site = make_site(episodes=draw_world())
    site.run(seed=seed, episodes=e)
    ex = site.expected(seed=seed, episodes=e)
    assert_outputs(site, ex)
    assert len(ex.shards) == 7
    assert max(len(s.records) for s in ex.shards) == min(e, 5)


def test_state_continues_the_plan_generator(make_site):
    """state.json of shard k: the generator after draw k and the session's key; drawing
    from it, on the pools without the sessions of shards 0..k, gives shard k + 1's key."""
    site = make_site(episodes=draw_world())
    site.run(seed=5)
    tree, ex = site.tree("shards"), site.expected(seed=5)
    pools = {s: [] for s in site.world.episodes}
    for x in ex.sessions:
        for found in x["sessions"]:
            pools[found["subset"]].append(found["pieces"])
    for k in range(len(ex.shards)):
        state = tree[f"shards/{k:06d}/state.json"]
        assert list(state) == ["generator", "clips", "session"] and state["clips"] == 0
        s, start = state["session"]["subset"], state["session"]["start"]
        pools[s] = [x for x in pools[s] if x[0] != start]
        if k + 1 == len(ex.shards):
            assert not any(pools.values())
            break
        g = np.random.Generator(np.random.PCG64())
        g.bit_generator.state = state["generator"]
        active = [x for x in pools if pools[x]]
        subset = active[g.integers(len(active))]
        nxt = pools[subset][g.integers(len(pools[subset]))]
        assert tree[f"shards/{k + 1:06d}/state.json"]["session"] == {"subset": subset, "start": nxt[0]}  # fmt: skip
