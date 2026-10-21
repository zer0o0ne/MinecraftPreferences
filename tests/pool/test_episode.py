"""(3) of specs/preference-pool.md: each episode of a draw by the checks of perception-data
(1) but `version` and its processing (2)-(5) with the pool's per_episode, the generator
Generator(PCG64([seed, 4, k, j])), its video at staging/<k>/<j>.mp4 and its clips at
staging/<k>/<id>.part/."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from clips.collect_gamma import Classifier

from .pool_site import assert_outputs
from .pool_world import chain, replace


def keyed_world() -> dict:
    """Three sessions of two `w`*6 pieces (starts 0 and 1, one clip each)."""
    return {s: chain(v, f"s{i}", "ab" * 6, ["w" * 6] * 2)
            for i, (s, v) in enumerate([("6xx", "6.13"), ("8xx", "8.1"), ("10xx", "10.0")])}  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
def test_episode_generator_is_keyed_by_draw_and_episode(make_site, seed):
    """Clip of episode j of draw k: the start Generator(PCG64([seed, 4, k, j])) picks."""
    site = make_site(episodes=keyed_world())
    site.run(seed=seed, per_episode=1)
    tree, n = site.tree("shards"), 0
    for k, name in enumerate(site.complete()):
        for j, rec in enumerate(tree[f"shards/{name}/records.json"]):
            pick = np.random.Generator(np.random.PCG64([seed, 4, k, j])).integers(2)
            stem = rec["relpath"].split("/")[2][:-4]
            assert rec["clips"] == [f"{rec['relpath'].split('/')[1]}_{stem}_{pick}"]
            n += 1
    assert n == 6


@pytest.mark.parametrize("ours, theirs", [(2, 1), (1, 3)])
def test_per_episode_is_the_pools(make_site, ours, theirs):
    """`w`*16 has room for two clips: the pool's per_episode counts, not perception's."""
    eps = {"9xx": chain("9.1", "t", "ab" * 6, ["w" * 16] * 2)}
    site = make_site(episodes=eps)
    site.run(per_episode=ours, p={"per_episode": theirs})
    assert_outputs(site, site.expected(per_episode=ours, p={"per_episode": theirs}))
    assert [len(r["clips"]) for r in site.records()] == [ours, ours]


def test_checks_but_version(make_site):
    """missing, empty, malformed options and `options` end an episode, the rest of the
    session goes on; a valid episode after them."""
    pieces = chain("7.1", "c", "ab" * 6, ["w" * 6] * 5)
    pieces = [replace(pieces[0], options=None), replace(pieces[1], options=b""),
              replace(pieces[2], options=b"[1]"), pieces[3],
              replace(pieces[4], options=b'{"guiScale": 2}')]  # fmt: skip
    site = make_site(episodes={"7xx": pieces})
    site.run(episodes=5)
    assert_outputs(site, site.expected(episodes=5))
    reasons = [r["reason"] for r in site.records()]
    assert reasons == ["missing", "empty", "malformed", None, "options"]


def test_video_of_episode_j_of_draw_k_in_staging(make_site):
    """The decodes read staging/<k>/<j>.mp4 under out_dir."""
    site = make_site(episodes=keyed_world())
    site.run(per_episode=1)
    ex = site.expected(per_episode=1)
    where = {}
    for k, s in enumerate(ex.shards):
        for j, rec in enumerate(s.records):
            where[rec["relpath"].split("/")[2][:-4]] = (k, j)
    decodes = site.decodes()
    assert len(decodes) == 12  # passes 1 and 2 of six episodes
    for d in decodes:
        k, j = where[d["key"]]
        video = Path(d["argv"][d["argv"].index("-i") + 1])
        assert video.resolve() == (site.out / "staging" / str(k) / f"{j}.mp4").resolve()


def wait(condition, seconds: float = 4.0) -> bool:
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.02)
    return condition()


def staging(site) -> dict | None:
    """site.tree("staging"), None while a file changes under it."""
    try:
        return site.tree("staging")
    except (FileNotFoundError, ValueError):
        return None


def staged(ex, k: int, ids) -> dict:
    """The files of the clips `ids` of shard k as staging/<k>/<id>.part/ holds them."""
    return {f"staging/{k}/{i}.part/{p.rsplit('/', 1)[1]}": x
            for p, x in ex.shards[k].files.items() for i in ids
            if p.startswith(f"clips/{i}/")}  # fmt: skip


def test_clips_wait_in_staging_until_their_shard(make_site):
    """Three workers, the video of draw 0's first episode held: its second episode and
    draw 1 (two episodes before it unwritten) are processed, their clips wait in
    staging/<k>/<id>.part/ with their files, their videos deleted; nothing is written to
    shards/."""
    site = make_site(episodes=keyed_world())
    ex = site.expected(per_episode=1)
    gate = threading.Event()
    site.gates[site.path(ex.shards[0].records[0]["relpath"], ".mp4")] = gate
    thread = site.in_thread(Classifier(site.world), workers=3, per_episode=1)
    try:
        ids = [r["clips"] for r in ex.shards[0].records + ex.shards[1].records]
        want = staged(ex, 0, ids[1]) | staged(ex, 1, ids[2] + ids[3])
        assert len(want) == 9
        assert wait(lambda: staging(site) == want), sorted(staging(site) or {})
        assert site.complete() == []
    finally:
        gate.set()
        thread.join(8.0)
    assert site.error is None
    assert_outputs(site, ex)
