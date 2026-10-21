"""(H) and the entries and groups of Notation of specs/preference-pool.md: one head request
per entry above min_version, the groups (subset, alias, hex) with names split from the
right, not contiguous in index order, the head shards cut after the group reaching
entries_per_shard, `entries`, the head_workers bound, a failing head (the exception of the
first failing entry in the head shard's order) and the rerun continuing at the first
missing head shard."""

import json
import shutil
import time

import pytest

from .pool_site import assert_outputs
from .pool_world import PoolEp, chain, relpath
from .test_plan_draws import draw_world


def pieces(site) -> list[list[str]]:
    """Every session of the head shards, as stems."""
    tree = site.tree("sessions")
    return [[r.split("/")[2][:-4] for r in x["pieces"]]
            for p in sorted(tree) for x in tree[p]["sessions"]]  # fmt: skip


def test_sessions_by_hand(site):
    """alpha of 6xx in milli order (listed a0, a2, a1); alpha of 7xx its own group, the
    continuation of 6xx alpha listed there dropped; kap-pa without its piece under hex
    0e (alias `kap-pa`, split from the right)."""
    site.run()
    a = "alpha-a1a1a1a1a1a1"
    assert pieces(site) == [
        [f"{a}-20211000-120000", f"{a}-20211001-120001", f"{a}-20211002-120002"],
        ["beta-b2b2b2b2b2b2-20211000-120000"],
        [f"{a}-20211005-120005", f"{a}-20211006-120006"],
        ["zeta-f6f6f6f6f6f6-20211000-120000", "zeta-f6f6f6f6f6f6-20211001-120001"],
        ["eta-0a0a0a0a0a0a-20211000-120000", "eta-0a0a0a0a0a0a-20211001-120001"],
        ["theta-0b0b0b0b0b0b-20211000-120000", "theta-0b0b0b0b0b0b-20211001-120001"],
        ["iota-0c0c0c0c0c0c-20211000-120000", "iota-0c0c0c0c0c0c-20211001-120001"],
        ["kap-pa-0d0d0d0d0d0d-20211000-120000", "kap-pa-0d0d0d0d0d0d-20211001-120001"],
    ]


def test_head_shards_cut_after_the_group_reaching_entries_per_shard(site):
    """Groups of 3 | 1, 3 | 3 | 2, 2 | 2, 2 | 2, 2 | 1 entries, across subsets; the
    second head shard holds the heads `missing`, `empty`, `malformed`."""
    site.run()
    tree = site.tree("sessions")
    entries = [tree[p]["entries"] for p in sorted(tree)]
    assert [sum(x.values()) for x in entries] == [3, 4, 3, 4, 4, 4, 1]
    assert entries[1] == {"ok": 1, "missing": 1, "empty": 1, "malformed": 1}
    assert all(list(x) == ["ok", "missing", "empty", "malformed"] for x in entries)
    assert sorted(tree) == [f"sessions/{k:06d}/sessions.json" for k in range(7)]


@pytest.mark.parametrize("per_shard", [1, 2, 5, 100])
def test_head_shards_follow_the_transcription(site, per_shard):
    site.run(entries_per_shard=per_shard)
    assert_outputs(site, site.expected(entries_per_shard=per_shard))


def test_one_head_request_per_entry_none_below_min_version(site):
    """Each entry's `.jsonl` asked once with Range bytes=0-1000; the entry of 6.9 and the
    repeat of an index asked nothing."""
    site.run()
    ex = site.expected()
    assert sorted(site.heads_asked()) == sorted(ex.heads)
    assert len(ex.heads) == 23
    assert not [k for k in site.asked() if "gamma" in k[1]]


def test_heads_beyond_head_bytes_give_the_same_outputs(site):
    """head_bytes 64: no line 0 fits; every head comes from the fallback."""
    site.run(head_bytes=64)
    assert_outputs(site, site.expected(head_bytes=64))
    heads = [
        k for k in site.asked() if k[1].endswith(".jsonl") and k[2] == "bytes=0-63"
    ]
    assert len(heads) == 23


@pytest.mark.parametrize("workers", [1, 3, 5])
def test_heads_fetched_on_head_workers_threads(make_site, workers):
    """Nine missing heads, each held 0.15 s: at most head_workers requests at once, and
    as many."""
    eps = [PoolEp(relpath("7.1", f"a{k}", "ab" * 6, 0), jsonl=None) for k in range(9)]
    site = make_site(episodes={"7xx": eps})
    for e in eps:
        site.stalls[("GET", site.path(e.relpath, ".jsonl"), "bytes=0-1000")] = 0.15
    site.run(head_workers=workers, entries_per_shard=100)
    assert site.server.max_active == workers
    assert site.tree("sessions") == {"sessions/000000/sessions.json": {"sessions": [], "entries": {"ok": 0, "missing": 9, "empty": 0, "malformed": 0}}}  # fmt: skip


def failing_world() -> dict:
    """7xx lists x0 (head shard 0), then a0, b0, a1 (head shard 1: a0, a1, b0)."""
    x = chain("7.1", "x", "ab" * 6, ["w" * 6] * 3)
    a = chain("7.1", "a", "ab" * 6, ["w" * 6] * 2)
    b = chain("7.1", "b", "ab" * 6, ["w" * 6])
    return {"7xx": [*x, a[0], b[0], a[1]]}


def test_failing_head_raises_the_first_in_head_shard_order(make_site):
    """b0 and a1 answer 403; a1, first in head shard 1's order though listed after b0,
    is held so that b0 fails first: a1's exception is raised; head shard 0 written, head
    shard 1 not; no shard, no counts."""
    site = make_site(episodes=failing_world())
    a1, b0 = relpath("7.1", "a", "ab" * 6, 1), relpath("7.1", "b", "ab" * 6, 0)
    for rp in (a1, b0):
        site.faults[("GET", site.path(rp, ".jsonl"), "bytes=0-1000")] = [403]
    site.stalls[("GET", site.path(a1, ".jsonl"), "bytes=0-1000")] = 0.3
    site.aborted(entries_per_shard=3, head_workers=3)
    assert type(site.error) is RuntimeError
    assert str(site.error) == f"GET {site.url_stem(a1)}.jsonl: HTTP 403"
    assert sorted(site.raw("sessions")) == ["sessions/000000/sessions.json"]
    assert not (site.out / "shards").exists() and not site.completed()


def test_rerun_continues_at_the_first_missing_head_shard(make_site):
    """After the failure: no head request for head shard 0's entries; the outputs of an
    uninterrupted run."""
    site = make_site(episodes=failing_world())
    b0 = relpath("7.1", "b", "ab" * 6, 0)
    site.faults[("GET", site.path(b0, ".jsonl"), "bytes=0-1000")] = [403]
    site.aborted(entries_per_shard=3)
    site.faults.clear()
    since = len(site.server.requests)
    site.run(entries_per_shard=3)
    assert_outputs(site, site.expected(entries_per_shard=3))
    assert sorted(site.heads_asked(since)) == sorted([*site.expected(entries_per_shard=3).entries[1]])  # fmt: skip


@pytest.mark.parametrize("h", [0, 2, 4])
def test_rerun_after_a_kill_inside_the_head_pass(make_site, h):
    """A complete run (the draw world of test_plan_draws.py, five head shards) turned
    into what a kill in head shard h leaves: head shards 0..h-1, a staged head shard h,
    no shards, run.json without counts. The rerun asks the heads of head shards h.. only
    and ends as an uninterrupted run."""
    site = make_site(episodes=draw_world())
    site.run()
    ex, root = site.expected(), site.out / "sessions"
    assert len(ex.sessions) == 5
    for name in sorted(p.name for p in root.iterdir())[h:]:
        shutil.rmtree(root / name)
    (root / f"{h:06d}.part").mkdir()
    (root / f"{h:06d}.part" / "sessions.json").write_text("{")
    shutil.rmtree(site.out / "shards")
    run = site.run_json()
    for key in ("episodes", "sessions", "clips", "comparisons"):
        del run[key]
    (site.out / "run.json").write_text(json.dumps(run))
    since = len(site.server.requests)
    site.run()
    assert_outputs(site, ex)
    assert sorted(site.heads_asked(since)) == sorted(
        r for x in ex.entries[h:] for r in x
    )


def test_running_heads_finish_before_the_exception(make_site):
    """a0 answers 403 at once, b0 (same head shard) is held 0.5 s: b0's request is
    answered, not dropped, before a0's exception is raised."""
    site = make_site(episodes=failing_world())
    a0, b0 = relpath("7.1", "a", "ab" * 6, 0), relpath("7.1", "b", "ab" * 6, 0)
    site.faults[("GET", site.path(a0, ".jsonl"), "bytes=0-1000")] = [403]
    held = ("GET", site.path(b0, ".jsonl"), "bytes=0-1000")
    site.stalls[held] = 0.5
    start = time.monotonic()
    site.aborted(entries_per_shard=100, head_workers=3)
    assert str(site.error) == f"GET {site.url_stem(a0)}.jsonl: HTTP 403"
    assert time.monotonic() - start >= 0.5
    (req,) = [r for r in site.server.requests if (r.method, r.path, r.headers.get("Range")) == held]  # fmt: skip
    assert not req.gone
