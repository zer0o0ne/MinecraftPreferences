"""Progress bars of specs/preference-pool.md (Interface; AGENTS.md Progress) and the end of
stderr on an abort ((9'): the bars closed, the exception re-raised, no message): `indices`,
`sessions` (total the entries, initial those of the complete head shards), `resume`,
`collect` (total n_clips, initial c, postfix `<n_e> episodes, <n_s> sessions`) and their
lines."""

import itertools

import pytest
from progress_bars import Bar, assert_progress, screen, states

from .pool_world import PoolEp, relpath

INDICES = Bar("indices", 5, 5, "file")
SESSIONS = Bar("sessions", 23, 23, "episode")
RESUME = "resume: nothing to do"
DONE = {d: f"{d}: already done" for d in ("indices", "sessions", "collect")}


def collect_bar(ex, n: int, upto: int | None = None) -> Bar:
    """`collect` closed after shards 0..upto-1 of `ex` (all if None); above its total,
    tqdm draws it without one."""
    shards = ex.shards[:upto]
    c = shards[-1].clips if shards else 0
    e = sum(len(s.records) for s in shards)
    return Bar("collect", c, n if c <= n else None, "clip", f"{e} episodes, {len(shards)} sessions")  # fmt: skip


def resume_bar(h: int) -> Bar:
    return Bar("resume", h, h, "shard")


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("n", [100, 4], ids=["pools_dry", "cut"])
def test_bars_of_a_first_run(site, capfd, workers, n):
    """n_clips 100 ends below it (the pools run dry); 4 ends at it (seed 0)."""
    site.run(n_clips=n, workers=workers)
    out, err = capfd.readouterr()
    assert out == ""
    assert screen(err) == [INDICES, SESSIONS, RESUME, collect_bar(site.expected(n_clips=n), n)]  # fmt: skip
    assert_progress(err)


def test_collect_bar_may_end_above_n_clips(site, capfd):
    """n_clips 5: the shard reaching it brings c to 7."""
    site.run(n_clips=5)
    shown = screen(capfd.readouterr().err)
    assert shown[-1] == collect_bar(site.expected(n_clips=5), 5) and shown[-1].n == 7


def test_collect_bar_advances_by_the_clips_of_each_shard(site, capfd):
    """Per shard: advanced by its clips, the postfix set (at creation too)."""
    site.run()
    ex = site.expected()
    done = [0] + [s.clips for s in ex.shards]
    seen = list(itertools.accumulate((len(s.records) for s in ex.shards), initial=0))
    post = [f"{seen[k]} episodes, {k} sessions" for k in range(len(done))]
    after = {(done[k], post[k]) for k in range(len(done))}
    between = {(done[k + 1], post[k]) for k in range(len(ex.shards))}
    between |= {(done[k], post[k + 1]) for k in range(len(ex.shards))}
    shown = states(capfd.readouterr().err, "collect")
    assert after <= {(b.n, b.postfix) for b in shown} <= after | between | {(0, None)}
    assert {(b.total, b.unit) for b in shown} == {(100, "clip")}


def test_sessions_bar_advances_per_head(site, capfd):
    site.run()
    shown = states(capfd.readouterr().err, "sessions")
    assert shown[0] == Bar("sessions", 0, 23, "episode") and shown[-1] == SESSIONS
    assert {(b.total, b.unit, b.postfix) for b in shown} == {(23, "episode", None)}


def assert_rerun_bars(err: str) -> None:
    """Each bar never going back, one total and unit."""
    for desc in dict.fromkeys(b.desc for b in states(err)):
        drawn = states(err, desc)
        assert all(a.n <= b.n for a, b in itertools.pairwise(drawn)), drawn
        assert len({(b.total, b.unit) for b in drawn}) == 1, drawn


def abort_in_draw(site) -> int:
    """A run aborted in a draw (a video failing after its retries); the shards written."""
    ex = site.expected()
    k = next(j for j, s in enumerate(ex.shards) if j >= 2 and s.records[0]["reason"] is None)  # fmt: skip
    path = site.path(ex.shards[k].records[0]["relpath"], ".mp4")
    site.faults[site.head(path)] = [503] * 2
    site.aborted()
    site.faults.clear()
    return k


def test_rerun_after_an_abort_in_a_draw(site, capfd):
    """`resume` reads the shards; `collect` starts at their clips, episodes, sessions."""
    k = abort_in_draw(site)
    capfd.readouterr()
    site.run()
    err, ex = capfd.readouterr().err, site.expected()
    assert screen(err) == [DONE["indices"], DONE["sessions"], resume_bar(k), collect_bar(ex, 100)]  # fmt: skip
    assert_rerun_bars(err)
    shown = states(err, "collect")
    assert shown[0].n == ex.shards[k - 1].clips and collect_bar(ex, 100, k) in shown


def test_rerun_after_an_abort_in_the_head_pass(site, capfd):
    """Head shards 0..2 complete (10 entries): `sessions` starts at 10 of 23."""
    first = site.expected().entries[3][0]
    site.faults[("GET", site.path(first, ".jsonl"), "bytes=0-1000")] = [403]
    site.aborted()
    capfd.readouterr()
    site.run()
    err = capfd.readouterr().err
    assert screen(err) == [DONE["indices"], SESSIONS, RESUME, collect_bar(site.expected(), 100)]  # fmt: skip
    assert states(err, "sessions")[0].n == 10
    assert_rerun_bars(err)


@pytest.mark.parametrize("n", [100, 5], ids=["pools_exhausted", "n_reached"])
def test_rerun_with_nothing_left(site, capfd, n):
    """`collect: already done`, no work, the outputs unchanged."""
    site.run(n_clips=n)
    before, h = site.raw(), len(site.expected(n_clips=n).shards)
    capfd.readouterr()
    site.run(n_clips=n)
    err = capfd.readouterr().err
    assert screen(err) == [DONE["indices"], DONE["sessions"], resume_bar(h), DONE["collect"]]  # fmt: skip
    assert site.raw() == before


def test_no_entry(make_site, capfd):
    """Empty indices: `sessions: nothing to do`, `collect: nothing to do`; again on a
    rerun; the counts empty."""
    site = make_site(episodes={})
    site.run()
    lines = ["sessions: nothing to do", RESUME, "collect: nothing to do"]
    assert screen(capfd.readouterr().err) == [INDICES, *lines]
    run = site.run_json()
    assert [run[k] for k in ("episodes", "sessions", "clips", "comparisons")] == [{}, 0, 0, 0]  # fmt: skip
    site.run()
    assert screen(capfd.readouterr().err) == [DONE["indices"], *lines]


def test_no_session(make_site, capfd):
    """Entries, none in a session: the heads fetched, `collect: nothing to do`; on a rerun
    `sessions: already done`."""
    eps = [PoolEp(relpath("7.1", f"a{k}", "ab" * 6, 0), jsonl=None) for k in range(2)]
    site = make_site(episodes={"7xx": eps})
    site.run()
    bar = Bar("sessions", 2, 2, "episode")
    assert screen(capfd.readouterr().err) == [INDICES, bar, RESUME, "collect: nothing to do"]  # fmt: skip
    site.run()
    assert screen(capfd.readouterr().err) == [DONE["indices"], DONE["sessions"], RESUME, "collect: nothing to do"]  # fmt: skip


def assert_bars_last(err: str) -> list:
    """`collect` prints nothing below its closed bars (the message is `main`'s)."""
    assert_rerun_bars(err)
    shown = screen(err)
    assert shown and isinstance(shown[-1], Bar), repr(err)
    return shown


def test_worker_exception_closes_the_bar_and_propagates(site, capfd):
    k = abort_in_draw(site)
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown == [INDICES, SESSIONS, RESUME, collect_bar(site.expected(), 100, k)]


def test_head_exception_closes_the_bar_and_propagates(site, capfd):
    """A head answering 403 in head shard 3: `sessions` closed at the heads fetched."""
    first = site.expected().entries[3][0]
    site.faults[("GET", site.path(first, ".jsonl"), "bytes=0-1000")] = [403]
    site.aborted()
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown[0] == INDICES and len(shown) == 2 and shown[1].desc == "sessions"
    assert 10 <= shown[1].n < 14 and shown[1].total == 23


def test_rerun_check_abort_draws_no_bar(site, capfd):
    site.run(n_clips=3)
    capfd.readouterr()
    site.aborted(n_clips=3, seed=1)
    assert capfd.readouterr().err == ""
