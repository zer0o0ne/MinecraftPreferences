"""Progress bars of specs/perception-data.md (Interface; AGENTS.md Progress) and the end of
stderr on an abort ((9): the bars closed, the exception re-raised, no message): `indices`,
`resume`, `collect` (total N, initial c, advance and postfix per shard) and their lines."""

import itertools

import pytest
from progress_bars import Bar, assert_progress, screen, states

from .collect_gamma import Classifier

INDICES, RESUME = Bar("indices", 5, 5, "file"), "resume: nothing to do"
IDLE = {"indices": "indices: already done", "collect": "collect: already done"}


def collect_bar(ex, n: int, upto: int | None = None) -> Bar:
    """`collect` closed after shards 0..upto-1 of `ex` (all if None)."""
    shards = ex.shards[:upto]
    c = shards[-1].clips if shards else 0
    return Bar(
        "collect", c, n, "clip", f"{sum(len(s.records) for s in shards)} episodes"
    )


def resume_bar(h: int) -> Bar:
    return Bar("resume", h, h, "shard")


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("dry", [True, False], ids=["pools_dry", "cut"])
def test_bars_of_a_first_run(site, capfd, workers, dry):
    """N = 100 ends below N (the pools run dry); the cut ends at N. No other bar."""
    n = 100 if dry else 3
    site.run(n_clips=n, workers=workers)
    out, err = capfd.readouterr()
    assert out == "" and screen(err) == [INDICES, RESUME, collect_bar(site.expected(n_clips=n), n)]  # fmt: skip
    assert {b.desc for b in states(err)} == {"indices", "collect"}
    assert_progress(err)


def shard_states(ex) -> tuple[set, set]:
    """(n, postfix) of `collect` after each shard, and those possible in between."""
    done = [0] + [s.clips for s in ex.shards]
    seen = list(itertools.accumulate((len(s.records) for s in ex.shards), initial=0))
    after = {(done[k], f"{seen[k]} episodes") for k in range(len(done))}
    between = {(done[k + 1], f"{seen[k]} episodes") for k in range(len(ex.shards))}
    between |= {(done[k], f"{seen[k + 1]} episodes") for k in range(len(ex.shards))}
    return after, between


def test_collect_bar_advances_by_the_clips_of_each_shard(site, capfd):
    """Per shard: advanced by its clips, postfix `<E> episodes` (set at creation too)."""
    site.run()
    after, between = shard_states(site.expected())
    shown = states(capfd.readouterr().err, "collect")
    assert after <= {(b.n, b.postfix) for b in shown} <= after | between | {(0, None)}
    assert {(b.total, b.unit) for b in shown} == {(100, "clip")}


def assert_rerun_bars(err: str) -> None:
    """Each bar drawn once: never going back, one total and unit; closed bars last."""
    for desc in dict.fromkeys(b.desc for b in states(err)):
        drawn = states(err, desc)
        assert all(a.n <= b.n for a, b in itertools.pairwise(drawn)), drawn
        assert len({(b.total, b.unit) for b in drawn}) == 1, drawn


def test_rerun_after_an_abort(site, capfd):
    """Aborted in draw 2: the rerun reads 2 shards (`resume`), `collect` starts at their
    clips and episodes."""
    site.aborted(site.failing_gamma(2))
    capfd.readouterr()
    site.run()
    err, ex = capfd.readouterr().err, site.expected()
    assert screen(err) == [IDLE["indices"], resume_bar(2), collect_bar(ex, 100)]
    assert_rerun_bars(err)
    resume, shown = states(err, "resume"), states(err, "collect")
    assert resume[0].n == 0 and resume[-1] == resume_bar(2)
    assert shown[0].n == ex.shards[1].clips and {b.n for b in shown if b.postfix is None} <= {shown[0].n}  # fmt: skip
    assert collect_bar(ex, 100, 2) in shown


@pytest.mark.parametrize("dry", [True, False], ids=["pools_exhausted", "n_reached"])
def test_rerun_with_nothing_left(site, capfd, dry):
    """N reached, or every pool exhausted by the shards: `collect: already done`, no work,
    the shards unchanged, the counts written."""
    n = 100 if dry else 3
    site.run(n_clips=n)
    before, h = site.raw(), len(site.expected(n_clips=n).shards)
    capfd.readouterr()
    gamma = site.run(n_clips=n)
    err = capfd.readouterr().err
    assert screen(err) == [IDLE["indices"], resume_bar(h), IDLE["collect"]]
    assert states(err, "collect") == [] and gamma.calls == []
    assert site.raw() == before and site.completed()


def test_no_subset_active_from_the_start(make_site, capfd):
    """Empty indices: `collect: nothing to do`, no shard, the counts empty; again on a
    rerun."""
    site = make_site(episodes={})
    site.run()
    assert screen(capfd.readouterr().err) == [INDICES, RESUME, "collect: nothing to do"]
    assert site.complete() == [] and (site.run_json()["episodes"], site.run_json()["clips"]) == ({}, 0)  # fmt: skip
    site.run()
    assert screen(capfd.readouterr().err) == [IDLE["indices"], RESUME, "collect: nothing to do"]  # fmt: skip


def test_indices_bar_counts_only_the_missing_indices(site, capfd):
    site.indices["8xx"] = None
    site.aborted()
    assert screen(capfd.readouterr().err) == [Bar("indices", 2, 5, "file")]
    del site.indices["8xx"]
    site.run()
    assert screen(capfd.readouterr().err)[:2] == [Bar("indices", 3, 3, "file"), RESUME]


def assert_bars_last(err: str) -> list:
    """`collect` prints nothing below its closed bars ((9): the message is `main`'s);
    returns the lines shown."""
    assert_rerun_bars(err)
    shown = screen(err)
    assert shown and isinstance(shown[-1], Bar), repr(err)
    assert [x for x in shown if isinstance(x, Bar)] == [states(err, d)[-1] for d in dict.fromkeys(b.desc for b in states(err))]  # fmt: skip
    return shown


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("k", [0, 3])
def test_classifier_exception_closes_the_bar_and_propagates(site, capfd, workers, k):
    """Raised in draw k: the same exception, after shards 0..k-1."""
    gamma = site.aborted(site.failing_gamma(k), workers=workers)
    assert site.error is gamma.raised and not site.completed()
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown == [INDICES, RESUME, collect_bar(site.expected(), 100, k)]


def test_abort_lets_the_running_work_finish(site):
    """Three workers, gamma raising at draw 0: every decode started has ended."""
    site.aborted(Classifier(site.world, fail_at=0), workers=3)
    assert len(site.decodes()) == len(site.decodes("end"))


def test_http_abort_while_planning_closes_the_bars_and_propagates(site, capfd):
    path = site.path("data/7.1/b7.mp4", "-options.json")  # draw 3
    site.faults[site.head(path)] = [503] * 5
    site.aborted()
    assert not site.completed()
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown == [INDICES, RESUME, collect_bar(site.expected(), 100, 3)]


def test_pass_2_failure_closes_the_bar_and_propagates(site, capfd):
    site.behaviours = {"a6": ["ok", "exit1"]}  # draw 2
    site.aborted()
    assert type(site.error) is RuntimeError and not site.completed()
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown == [INDICES, RESUME, collect_bar(site.expected(), 100, 2)]


def test_rerun_check_abort_draws_no_bar(site, capfd):
    site.run(n_clips=3)
    capfd.readouterr()
    site.aborted(n_clips=3, seed=1)
    assert capfd.readouterr().err == ""
