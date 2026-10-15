"""Progress bars of specs/perception-data.md (Interface; AGENTS.md Progress) and the end of
stderr on an abort ((8): the bars closed, the exception re-raised): `indices`, `collect`
(total N, initial value, advance and postfix at each commit), `collect: already done` on a
rerun with nothing left to do, no bar inside an episode."""

import itertools
import json

import pytest
from progress_bars import Bar, assert_progress, screen, states

from .collect_gamma import Classifier

INDICES = Bar("indices", 5, 5, "file")


def collect_bar(site, n: int) -> Bar:
    ex = site.expected(n_clips=n)
    return Bar("collect", len(ex.clips), n, "clip", f"{len(ex.records)} episodes")


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("dry", [True, False], ids=["pools_dry", "cut"])
def test_bars_totals_and_final_counts(site, capfd, workers, dry):
    """N = 100 ends below N (the pools run dry); the cut ends at N."""
    n = 100 if dry else site.cut()
    site.run(n_clips=n, workers=workers)
    out, err = capfd.readouterr()
    assert out == "" and screen(err) == [INDICES, collect_bar(site, n)]
    assert {b.desc for b in states(err)} == {"indices", "collect"}
    assert_progress(err)


def test_collect_bar_advances_by_the_clips_of_each_commit(site, capfd):
    """Per committed episode: advanced by its committed clips, then postfix `<k> episodes`."""
    site.run()
    records = site.expected().records
    done = list(
        itertools.accumulate((len(r["clips"] or []) for r in records), initial=0)
    )
    after = {(done[k], f"{k} episodes") for k in range(len(records) + 1)}
    between = {(done[k + 1], f"{k} episodes") for k in range(len(records))}
    shown = states(capfd.readouterr().err, "collect")
    assert after <= {(b.n, b.postfix) for b in shown} <= after | between | {(0, None)}
    assert {(b.total, b.unit) for b in shown} == {(100, "clip")}


def test_rerun_bar_starts_at_the_committed_clips(site, capfd):
    site.aborted(site.failing_gamma())
    records = site.read_jsonl("episodes.jsonl")
    done = sum(len(r["clips"] or []) for r in records)
    capfd.readouterr()
    site.run()
    err = capfd.readouterr().err
    shown = states(err, "collect")
    assert shown[0].n == done and {b.n for b in shown if b.postfix is None} <= {done}
    assert (done, f"{len(records)} episodes") in {(b.n, b.postfix) for b in shown}
    assert all(a.n <= b.n for a, b in itertools.pairwise(shown))
    assert screen(err) == [INDICES, collect_bar(site, 100)]


def strip_counts(site) -> None:
    """run.json as a run killed after its last commit leaves it."""
    run = site.run_json()
    del run["episodes"], run["clips"]
    (site.out / "run.json").write_text(json.dumps(run))


@pytest.mark.parametrize(
    "dry, killed",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=[
        "n_committed",
        "n_committed_killed",
        "pools_exhausted",
        "pools_exhausted_killed",
    ],
)
def test_rerun_with_nothing_left_prints_already_done(site, capfd, dry, killed):
    """N already committed, or every pool exhausted by the committed draws (N = 100: 8
    clips); killed: run.json without counts. No bar, no work, the counts written."""
    n = 100 if dry else site.cut()
    site.run(n_clips=n)
    before, asked = site.tree(), len(site.server.requests)
    if killed:
        strip_counts(site)
    capfd.readouterr()
    gamma = site.run(n_clips=n)
    err = capfd.readouterr().err
    assert screen(err) == [INDICES, "collect: already done"]
    assert states(err, "collect") == [] and gamma.calls == []
    assert site.tree() == before
    assert {r.path for r in site.server.requests[asked:]} <= {
        f"/idx/{s}.json" for s in site.etags
    }


def assert_bars_last(err: str) -> list:
    """`collect` prints nothing below its closed bars ((8): the message is `main`'s);
    returns the lines shown."""
    assert_progress(err)
    shown = screen(err)
    assert shown and all(isinstance(x, Bar) for x in shown), repr(err)
    return shown


@pytest.mark.parametrize("workers", [1, 3])
def test_classifier_exception_closes_the_bar_and_propagates(site, capfd, workers):
    gamma = site.aborted(Classifier(site.world, fail_at=0), workers=workers)
    assert site.error is gamma.raised and not site.completed()
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown == [INDICES, Bar("collect", 0, 100, "clip", "0 episodes")]


def test_abort_lets_the_running_episodes_finish(site):
    """One worker, gamma raising at its first call: the running decodes end, at most one
    more episode was started."""
    site.aborted(Classifier(site.world, fail_at=0))
    starts, ends = site.decodes(), site.decodes("end")
    assert len(starts) == len(ends) and 1 <= len({d["key"] for d in starts}) <= 2


def test_http_abort_while_planning_closes_the_bars_and_propagates(site, capfd):
    first = next(r for r in site.expected().records if r["reason"] != "version")
    path = site.path(first["subset"], first["relpath"], ".jsonl")
    site.faults[site.head(path)] = [503] * 5
    site.aborted()
    assert not site.completed()
    shown = assert_bars_last(capfd.readouterr().err)
    # when `collect` is first drawn is not fixed: before or after this first fetch
    assert shown[0] == INDICES and [(b.desc, b.n) for b in shown[1:]] in (
        [],
        [("collect", 0)],
    )


def test_pass_2_failure_closes_the_bar_and_propagates(site, capfd):
    site.behaviours = {"a6": ["ok", "exit1"]}
    site.aborted()
    assert not site.completed()
    shown = assert_bars_last(capfd.readouterr().err)
    assert shown[0] == INDICES and [b.desc for b in shown] == ["indices", "collect"]


def test_rerun_abort_leaves_the_indices_bar_last(site, capfd):
    site.run(n_clips=3)
    before = site.tree()
    capfd.readouterr()
    site.aborted(n_clips=3, seed=1)
    assert assert_bars_last(capfd.readouterr().err) == [INDICES]
    assert site.tree() == before
