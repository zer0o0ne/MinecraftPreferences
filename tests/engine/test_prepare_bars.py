"""The bars of `prepare` (specs/engine.md Interface, Progress bars): `lwjgl`, `engine.zip`
(`unit_scale=True`: its counts read here in tqdm's own size format) and `patch`: units,
totals, final counts; `reset()` at the start of every attempt; a non-200 attempt leaves
the count at 0; the bars closed above the error line on an abort."""

import re

import pytest
import tqdm
from progress_bars import Bar, assert_aborted, assert_progress, screen, states

from .jars import entry_count
from .prepare_site import JARS, SOURCE, ZIP, cut, status

SIZE = r"[\d.]+[kMGTPEZY]?"
SCALED = re.compile(
    rf"engine\.zip: +\d+%\|[^|]*\| (?P<n>{SIZE})/(?P<total>{SIZE}) "
    rf"\[[\d:]+<(?:[\d:]+|\?), (?:(?:{SIZE}|\?)B/s|{SIZE}s/B)\]"
)
fmt = tqdm.tqdm.format_sizeof


def scaled(err: str) -> list[tuple[str, str]]:
    """(count, total) of every drawn state of the `engine.zip` bar, in order."""
    drawn = (SCALED.fullmatch(s.rstrip(" ")) for s in re.split(r"[\r\n]", err))
    return [(m["n"], m["total"]) for m in drawn if m]


def test_bars_of_a_run(site):
    """lwjgl 2|M| files; engine.zip with total `engine_size`, created, reset by the one
    attempt, closed at the bytes written; patch over every entry of S and the jars."""
    site.fields["engine_size"] = 7_654_321
    r = site.run()
    assert r.code == 0, r.err
    assert_progress(r.err)
    entries = entry_count(SOURCE, JARS)
    shown = screen(r.err)
    assert shown[::2] == [
        Bar("lwjgl", 14, 14, "file"),
        Bar("patch", entries, entries, "entry"),
    ]
    assert len(shown) == 3 and SCALED.fullmatch(shown[1]), shown
    assert states(r.err, "lwjgl")[0] == Bar("lwjgl", 0, 14, "file")
    drawn = scaled(r.err)
    assert drawn[:2] == [("0.00", "7.65M")] * 2
    assert drawn[-1] == (fmt(len(ZIP)), "7.65M")


@pytest.mark.parametrize(
    "replies", [[cut(30_000)], [status(503)], [cut(30_000), status(500)]]
)
def test_engine_bar_reset_at_every_attempt(site, replies):
    """One 0 state at creation and one per attempt; the last count is Z's size alone."""
    site.cdn = list(replies)
    r = site.run()
    assert r.code == 0, r.err
    size = fmt(len(ZIP))
    drawn = scaled(r.err)
    assert [d for d in drawn if d[0] == "0.00"] == [("0.00", size)] * (len(replies) + 2)
    assert drawn[-1] == (size, size)


def test_engine_bar_after_5xx_exhausted(site):
    """Every attempt a 503: the bar is never advanced."""
    site.cdn = [status(503)] * 3
    r = site.run()
    assert_aborted(r.code, r.err)
    drawn = scaled(r.err)
    assert len(drawn) >= 5 and set(drawn) == {("0.00", fmt(len(ZIP)))}
    assert not states(r.err, "patch")


def test_lwjgl_bar_counts_verified_jars(site):
    """A mismatch at jar 8 of 14: the bar closed at 7, no later bar."""
    site.fields["lwjgl_sha256"]["lwjgl-stb"]["natives"] = "f" * 64
    r = site.run()
    assert assert_aborted(r.code, r.err) == [Bar("lwjgl", 7, 14, "file")]
    assert "engine.zip" not in r.err and "patch" not in r.err
