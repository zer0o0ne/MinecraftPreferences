"""(S) of specs/preference-pool.md: `sessions` on hand groups — the `ok` entries by (milli,
relpath), one current chain: a session start (empty stats) starts a session, an entry joins
the current chain iff d > 0, tick_ms d (1 - tick_slack) <= its milli gap (exact) and no
stat outside resetting_stats decreased or vanished, else it starts an orphan chain, which
is dropped."""

import random
from fractions import Fraction

import pytest

from mcpref.pool.run import load_config
from mcpref.pool.sessions import sessions

from .pool_site import write_config
from .pool_world import relpath

Z = "minecraft.custom:minecraft.time_since_rest"


def rp(k: int) -> str:
    """Names in the order of k."""
    return relpath("7.1", "al", "ab" * 6, k)


def ok(k: int, stats: dict, tick: int, milli: int) -> tuple[str, dict]:
    return rp(k), {"status": "ok", "stats": stats, "serverTick": tick, "milli": milli}


@pytest.fixture
def cut(tmp_path):
    """sessions(entries, cfg) of a config with `changes` (tick_ms 50, tick_slack 0.03,
    resetting_stats [Z, ...] by default), as lists of k."""

    def run(entries, **changes) -> list[list[int]]:
        cfg = load_config(write_config(tmp_path / "pool.yaml", **changes))
        got = sessions(entries, cfg)
        return [[int(r.rsplit("-", 2)[1]) - 20211000 for r in s] for s in got]

    return run


START = ok(0, {}, 100, 1000)
P1 = ok(1, {"a": 5, "b": 2, Z: 9}, 200, 6000)  # 50 * 100 * 0.97 = 4850 <= 5000


def test_session_of_several_pieces(cut):
    p2 = ok(2, {"a": 6, "b": 2, Z: 9}, 220, 6970)  # 50 * 20 * 0.97 = 970: exactly met
    assert cut([START, P1, p2]) == [[0, 1, 2]]


def test_entries_are_taken_by_milli_not_in_pool_order(cut):
    p2 = ok(2, {"a": 6, "b": 2, Z: 9}, 220, 6970)
    for seed in range(5):
        entries = [START, P1, p2]
        random.Random(seed).shuffle(entries)
        assert cut(entries) == [[0, 1, 2]]


def test_start_inside_a_group(cut):
    """A session start ends the current chain, even where it could join it."""
    s2 = ok(2, {}, 300, 11000)
    p3 = ok(3, {"a": 1}, 310, 11500)
    assert cut([START, P1, s2, p3]) == [[0, 1], [2, 3]]


def test_orphan_chain_dropped_with_its_continuation(cut):
    o0 = ok(0, {"a": 1}, 10, 100)
    o1 = ok(1, {"a": 2}, 20, 600)  # joins o0
    s2, p3 = ok(2, {}, 100, 5000), ok(3, {"a": 3}, 110, 5500)
    assert cut([o0, o1, s2, p3]) == [[2, 3]]


def test_one_current_chain_per_group(cut):
    """p1 fails against the session (d < 0) and starts an orphan chain; p2, which could
    join either, joins the orphan chain and is dropped with it."""
    p1 = ok(1, {"a": 5}, 90, 2000)
    p2 = ok(2, {"a": 6}, 120, 3500)  # vs p1: d 30, 1455 <= 1500; vs START: d 20, 970
    assert cut([START, p1, p2]) == [[0]]


@pytest.mark.parametrize(
    "p2",
    [ok(2, {"a": 3, "b": 2, Z: 9}, 300, 20000), ok(2, {"a": 6, "b": 2, Z: 9}, 150, 20000)],
    ids=["stat", "tick"],
)  # fmt: skip
def test_compared_with_the_last_piece_of_the_chain(cut, p2):
    """p2 passes against the first piece (empty stats, d > 0), not against the last."""
    assert cut([START, P1, p2]) == [[0, 1]]


BASE = {"a": 5, "b": 3, Z: 9}  # vs P1: no stat outside Z decreased or vanished
JOINS = {
    "time bound exactly met": (BASE, 300, 10850),  # d 100: 50 * 100 * 0.97 = 4850
    "equal stats": ({"a": 5, "b": 2, Z: 9}, 300, 10850),
    "a key of Z decreasing": (BASE | {Z: 1}, 300, 10850),
    "a key of Z vanished": ({"a": 5, "b": 3}, 300, 10850),
    "a new key": (BASE | {"c": 1}, 300, 10850),
    "float stats": ({"a": 5.5, "b": 2.0, Z: 9}, 300, 10850),
}
ORPHANS = {
    "d = 0": (BASE, 200, 10850),
    "d < 0": (BASE, 199, 10850),
    "time bound missed": (BASE, 300, 10849),
    "a decreasing stat": (BASE | {"a": 4}, 300, 10850),
    "a vanished key": ({"a": 5, Z: 9}, 300, 10850),
}


@pytest.mark.parametrize("stats, tick, milli", JOINS.values(), ids=JOINS)
def test_joins(cut, stats, tick, milli):
    assert cut([START, P1, ok(2, stats, tick, milli)]) == [[0, 1, 2]]


@pytest.mark.parametrize("stats, tick, milli", ORPHANS.values(), ids=ORPHANS)
def test_orphans(cut, stats, tick, milli):
    assert cut([START, P1, ok(2, stats, tick, milli)]) == [[0, 1]]


@pytest.mark.parametrize("gap, joined", [(5000, True), (4999, False)])
def test_time_bound_without_slack(cut, gap, joined):
    """tick_slack 0: 50 * 100 = 5000."""
    got = cut([START, P1, ok(2, BASE, 300, 6000 + gap)], tick_slack=0)
    assert got == [[0, 1, 2] if joined else [0, 1]]


@pytest.mark.parametrize("gap, joined", [(200, True), (199, False)])
def test_time_bound_scales_with_tick_ms(cut, gap, joined):
    """tick_ms 2, tick_slack 0: 2 * 100 = 200."""
    got = cut([START, P1, ok(2, BASE, 300, 6000 + gap)], tick_ms=2, tick_slack=0)
    assert got == [[0, 1, 2] if joined else [0, 1]]


@pytest.mark.parametrize("gap, joined", [(105, True), (104, False)])
def test_time_bound_in_exact_arithmetic(cut, gap, joined):
    """tick_slack 0.7, d = 7: 50 * 7 * 3/10 = 105 exactly; in floats it exceeds 105."""
    assert 50 * 7 * (1 - 0.7) > 105
    got = cut([START, P1, ok(2, BASE, 207, 6000 + gap)], tick_slack=0.7)
    assert got == [[0, 1, 2] if joined else [0, 1]]


def test_tick_slack_exact_value_is_the_shortest_repr(tmp_path):
    """Notation: `0.69999999999999999` reads as the float 0.7, exact value 7/10: d = 7,
    gap 105 joins (the decimal as written, or the float's own value, gives a bound above
    105)."""
    written = "0.69999999999999999"
    assert 350 * (1 - Fraction(written)) > 105 and 350 * (1 - Fraction(0.7)) > 105
    path = write_config(tmp_path / "pool.yaml", tick_slack=0.7)
    path.write_text(path.read_text().replace("tick_slack: 0.7\n", f"tick_slack: {written}\n"))  # fmt: skip
    cfg = load_config(path)
    assert cfg.tick_slack == 0.7
    got = sessions([START, P1, ok(2, BASE, 207, 6105)], cfg)
    assert len(got) == 1 and len(got[0]) == 3


def test_resetting_stats_from_the_config(cut):
    """Without Z in resetting_stats, its decrease makes an orphan."""
    p2 = ok(2, BASE | {Z: 1}, 300, 10850)
    assert cut([START, P1, p2], resetting_stats=["other"]) == [[0, 1]]
    assert cut([START, P1, p2], resetting_stats=["other", Z]) == [[0, 1, 2]]


@pytest.mark.parametrize("names", [(1, 2), (2, 1)])
def test_ties_of_milli_broken_by_relpath(cut, names):
    """Two session starts with the same milli: the later relpath is the current chain
    when p3 comes."""
    x, y = (ok(k, {}, 100, 1000) for k in names)
    p3 = ok(3, {"a": 1}, 110, 1500)
    assert cut([x, p3, y]) == [[1], [2, 3]]


def test_entries_not_ok_are_ignored(cut):
    """They lie anywhere in pool order and carry no fields."""
    gone = [(rp(10 + i), {"status": s}) for i, s in enumerate(("missing", "empty", "malformed"))]  # fmt: skip
    assert cut([gone[0], START, gone[1], P1, gone[2]]) == [[0, 1]]
    assert cut(gone) == []


def test_group_without_a_session_start(cut):
    assert cut([ok(0, {"a": 1}, 10, 100), ok(1, {"a": 2}, 20, 600)]) == []
