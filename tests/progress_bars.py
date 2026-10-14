"""Reader of tqdm 4.70.1 progress bars on stderr in tqdm's default format (AGENTS.md Code,
Progress; Library mapping `progress bars` of the specs). tqdm draws a state as '\\r', the
state, space padding; it draws at creation and, with leave=True, at close followed by '\\n';
states in between are drawn when its timing allows, or when a postfix is set."""

import itertools
import os
import re
from dataclasses import dataclass

import pytest

NUM = r" ?\d+\.\d\d"  # f"{rate:5.2f}"
# a rate below 1 per second is drawn inverted
RATE = rf"(?:(?:{NUM}|\?)(?P<unit>[a-z]+)/s|{NUM}s/(?P<inv>[a-z]+))"
END = rf"{RATE}(?:, (?P<postfix>.+))?\]"
TOTAL = re.compile(
    rf"(?P<desc>[^:]+): +\d+%\|[^|]*\| (?P<n>\d+)/(?P<total>\d+) "
    rf"\[[\d:]+<(?:[\d:]+|\?), {END}"
)
NO_TOTAL = re.compile(
    rf"(?P<desc>[^:]+): (?P<n>\d+)(?P<counted>[a-z]+) \[[\d:]+, {END}"
)
IDLE = (": nothing to do", ": already done")


@dataclass(frozen=True)
class Bar:
    """A drawn state: count n of total (None: no total), unit, postfix (None: none)."""

    desc: str
    n: int
    total: int | None
    unit: str
    postfix: str | None = None


def parse(state: str) -> Bar | None:
    """The bar state `state` draws (its padding ignored), else None."""
    state = state.rstrip(" ")
    if m := TOTAL.fullmatch(state):
        total = int(m["total"])
    elif (m := NO_TOTAL.fullmatch(state)) and m["counted"] in (m["unit"], m["inv"]):
        total = None
    else:
        return None
    return Bar(m["desc"], int(m["n"]), total, m["unit"] or m["inv"], m["postfix"])


def states(err: str, desc: str | None = None) -> list[Bar]:
    """Every state drawn of the bar `desc` (of every bar if None), in order."""
    bars = map(parse, re.split(r"[\r\n]", err))
    return [b for b in bars if b and desc in (None, b.desc)]


def screen(err: str) -> list[Bar | str]:
    """The lines a terminal shows for `err` (each line's text after its last '\\r', its
    padding ignored), as a Bar where it is one; `err` ends with a full line."""
    assert err == "" or err.endswith("\n"), repr(err)
    shown = [line.rsplit("\r", 1)[-1].rstrip(" ") for line in err.split("\n")[:-1]]
    return [parse(line) or line for line in shown]


def assert_progress(err: str) -> None:
    """The bars of one run, each drawn once: each starts at 0, never goes back, keeps its
    total and unit, and is closed with its last state left alone on a line of its own."""
    descs = list(dict.fromkeys(b.desc for b in states(err)))
    for desc in descs:
        drawn = states(err, desc)
        assert drawn[0].n == 0 and len({(b.total, b.unit) for b in drawn}) == 1, drawn
        assert all(a.n <= b.n for a, b in itertools.pairwise(drawn)), drawn
    closed = [line for line in screen(err) if isinstance(line, Bar)]
    assert closed == [states(err, d)[-1] for d in descs], repr(err)
    for line, shown in zip(err.split("\n"), screen(err)):
        parts = [parse(s) for s in line.split("\r") if s.strip(" ")]
        if isinstance(shown, Bar):
            assert all(b and b.desc == shown.desc for b in parts), repr(line)


def assert_aborted(code: int, err: str) -> list[Bar | str]:
    """Exit 1 with a one-line message as the last line of stderr, below the bars drawn before
    (`assert_progress`); returns the lines above the message as `screen` shows them."""
    assert code == 1 and err.endswith("\n") and "Traceback" not in err, repr(err)
    last = err[:-1].rsplit("\n", 1)[-1]
    assert last.strip() and "\r" not in last and parse(last) is None, repr(err)
    assert not last.endswith(IDLE), repr(err)
    assert_progress(err)
    return screen(err)[:-1]


def tqdm_defaults(mp: pytest.MonkeyPatch) -> None:
    """Unsets every TQDM_* variable: tqdm runs with the defaults of the Library mapping."""
    for name in [k for k in os.environ if k.startswith("TQDM_")]:
        mp.delenv(name)
