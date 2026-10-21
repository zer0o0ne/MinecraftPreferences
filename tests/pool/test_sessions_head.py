"""(1) of specs/preference-pool.md: `head` on the fake Azure server of pool_site.py — one
ranged GET with `x-ms-version`; 206 with and without a newline, the whole file shorter
than B, the fallback to the ranged fetch (f) when the size of Content-Range exceeds B (its
None and b""), a 206 needing it without a usable Content-Range, 416, 404, 5xx and transport
errors retried, any other status aborting; line 0 `ok` or `malformed`."""

import json

import httpx
import pytest

from mcpref.clips.collect import load_config as load_perception
from mcpref.pool.sessions import head

from .pool_world import PoolEp, relpath

RP = relpath("7.1", "solo", "ab" * 6, 0)
OK = {"stats": {"s": 7}, "serverTick": 5, "milli": 9}  # 48 bytes as JSON


def line(value=None) -> bytes:
    return json.dumps(OK if value is None else value).encode()


class Asked:
    """`head` of RP served as `data`: the result and the requests received."""

    def __init__(self, make_site, data, b: int, faults=None, ranges=None, **p):
        self.site = site = make_site(episodes={"7xx": [PoolEp(RP, jsonl=data)]})
        self.jsonl, self.b = site.path(RP, ".jsonl"), b
        self.get = ("GET", self.jsonl, f"bytes=0-{b - 1}")
        site.faults |= {self.get: faults} if faults else {}
        site.content_ranges |= {self.get: ranges} if ranges != "keep" else {}
        site.config(p=p)
        cfg = load_perception(site.paths["perception-data"])
        self.url = site.url_stem(RP) + ".jsonl"
        self.client = httpx.Client(headers={"x-ms-version": cfg.x_ms_version})
        self.call = lambda: head(self.client, cfg, site.url_stem(RP), b)
        try:
            self.result, self.error = self.call(), None
        except Exception as error:  # noqa: BLE001 - examined by the test
            self.result, self.error = None, error

    @property
    def keys(self) -> list:
        return self.site.asked()

    def versions(self) -> set:
        return {r.headers.get("x-ms-version") for r in self.site.server.requests}


@pytest.fixture
def ask(make_site):
    """Asked factory; the clients closed after the test."""
    made = []

    def make(data, b=64, faults=None, ranges="keep", **p) -> Asked:
        made.append(Asked(make_site, data, b, faults, ranges, **p))
        return made[-1]

    yield make
    for asked in made:
        asked.client.close()


def test_206_with_a_newline_is_line_0(ask):
    """One GET with Range bytes=0-63 and x-ms-version; line 0 within it."""
    got = ask(line() + b"\n" + b"x" * 300 + b"\n")
    assert got.result == {"status": "ok"} | OK
    assert got.keys == [got.get] and got.versions() == {"2019-12-12"}


def test_file_shorter_than_b_is_line_0_without_a_newline(ask):
    """206 with the whole file (no newline, size <= B): line 0 is all of it."""
    data = line()
    got = ask(data, b=len(data) + 10)
    assert got.result == {"status": "ok"} | OK and got.keys == [got.get]


def padded(n: int) -> bytes:
    """Line 0 of exactly n bytes (JSON allows trailing spaces), no newline."""
    assert n >= len(line())
    return line() + b" " * (n - len(line()))


def test_size_exactly_b_needs_no_fallback(ask):
    got = ask(padded(80), b=80)
    assert got.result == {"status": "ok"} | OK and got.keys == [got.get]


@pytest.mark.parametrize("n", [65, 300])
def test_size_above_b_falls_back_to_the_whole_file(ask, n):
    """No newline in the 64 bytes and a size above 64: HEAD and the parts of (f) (three
    streams), each with x-ms-version; line 0 from the whole file."""
    data = padded(n) + b"\n" + line({"other": 1}) + b"\n"
    got = ask(data, b=64)
    assert got.result == {"status": "ok"} | OK
    p = -(-len(data) // 3)
    parts = [("GET", got.jsonl, f"bytes={a}-{min(a + p, len(data)) - 1}") for a in range(0, len(data), p)]  # fmt: skip
    assert got.keys[:2] == [got.get, ("HEAD", got.jsonl, None)]
    assert sorted(got.keys[2:]) == sorted(parts) and got.versions() == {"2019-12-12"}


def test_whole_file_without_a_newline_after_the_fallback(ask):
    got = ask(padded(200), b=64)
    assert got.result == {"status": "ok"} | OK and ("HEAD", got.jsonl, None) in got.keys


@pytest.mark.parametrize("reply, status", [(404, "missing"), ((200, b""), "empty")])
def test_fallback_none_is_missing_and_empty_is_empty(ask, reply, status):
    """The file gone (HEAD 404) or emptied (Content-Length 0) between the two requests."""
    got = ask(padded(300) + b"\n", b=64)
    assert got.result["status"] == "ok"
    got.site.faults[("HEAD", got.jsonl, None)] = [reply] * 3
    assert got.call() == {"status": status}


@pytest.mark.parametrize("value", [None, "bytes 0-63/*", "bytes */300", "items 0-63/300", "0-63/300"])  # fmt: skip
def test_206_needing_the_fallback_without_content_range_aborts(ask, value):
    got = ask(padded(300) + b"\n", b=64, ranges=value)
    assert type(got.error) is RuntimeError
    assert str(got.error) == f"GET {got.url}: HTTP 206 without Content-Range"
    assert got.keys == [got.get]


def test_206_with_a_newline_does_not_read_content_range(ask):
    """The size is needed only for a body without a newline."""
    got = ask(line() + b"\n" + b"x" * 300 + b"\n", ranges=None)
    assert got.result == {"status": "ok"} | OK


def test_416_is_empty(ask):
    got = ask(b"")
    assert got.result == {"status": "empty"} and got.keys == [got.get]


def test_404_is_missing(ask):
    got = ask(None)
    assert got.result == {"status": "missing"} and got.keys == [got.get]


@pytest.mark.parametrize("fault", [503, 500, "drop"])
def test_5xx_and_transport_errors_are_retried_after_retry_delay(ask, fault):
    got = ask(line() + b"\n", faults=[fault])
    assert got.result == {"status": "ok"} | OK and got.keys == [got.get] * 2
    first, second = got.site.server.requests
    assert second.t - first.t >= 0.05  # retry_delay


def test_retries_exhausted_abort(ask):
    """http_retries = 1: two attempts, then the run aborts."""
    got = ask(line() + b"\n", faults=[503, 503, 503])
    assert got.error is not None and got.keys == [got.get] * 2


@pytest.mark.parametrize("reply", [200, 403, 302, 412])
def test_any_other_status_aborts(ask, reply):
    """A 302 is not followed."""
    got = ask(line() + b"\n", faults=[(reply, line() + b"\n")])
    assert type(got.error) is RuntimeError
    assert str(got.error) == f"GET {got.url}: HTTP {reply}" and got.keys == [got.get]


MALFORMED = {
    "invalid JSON": b'{"stats": {}, "serverTick": 1, "milli": 2',
    "invalid UTF-8": b'{"stats": {}, "serverTick": 1, "milli": 2, "x": "\xff"}',
    "empty line 0": b"",
    "an array": b"[1, 2]",
    "a number": b"3",
    "no stats": line({"serverTick": 1, "milli": 2}),
    "no serverTick": line({"stats": {}, "milli": 2}),
    "no milli": line({"stats": {}, "serverTick": 1}),
    "stats a list": line({"stats": [], "serverTick": 1, "milli": 2}),
    "stats null": line({"stats": None, "serverTick": 1, "milli": 2}),
    "boolean serverTick": line({"stats": {}, "serverTick": True, "milli": 2}),
    "float serverTick": line({"stats": {}, "serverTick": 1.0, "milli": 2}),
    "boolean milli": line({"stats": {}, "serverTick": 1, "milli": False}),
    "float milli": line({"stats": {}, "serverTick": 1, "milli": 2.0}),
    "string milli": line({"stats": {}, "serverTick": 1, "milli": "2"}),
}


@pytest.mark.parametrize("first", MALFORMED.values(), ids=MALFORMED)
def test_malformed_line_0(ask, first):
    """The next line is a well-formed head: only line 0 counts."""
    got = ask(first + b"\n" + line() + b"\n", b=4096)
    assert got.result == {"status": "malformed"}


@pytest.mark.parametrize(
    "value",
    [{"stats": {}, "serverTick": 0, "milli": -3},
     {"x": [1], "stats": {"a": 1.5, "b": 2}, "serverTick": 2**40, "milli": 10**13, "é": "é"}],
)  # fmt: skip
def test_ok_heads(ask, value):
    """Empty stats (a session start), float stat values, other fields, any integers."""
    data = json.dumps(value, ensure_ascii=False).encode()
    got = ask(data + b"\n", b=4096)
    keep = {k: value[k] for k in ("stats", "serverTick", "milli")}
    assert got.result == {"status": "ok"} | keep
