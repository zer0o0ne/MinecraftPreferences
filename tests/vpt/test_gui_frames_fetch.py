"""(f) of specs/vpt-gui-frames.md: `fetch` against a fake Azure server serving HEAD and Range
as [R2] (exact Range values, parts in flight at once, L = 0, a missing file), and the aborts
of (f) through `main run` (the other parts finish; the message of the lowest failing part)."""

import random
import threading
import time

import httpx
import pytest
from progress_bars import assert_aborted
from vpt_checks import part_ranges

from mcpref.vpt.gui_frames import GuiFramesConfig, fetch, main

PATH = "/b1/data/7.1/f.jsonl"
RETRIED = 3  # FIELDS http_retries = 2: three requests


def fetch_file(site, data: bytes | None, q: int) -> bytes | None:
    """`fetch` of PATH holding `data` (None: absent) with fetch_streams = q, on a client
    as the Interface and Library mapping give it: x-ms-version, timeout, no redirects."""
    if data is not None:
        site.world.files[site.host + PATH] = data
    cfg = GuiFramesConfig(**(site.fields | {"fetch_streams": q}))
    with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}, follow_redirects=False,
                      timeout=httpx.Timeout(cfg.http_timeout)) as client:  # fmt: skip
        return fetch(client, cfg, site.host + PATH)


def body(size: int) -> bytes:
    return random.Random(size).randbytes(size)


def assert_head_then(site, ranges: list[str]) -> None:
    """One HEAD, then one GET per Range value, every request with x-ms-version."""
    asked = site.asked()
    assert asked[0] == site.head(PATH)
    assert sorted(asked[1:]) == sorted(("GET", PATH, f"bytes={r}") for r in ranges)
    assert {r.headers["x-ms-version"] for r in site.server.requests} == {"2019-12-12"}


@pytest.mark.parametrize(
    "size, q, ranges",
    [(3, 5, ["0-0", "1-1", "2-2"]),  # L < q: p = 1, L parts
     (12, 4, ["0-2", "3-5", "6-8", "9-11"]),  # L a multiple of q: p = 3, q parts
     (5, 4, ["0-1", "2-3", "4-4"]),  # L >= q: p = 2, 3 < q parts
     (10, 8, ["0-1", "2-3", "4-5", "6-7", "8-9"]),  # p = 2, 5 < q parts
     (13, 4, ["0-3", "4-7", "8-11", "12-12"]),  # p = 4, the last part shorter
     (7, 1, ["0-6"]),  # q = 1: the whole file in one part
     (1, 20, ["0-0"])],
)  # fmt: skip
def test_parts_are_exact_ranges_after_one_head(site, size, q, ranges):
    data = body(size)
    assert fetch_file(site, data, q) == data
    assert_head_then(site, ranges)


@pytest.mark.parametrize("q", [1, 2, 3, 7, 19, 20])
def test_parts_of_many_sizes_rebuild_the_file(site, q):
    """Against the transcription of (f); ceil(L / p) <= q parts."""
    for size in (1, 2, 3, 19, 20, 21, 39, 40, 41, 400, 401):
        site.server.requests.clear()
        data = body(size)
        assert fetch_file(site, data, q) == data
        ranges = part_ranges(size, q)
        assert len(ranges) <= q
        assert_head_then(site, [r.removeprefix("bytes=") for r in ranges])


@pytest.mark.parametrize("size, q", [(100, 8), (20, 20), (5, 4)])
def test_all_parts_in_flight_at_once(site, size, q):
    """Each part is held until every part has arrived (at most 2 s)."""
    k, arrived, ready, app = (
        len(part_ranges(size, q)),
        [],
        threading.Condition(),
        site.app,
    )

    def gathered(req):
        if req.headers.get("Range"):
            with ready:
                arrived.append(req)
                ready.notify_all()
                ready.wait_for(lambda: len(arrived) >= k, timeout=2.0)
        return app(req)

    site.server.app = gathered
    data = body(size)
    assert fetch_file(site, data, q) == data
    assert site.server.max_active == k


def test_empty_file_is_its_head_alone(site):
    assert fetch_file(site, b"", 4) == b""
    assert site.asked() == [site.head(PATH)]


def test_missing_file_is_none(site):
    assert fetch_file(site, None, 4) is None
    assert site.asked() == [site.head(PATH)]


def aborted_run(site, http, capfd, faults: dict, stalls: dict | None = None) -> str:
    """`main run` with `faults` and `stalls` on a cleared request log; 3xx replies point at
    another server, which must see no request. Returns the abort message."""
    elsewhere = http(lambda req: (200, b""))
    site.server.requests.clear()
    site.faults, site.stalls = faults, stalls or {}
    site.redirects = {
        key[1]: f"http://127.0.0.1:{elsewhere.port}/moved" for key in faults
    }
    capfd.readouterr()
    code = main(["run", site.config()])
    err = capfd.readouterr().err
    assert_aborted(code, err)
    assert elsewhere.requests == [] and site.calls() == [] and site.files() == {}
    return err[:-1].rsplit("\n", 1)[-1]


HEAD_REPLIES = {"403": 403, "302": 302, "206": "206", "no Content-Length": "bare"}


@pytest.mark.parametrize("reply", HEAD_REPLIES.values(), ids=HEAD_REPLIES)
def test_head_other_than_200_with_length_aborts(site, http, capfd, reply):
    """Not retried (no 5xx, no transport error); no part is asked."""
    path = site.first_fetched(".jsonl")
    reply = (206, site.data(path)) if reply == "206" else reply
    aborted_run(site, http, capfd, {site.head(path): [reply]})
    assert site.asked(path) == [site.head(path)]


PART_REPLIES = ["200", 404, 416, 302, "short", "long"]


@pytest.mark.parametrize("reply", PART_REPLIES)
def test_part_other_than_206_of_its_length_aborts(site, http, capfd, reply):
    """Part 1 fails once and is not retried; every other part is asked once."""
    path = site.first_fetched(".jsonl")
    data, key = site.data(path), site.part(path, 1)
    first, last = map(int, key[2].removeprefix("bytes=").split("-"))
    part = data[first : last + 1]
    reply = {"200": (200, data), "short": (206, part[:-1]),
             "long": (206, part + b"\0")}.get(reply, reply)  # fmt: skip
    aborted_run(site, http, capfd, {key: [reply]})
    ranges = part_ranges(len(data), site.fields["fetch_streams"])
    assert sorted(site.asked(path)) == sorted(
        [site.head(path)] + [("GET", path, r) for r in ranges]
    )


def test_abort_waits_for_the_other_parts(site, http, capfd):
    """Part 2 fails at once, part 0 answers after 0.5 s: it is received, not cut off."""
    path = site.first_fetched(".jsonl")
    fail, slow = site.part(path, 2), site.part(path, 0)
    alone = aborted_run(site, http, capfd, {fail: [404]})
    assert aborted_run(site, http, capfd, {fail: [404]}, {slow: 0.5}) == alone
    held = next(r for r, k in zip(site.server.requests, site.asked()) if k == slow)
    assert not held.gone and time.monotonic() >= held.t + 0.5


def test_abort_reports_the_lowest_failing_part(site, http, capfd):
    """Part 3 fails at once, part 1 after 0.5 s: the message is part 1's."""
    path = site.first_fetched(".jsonl")
    low, high = site.part(path, 1), site.part(path, 3)
    low_alone = aborted_run(site, http, capfd, {low: [416]})
    assert aborted_run(site, http, capfd, {high: [404]}) != low_alone
    assert (
        aborted_run(site, http, capfd, {low: [416], high: [404]}, {low: 0.5})
        == low_alone
    )


def test_abort_waits_for_the_retries_of_a_lower_part(site, http, capfd):
    """Part 2 fails at once; part 0 answers 503 until its retries are exhausted."""
    path = site.first_fetched(".jsonl")
    low, high = site.part(path, 0), site.part(path, 2)
    low_alone = aborted_run(site, http, capfd, {low: [503] * 10})
    assert aborted_run(site, http, capfd, {high: [404]}) != low_alone
    assert aborted_run(site, http, capfd, {low: [503] * 10, high: [404]}) == low_alone
    assert site.asked(path).count(low) == RETRIED
