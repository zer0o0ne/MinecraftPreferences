"""(0) of specs/perception-data.md: each index downloaded once with one GET, checked and
stored byte-identical under indices/, in config order; retries of 5xx and transport
errors; any other non-2xx and a bad body abort the run; a rerun reads the stored indices."""

import itertools

import httpx
import pytest
from vpt_world import SUBSETS

from .collect_out import assert_outputs


def idx(subset: str) -> str:
    return f"/idx/{subset}.json"


def stored(site) -> dict:
    return site.raw("indices")


def index_requests(site, since: int = 0) -> list:
    return [r for r in site.server.requests[since:] if r.path.startswith("/idx/")]


def test_indices_are_downloaded_once_in_config_order_and_stored_as_served(site):
    site.run()
    assert [(r.method, r.path) for r in index_requests(site)] == [("GET", idx(s)) for s in SUBSETS]  # fmt: skip
    assert all(r.headers["x-ms-version"] == "2019-12-12" for r in index_requests(site))
    assert all(r.headers.get("Range") is None for r in index_requests(site))
    assert stored(site) == {f"indices/{s}.json": site.index_body(s) for s in SUBSETS}


def test_rerun_reads_the_stored_indices_without_a_request(site):
    """The served indices and ETags change after the first run: the continuation uses the
    stored ones and ends as the uninterrupted run."""
    site.aborted(site.failing_gamma())
    since = len(site.server.requests)
    site.indices = {
        s: b'{"basedir": "http://elsewhere/", "relpaths": []}' for s in SUBSETS
    }
    site.etags = dict.fromkeys(SUBSETS, "0x8DA5AB90A1B2FFF")
    site.run()
    assert index_requests(site, since) == []
    site.indices = {}
    assert_outputs(site, site.expected())


def test_only_the_missing_indices_are_downloaded(site):
    """A 404 for 8xx aborts after 6xx and 7xx are stored; 9xx, 10xx are not asked; the
    rerun downloads 8xx, 9xx, 10xx only."""
    site.indices["8xx"] = None
    site.aborted()
    assert isinstance(site.error, httpx.HTTPStatusError)
    assert set(stored(site)) == {"indices/6xx.json", "indices/7xx.json"}
    assert [r.path for r in index_requests(site)] == [idx(s) for s in SUBSETS[:3]]
    del site.indices["8xx"]
    since = len(site.server.requests)
    site.run()
    assert [r.path for r in index_requests(site, since)] == [
        idx(s) for s in SUBSETS[2:]
    ]
    assert_outputs(site, site.expected())


@pytest.mark.parametrize("status", [404, 403, 302, 304, 400])
def test_a_non_2xx_index_aborts_without_retry(site, status):
    """Not 5xx, no transport error: one GET, no redirect followed; run.json written first."""
    site.faults[site.get(idx("6xx"))] = [status] * 3
    site.redirects[idx("6xx")] = site.host + idx("7xx")
    site.aborted()
    assert isinstance(site.error, httpx.HTTPStatusError)
    assert [r.path for r in index_requests(site)] == [idx("6xx")]
    assert stored(site) == {} and (site.out / "indices").is_dir()
    assert site.run_json() == site.expected_run()
    assert not (site.out / "shards").exists()


@pytest.mark.parametrize("retries", [0, 2])
def test_5xx_after_the_retries_aborts(site, retries):
    site.faults[site.get(idx("7xx"))] = [503, 500, 502]
    site.aborted(http_retries=retries)
    assert isinstance(site.error, httpx.HTTPStatusError)
    assert [r.path for r in index_requests(site)] == [idx("6xx")] + [idx("7xx")] * (retries + 1)  # fmt: skip
    assert set(stored(site)) == {"indices/6xx.json"}


def test_transport_errors_after_the_retries_abort(site):
    site.faults[site.get(idx("6xx"))] = ["drop"] * 3
    site.aborted(http_retries=2)
    assert len(index_requests(site)) == 3 and stored(site) == {}


def test_5xx_and_transport_errors_are_retried_after_retry_delay(site):
    site.faults[site.get(idx("9xx"))] = [503, "drop"]
    site.run(http_retries=2, retry_delay=0.3)
    tries = [r.t for r in index_requests(site) if r.path == idx("9xx")]
    assert len(tries) == 3 and all(b - a >= 0.3 for a, b in itertools.pairwise(tries))
    assert stored(site)["indices/9xx.json"] == site.index_body("9xx")


BODIES = [b"[]", b'"x"', b"null", b'{"basedir": "x", "relpaths": [', b"",
          b'{"relpaths": []}', b'{"basedir": "x"}', b'{"basedir": 1, "relpaths": []}',
          b'{"basedir": null, "relpaths": []}', b'{"basedir": "x", "relpaths": {}}',
          b'{"basedir": "x", "relpaths": "a.mp4"}']  # fmt: skip


@pytest.mark.parametrize("body", BODIES)
def test_a_bad_index_body_aborts_before_it_is_stored(site, body):
    site.indices["8xx"] = body
    site.aborted()
    assert type(site.error) is not httpx.HTTPStatusError
    assert isinstance(site.error, ValueError)
    assert set(stored(site)) == {"indices/6xx.json", "indices/7xx.json"}
    assert [r.path for r in index_requests(site)] == [idx(s) for s in SUBSETS[:3]]


def test_an_empty_relpaths_list_is_an_empty_pool(site):
    body = b'{"basedir": "http://elsewhere/", "relpaths": []}'
    site.indices["6xx"] = body
    site.run()
    assert stored(site)["indices/6xx.json"] == body
    assert all(r["subset"] != "6xx" for r in site.records())
    assert len(site.records()) == 11
