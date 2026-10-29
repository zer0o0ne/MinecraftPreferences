"""(3) of specs/engine.md: the engine fetched by `retried` with the attempt of (3): a cut body
(a transport error) and a 5xx retried, Z truncated by each 200 attempt and untouched by a
5xx, the sha256 of the last attempt alone; retries exhausted (HTTPStatusError of the last
5xx, the last transport error re-raised); a 4xx, a final 3xx, a 2xx other than 200,
TooManyRedirects and DecodingError not retried ([H]); `http_timeout` per attempt."""

import httpx
import pytest
from progress_bars import assert_aborted

from .prepare_site import PREFIX, ZIP, cut, encoded, expected_entries, held, status


def last_line(site) -> str:
    r = site.run()
    assert_aborted(r.code, r.err)
    assert not site.jar.exists()
    return r.err.splitlines()[-1]


def test_cut_body_then_5xx_then_200(site):
    """http_retries = 2: Z holds the first attempt's 30 kB, untouched by the 503; the
    third attempt rewrites it whole and its sha256 matches."""
    site.cdn = [cut(30_000), status(503)]
    r = site.run()
    assert r.code == 0, r.err
    assert site.paths().count("/hf/engine.zip") == site.attempts() == 3
    assert site.seen == [None, ZIP[:30_000], ZIP[:30_000]]
    assert site.entries() == expected_entries()


def test_each_200_attempt_truncates(site):
    """A longer cut body, then a shorter one: Z holds the shorter one alone."""
    site.cdn = [cut(30_000), cut(10_000)]
    assert site.run().code == 0
    assert site.seen == [None, ZIP[:30_000], ZIP[:10_000]]
    assert site.entries() == expected_entries()


@pytest.mark.parametrize("code", [500, 502, 503, 504, 599])
def test_5xx_retried(site, code):
    site.cdn = [status(code)]
    assert site.run().code == 0
    assert site.attempts() == 2 and site.seen == [None, None]


@pytest.mark.parametrize("retries", [0, 2])
def test_5xx_retries_exhausted(site, retries):
    """1 + http_retries attempts, then raise_for_status() of the last 503."""
    site.fields["http_retries"] = retries
    site.cdn = [status(503)] * (retries + 1)
    error = f"HTTPStatusError: Server error '503 Service Unavailable' for url '{site.location}'"
    assert last_line(site).startswith(PREFIX + error)
    assert site.attempts() == retries + 1
    assert not (site.download / "engine.zip").exists()


def test_transport_retries_exhausted(site):
    """Three cut bodies: the last transport error raised."""
    site.cdn = [cut(1000)] * 3
    name = last_line(site)[len(PREFIX) :].split(":")[0]
    assert issubclass(getattr(httpx, name), httpx.TransportError), name
    assert site.attempts() == 3


@pytest.mark.parametrize(
    "code, error",
    [
        (404, "HTTPStatusError: Client error '404 Not Found' for url '{cdn}'"),
        (429, "HTTPStatusError: Client error '429 Too Many Requests' for url '{cdn}'"),
        (302, "HTTPStatusError: Redirect response '302 Found' for url '{cdn}'"),
        (304, "HTTPStatusError: Redirect response '304 Not Modified' for url '{cdn}'"),
        (204, "ValueError: {engine}: HTTP 204"),
        (206, "ValueError: {engine}: HTTP 206"),
    ],
)
def test_status_not_retried(site, code, error):
    """A 4xx; a final 3xx (no Location: not followed); a 2xx other than 200 with
    `engine_url` as configured. One attempt; Z untouched."""
    site.cdn = [status(code)]
    line = last_line(site)
    assert line.startswith(
        PREFIX + error.format(cdn=site.location, engine=site.engine_url)
    )
    assert site.attempts() == 1 and not (site.download / "engine.zip").exists()


def test_decoding_error_not_retried(site):
    site.cdn = [encoded]
    assert last_line(site).startswith(f"{PREFIX}DecodingError: ")
    assert site.attempts() == 1


def test_redirect_loop_not_retried(site):
    """TooManyRedirects after 20 redirects ([H]): one attempt."""
    site.location = site.engine_url
    assert last_line(site).startswith(f"{PREFIX}TooManyRedirects: ")
    assert site.paths().count("/hf/engine.zip") == 21


def test_timeout_per_attempt(site):
    """A reply held past `http_timeout`: the attempt abandoned and retried."""
    site.fields["http_timeout"] = 0.3
    site.cdn = [held(3.0)]
    r = site.run()
    assert r.code == 0, r.err
    first = next(q for q in site.server.requests if q.path == "/cdn/engine.zip")
    assert site.attempts() == 2 and first.cancelled()


@pytest.mark.parametrize("retries, code", [(2, 0), (1, 1), (0, 1)])
def test_lwjgl_download_retries(site, retries, code):
    """(2) by `download` with `http_retries`: a jar answering 503 twice."""
    site.fields["http_retries"] = retries
    site.busy[site.jar_paths()[3]] = [503, 503]
    r = site.run()
    assert r.code == code, r.err
    assert site.paths().count(site.jar_paths()[3]) == min(retries + 1, 3)
