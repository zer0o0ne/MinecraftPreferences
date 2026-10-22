"""`download` of src/mcpref/http/retry.py (specs/perception.md Interface; the retries of
`get` as specs/gui-screens.md states them) against a fake HTTP server: a body, a retried
5xx and a retried transport error, the retries exhausted, a 404 and a redirect raising at
once, the timeout per phase."""

import httpx
import pytest

from mcpref.http.retry import download

BODY = bytes(range(256)) * 3


def url(server, path: str = "/files/a.bin") -> str:
    return f"http://127.0.0.1:{server.port}{path}"


def replies(*answers):
    """An app giving `answers` in order (None drops the connection), then 200 BODY."""
    queue = list(answers)

    def app(req):
        return queue.pop(0) if queue else (200, BODY)

    return app


def test_body(http):
    server = http(replies())
    assert download(url(server), 5, 2, 0.01) == BODY
    assert [(r.method, r.path) for r in server.requests] == [("GET", "/files/a.bin")]


def test_retried_5xx(http):
    """A 503: retried after `delay`; the body of the retry."""
    server = http(replies((503, b"busy")))
    assert download(url(server), 5, 2, 0.3) == BODY
    first, second = server.requests
    assert second.t - first.t >= 0.3


def test_retried_transport_error(http):
    """A dropped connection (a transport error): retried."""
    server = http(replies(None))
    assert download(url(server), 5, 1, 0.01) == BODY
    assert len(server.requests) == 2


def test_retries_exhausted(http):
    """`retries` = 2: three requests answering 500, then an exception."""
    server = http(lambda req: (500, b"down"))
    with pytest.raises(Exception):  # noqa: B017  (the spec fixes no type)
        download(url(server), 5, 2, 0.01)
    assert len(server.requests) == 3


def test_no_retry(http):
    """`retries` = 0: one request."""
    server = http(lambda req: (502, b"down"))
    with pytest.raises(Exception):  # noqa: B017  (the spec fixes no type)
        download(url(server), 5, 0, 0.01)
    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [404, 403, 302])
def test_other_status_raises_at_once(http, status):
    """`raise_for_status()`: a 4xx or a redirect (not followed) raises HTTPStatusError,
    without a retry."""
    server = http(lambda req: (status, b"no"))
    with pytest.raises(httpx.HTTPStatusError):
        download(url(server), 5, 3, 0.01)
    assert len(server.requests) == 1


def test_timeout_per_phase(http):
    """A reply held longer than `timeout`: a timeout, retried."""
    state = {"n": 0}

    def app(req):
        state["n"] += 1
        if state["n"] == 1:
            req.hold(3.0)
        return (200, BODY)

    server = http(app)
    assert download(url(server), 0.3, 1, 0.01) == BODY
    assert len(server.requests) == 2
    assert server.requests[0].cancelled()
