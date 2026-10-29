"""Retries after a delay: specs/gui-screens.md (textures), specs/vlm-client.md (6),
specs/vpt-gui-frames.md Retries, (c); `download` of specs/perception.md; specs/engine.md
(2), (3)."""

import time
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import httpx

T = TypeVar("T")
Outcome = httpx.Response | httpx.TransportError
Method = Callable[..., httpx.Response]


def retried(
    call: Callable[[], T], again: Callable[[T], bool], retries: int, delay: float
) -> T:
    """The result of `call`, repeated after `delay` s at most `retries` times while `again`."""
    result = call()
    for _ in range(retries):
        if not again(result):
            break
        time.sleep(delay)
        result = call()
    return result


def attempt(method: Method, url: str, timeout: float) -> Outcome:
    """One request by `method` (a client's `get` or `head`, extra headers bound):
    `timeout` per phase, no redirects; returns a transport error."""
    try:
        return method(url, timeout=httpx.Timeout(timeout), follow_redirects=False)
    except httpx.TransportError as error:
        return error


def retryable(outcome: Outcome) -> bool:
    """Any `httpx.TransportError` or HTTP 5xx is retried."""
    return isinstance(outcome, httpx.TransportError) or outcome.is_server_error


def last_response(outcome: Outcome) -> httpx.Response:
    """The last outcome of `retried`: a transport error is raised, a response returned."""
    if isinstance(outcome, httpx.TransportError):
        raise outcome
    return outcome


def request(
    method: Method, url: str, timeout: float, retries: int, delay: float
) -> httpx.Response:
    """A request retried on a transport error or HTTP 5xx; a last transport error is raised."""
    call = partial(attempt, method, url, timeout)
    return last_response(retried(call, retryable, retries, delay))


def get(
    client: httpx.Client, url: str, timeout: float, retries: int, delay: float
) -> httpx.Response:
    """GET retried on a transport error or HTTP 5xx; a last transport error is raised."""
    return request(client.get, url, timeout, retries, delay)


def download(url: str, timeout: float, retries: int, delay: float) -> bytes:
    """The body of `url` by `get` on a new client, closed after; an HTTP error raises."""
    with httpx.Client() as client:
        response = get(client, url, timeout, retries, delay)
    return response.raise_for_status().content
