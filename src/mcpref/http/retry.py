"""Retries after a delay: specs/gui-screens.md (textures), specs/vlm-client.md (6),
specs/vpt-gui-frames.md Retries, (c)."""

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


def request(
    method: Method, url: str, timeout: float, retries: int, delay: float
) -> httpx.Response:
    """A request retried on a transport error or HTTP 5xx; a last transport error is raised."""
    outcome = retried(partial(attempt, method, url, timeout), retryable, retries, delay)
    if isinstance(outcome, httpx.TransportError):
        raise outcome
    return outcome


def get(
    client: httpx.Client, url: str, timeout: float, retries: int, delay: float
) -> httpx.Response:
    """GET retried on a transport error or HTTP 5xx; a last transport error is raised."""
    return request(client.get, url, timeout, retries, delay)
