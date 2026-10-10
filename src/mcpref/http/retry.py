"""Retries after a delay: specs/gui-screens.md (textures), specs/vlm-client.md (6),
specs/vpt-gui-frames.md (a)-(c)."""

import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")
Outcome = httpx.Response | httpx.TransportError


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


def attempt(client: httpx.Client, url: str, timeout: float) -> Outcome:
    """One GET of `url`: `timeout` per phase, no redirects; returns a transport error."""
    try:
        return client.get(url, timeout=httpx.Timeout(timeout), follow_redirects=False)
    except httpx.TransportError as error:
        return error


def retryable(outcome: Outcome) -> bool:
    """Any `httpx.TransportError` or HTTP 5xx is retried."""
    return isinstance(outcome, httpx.TransportError) or outcome.is_server_error


def get(
    client: httpx.Client, url: str, timeout: float, retries: int, delay: float
) -> httpx.Response:
    """GET retried on a transport error or HTTP 5xx; a last transport error is raised."""
    outcome = retried(lambda: attempt(client, url, timeout), retryable, retries, delay)
    if isinstance(outcome, httpx.TransportError):
        raise outcome
    return outcome
