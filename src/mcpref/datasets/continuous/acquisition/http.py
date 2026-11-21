"""Persistent pooled HTTP with bounded blocks and pinned source versions."""

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from threading import Event

import httpx

from mcpref.infrastructure.http.retry import last_response, request, retried, retryable
from mcpref.infrastructure.shards.store import publish


def check_response(response, expected, first, last):
    if response.status_code == 412:
        raise RuntimeError(f"Source ETag changed: {expected['url']}")
    status = 206 if expected["bytes"] else 200
    if response.status_code != status:
        raise RuntimeError(f"GET {expected['url']}: HTTP {response.status_code}")
    if response.headers.get("ETag") != expected["etag"]:
        raise RuntimeError(f"Source ETag differs: {expected['url']}")
    if expected["bytes"] and response.headers.get("Content-Range") != (
        f"bytes {first}-{last}/{expected['bytes']}"
    ):
        raise RuntimeError("Source returned an incorrect Content-Range")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != last - first + 1:
        raise RuntimeError("Source returned an incorrect Content-Length")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise RuntimeError("Source returned an encoded byte range")


def write(fd, chunk, offset):
    view = memoryview(chunk)
    while view:
        written = os.pwrite(fd, view, offset)
        if not written:
            raise OSError("Range write made no progress")
        view, offset = view[written:], offset + written
    return offset


class Transport:
    def __init__(self, access, settings):
        self.access, self.settings = access, settings
        self.client = httpx.Client(
            headers={
                "x-ms-version": access.x_ms_version,
                "Accept-Encoding": "identity",
            },
            limits=httpx.Limits(
                max_connections=settings.connections,
                max_keepalive_connections=settings.connections,
                keepalive_expiry=settings.keepalive_expiry,
            ),
            timeout=httpx.Timeout(access.http_timeout),
            follow_redirects=False,
        )
        self.threads = ThreadPoolExecutor(settings.connections)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.threads.shutdown(wait=True, cancel_futures=True)
        finally:
            self.client.close()

    def options(self, url: str) -> httpx.Response:
        """GET options with the legacy transport/5xx retries; raise for HTTP errors."""
        cfg = self.access
        return request(
            self.client.get, url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay
        ).raise_for_status()

    def _check_deadline(self, deadline, cancelled):
        if cancelled.is_set():
            raise RuntimeError("Source download cancelled")
        if time.monotonic() >= deadline:
            raise httpx.ReadTimeout("Source block attempt exceeded block_timeout")

    def _body(self, response, fd, first, last, deadline, cancelled):
        offset = first
        # iter_raw(chunk_size=...) buffers trickles before yielding and hides deadlines.
        for raw in response.iter_raw():
            self._check_deadline(deadline, cancelled)
            if offset + len(raw) > last + 1:
                raise RuntimeError("Source returned too many bytes")
            for start in range(0, len(raw), self.settings.read_bytes):
                self._check_deadline(deadline, cancelled)
                offset = write(
                    fd, raw[start : start + self.settings.read_bytes], offset
                )
        self._check_deadline(deadline, cancelled)
        if offset != last + 1:
            raise RuntimeError("Source returned too few bytes")

    def _attempt(self, expected, fd, first, last, cancelled):
        deadline = time.monotonic() + self.settings.block_timeout
        headers = {"If-Match": expected["etag"]}
        if expected["bytes"]:
            headers["Range"] = f"bytes={first}-{last}"
        timeout = min(self.access.http_timeout, self.settings.block_timeout)
        try:
            self._check_deadline(deadline, cancelled)
            with self.client.stream(
                "GET", expected["url"], headers=headers, timeout=httpx.Timeout(timeout)
            ) as response:
                self._check_deadline(deadline, cancelled)
                if response.status_code == 404 or response.is_server_error:
                    return response
                check_response(response, expected, first, last)
                self._body(response, fd, first, last, deadline, cancelled)
                return response
        except httpx.TransportError as error:
            return error

    def _block(self, expected, fd, first, last, cancelled):
        cfg = self.access
        call = partial(self._attempt, expected, fd, first, last, cancelled)
        outcome = retried(call, retryable, cfg.http_retries, cfg.retry_delay)
        response = last_response(outcome)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def _blocks(self, expected, fd):
        length = expected["bytes"]
        starts = iter(range(0, length, self.settings.block_bytes) if length else (0,))
        pending, cancelled = set(), Event()
        try:
            while True:
                while len(pending) < self.settings.file_connections:
                    first = next(starts, None)
                    if first is None:
                        break
                    last = min(first + self.settings.block_bytes, length) - 1
                    pending.add(
                        self.threads.submit(
                            self._block, expected, fd, first, last, cancelled
                        )
                    )
                if not pending:
                    return True
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for job in done:
                    if not job.result():
                        return False
                pending.difference_update(done)
        finally:
            cancelled.set()
            for job in pending:
                job.cancel()
            wait(pending)

    def fetch(self, expected: dict, target: Path) -> bool:
        """Publish all verified ranges atomically; False for a source returning 404."""
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".part")
        try:
            with staged.open("wb") as output:
                if not self._blocks(expected, output.fileno()):
                    return False
            publish(staged, target)
            return True
        finally:
            staged.unlink(missing_ok=True)
