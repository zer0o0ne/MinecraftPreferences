"""Bounded, version-pinned downloads across a persistent HTTP connection pool."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import httpx
import pytest

from mcpref.datasets.continuous.acquisition import http as transfer


@pytest.fixture
def config():
    access = SimpleNamespace(
        x_ms_version="test-version", http_timeout=3, http_retries=1, retry_delay=0
    )
    settings = SimpleNamespace(
        connections=3,
        file_connections=2,
        block_bytes=4,
        read_bytes=2,
        block_timeout=5,
        keepalive_expiry=20,
    )
    return access, settings


def manifest(data, name="video"):
    return {"url": f"https://source.test/{name}", "etag": '"v1"', "bytes": len(data)}


def response(request, data, **changes):
    first, last = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
    headers = {
        "ETag": '"v1"',
        "Content-Range": f"bytes {first}-{last}/{len(data)}",
        "Content-Length": str(last - first + 1),
    } | changes
    return httpx.Response(
        206, headers=headers, stream=httpx.ByteStream(data[first : last + 1])
    )


def mock_client(monkeypatch, handler):
    clients, options = [], []
    constructor = httpx.Client

    def create(**kwargs):
        options.append(kwargs)
        client = constructor(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(transfer.httpx, "Client", create)
    return clients, options


def test_slow_first_block_does_not_block_queue_refill_and_client_is_reused(
    monkeypatch, tmp_path, config
):
    data = b"abcdefghijklmnopqrst"
    next_block = Event()
    ranges = []

    def handler(request):
        assert request.headers["x-ms-version"] == "test-version"
        if request.url.path == "/options":
            return httpx.Response(200, json={"ok": True})
        assert request.headers["If-Match"] == '"v1"'
        first = int(request.headers["Range"].split("=")[1].split("-")[0])
        ranges.append(first)
        if first == 0:
            assert next_block.wait(2), "next block waited for the slow first block"
        elif first == 8:
            next_block.set()
        return response(request, data)

    clients, options = mock_client(monkeypatch, handler)
    with transfer.Transport(*config) as transport:
        assert transport.options("https://source.test/options").json() == {"ok": True}
        assert transport.fetch(manifest(data), tmp_path / "first")
        assert transport.fetch(manifest(data), tmp_path / "second")
        assert (tmp_path / "first").read_bytes() == data
        assert (tmp_path / "second").read_bytes() == data
    assert sorted(ranges) == sorted([0, 4, 8, 12, 16] * 2)
    assert len(clients) == 1 and clients[0].is_closed
    assert options[0]["limits"] == httpx.Limits(
        max_connections=3, max_keepalive_connections=3, keepalive_expiry=20
    )
    assert not list(tmp_path.glob("*.part"))


def test_parallel_files_obey_global_and_per_file_bounds(monkeypatch, tmp_path, config):
    data = b"abcdefghijklmnop"
    all_started, release, lock = Event(), Event(), Lock()
    active, peak, by_file, file_peaks = 0, 0, {}, {}

    def handler(request):
        nonlocal active, peak
        name = request.url.path
        with lock:
            active += 1
            by_file[name] = by_file.get(name, 0) + 1
            peak = max(peak, active)
            file_peaks[name] = max(file_peaks.get(name, 0), by_file[name])
            if active == config[1].connections:
                all_started.set()
        try:
            assert release.wait(2)
            return response(request, data)
        finally:
            with lock:
                active -= 1
                by_file[name] -= 1

    mock_client(monkeypatch, handler)
    with transfer.Transport(*config) as transport, ThreadPoolExecutor(4) as threads:
        jobs = [
            threads.submit(transport.fetch, manifest(data, str(i)), tmp_path / str(i))
            for i in range(4)
        ]
        try:
            assert all_started.wait(2)
            assert peak == 3
        finally:
            release.set()
        assert all(job.result() for job in jobs)
    assert max(file_peaks.values()) <= 2
    assert peak == 3
    assert all((tmp_path / str(i)).read_bytes() == data for i in range(4))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"ETag": '"other"'}, "ETag"),
        ({"Content-Range": "bytes 1-4/4"}, "Content-Range"),
        ({"Content-Length": "3"}, "Content-Length"),
        ({"Content-Encoding": "gzip"}, "encoded"),
    ],
)
def test_invalid_headers_abort_without_publish(
    monkeypatch, tmp_path, config, changes, message
):
    calls = []

    def handler(request):
        calls.append(request)
        return response(request, b"abcd", **changes)

    mock_client(monkeypatch, handler)
    target = tmp_path / "file"
    target.write_bytes(b"previous-complete-file")
    with (
        transfer.Transport(*config) as transport,
        pytest.raises(RuntimeError, match=message),
    ):
        transport.fetch(manifest(b"abcd"), target)
    assert len(calls) == 1
    assert target.read_bytes() == b"previous-complete-file"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("body,message", [(b"abc", "few"), (b"abcde", "many")])
def test_actual_body_length_is_checked(monkeypatch, tmp_path, config, body, message):
    def handler(request):
        value = response(request, b"abcd")
        value.stream = httpx.ByteStream(body)
        return value

    mock_client(monkeypatch, handler)
    with (
        transfer.Transport(*config) as transport,
        pytest.raises(RuntimeError, match=message),
    ):
        transport.fetch(manifest(b"abcd"), tmp_path / "file")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", [200, 302, 412])
def test_non_range_or_changed_source_responses_are_not_retried(
    monkeypatch, tmp_path, config, status
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    mock_client(monkeypatch, handler)
    with transfer.Transport(*config) as transport, pytest.raises(RuntimeError):
        transport.fetch(manifest(b"abcd"), tmp_path / "file")
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_missing_source_returns_false_and_cleans_partial_file(
    monkeypatch, tmp_path, config
):
    mock_client(monkeypatch, lambda request: httpx.Response(404))
    with transfer.Transport(*config) as transport:
        assert not transport.fetch(manifest(b"abcdefghijklmnop"), tmp_path / "file")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [503, "transport"])
def test_retry_contract_for_ranges_and_options(monkeypatch, tmp_path, config, failure):
    calls = {}

    def handler(request):
        name = request.url.path
        calls[name] = calls.get(name, 0) + 1
        if calls[name] == 1:
            if failure == "transport":
                raise httpx.ReadError("connection lost")
            return httpx.Response(failure)
        if name == "/options":
            return httpx.Response(200, json={"ok": True})
        return response(request, b"abcd")

    mock_client(monkeypatch, handler)
    with transfer.Transport(*config) as transport:
        assert transport.options("https://source.test/options").json() == {"ok": True}
        assert transport.fetch(manifest(b"abcd"), tmp_path / "file")
    assert calls == {"/options": 2, "/video": 2}
    assert (tmp_path / "file").read_bytes() == b"abcd"


@pytest.mark.parametrize("failure", [503, "transport"])
def test_retries_are_finite_and_cleanup_on_exhaustion(
    monkeypatch, tmp_path, config, failure
):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "transport":
            raise httpx.ReadError("connection lost")
        return httpx.Response(failure)

    mock_client(monkeypatch, handler)
    error = httpx.ReadError if failure == "transport" else httpx.HTTPStatusError
    with transfer.Transport(*config) as transport, pytest.raises(error):
        transport.fetch(manifest(b"abcd"), tmp_path / "file")
    assert len(calls) == config[0].http_retries + 1
    assert not list(tmp_path.iterdir())


def test_options_http_failure_is_raised(monkeypatch, config):
    mock_client(monkeypatch, lambda request: httpx.Response(404))
    with transfer.Transport(*config) as transport, pytest.raises(httpx.HTTPStatusError):
        transport.options("https://source.test/options")


def test_trickle_deadline_interrupts_before_read_buffer_fills_and_retries(
    monkeypatch, tmp_path, config
):
    access, settings = config
    settings.block_timeout = 3
    settings.read_bytes = 100
    clock, chunks, attempts, closed = [0], [], [], []
    monkeypatch.setattr(transfer.time, "monotonic", lambda: clock[0])

    class Trickle(httpx.SyncByteStream):
        def __iter__(self):
            for byte in b"abcd":
                clock[0] += 2
                chunks.append(byte)
                yield bytes([byte])

        def close(self):
            closed.append(True)

    def handler(request):
        attempts.append(request)
        value = response(request, b"abcd")
        if len(attempts) == 1:
            value.stream = Trickle()
        return value

    mock_client(monkeypatch, handler)
    with transfer.Transport(access, settings) as transport:
        assert transport.fetch(manifest(b"abcd"), tmp_path / "file")
    assert len(attempts) == 2
    assert chunks == list(b"ab")
    assert closed == [True]
    assert attempts[0].extensions["timeout"]["read"] == 3
    assert (tmp_path / "file").read_bytes() == b"abcd"


def test_failed_download_keeps_fd_open_until_inflight_writer_stops(
    monkeypatch, tmp_path, config
):
    import os

    data = b"abcdefgh"
    wrote, failed, release, finished = Event(), Event(), Event(), Event()
    descriptors = []
    real_write, real_check = transfer.write, transfer.check_response

    def write(fd, chunk, offset):
        descriptors.append(fd)
        result = real_write(fd, chunk, offset)
        wrote.set()
        return result

    def check(*args):
        try:
            real_check(*args)
        except RuntimeError:
            failed.set()
            raise

    class Paused(httpx.SyncByteStream):
        def __iter__(self):
            yield b"ef"
            assert release.wait(2)
            yield b"gh"

    def handler(request):
        if request.headers["Range"] == "bytes=0-3":
            assert wrote.wait(2)
            return response(request, data, ETag='"different"')
        value = response(request, data)
        value.stream = Paused()
        return value

    mock_client(monkeypatch, handler)
    monkeypatch.setattr(transfer, "write", write)
    monkeypatch.setattr(transfer, "check_response", check)
    with transfer.Transport(*config) as transport, ThreadPoolExecutor(1) as threads:

        def fetch():
            try:
                return transport.fetch(manifest(data), tmp_path / "file")
            finally:
                finished.set()

        job = threads.submit(fetch)
        try:
            assert failed.wait(2)
            os.fstat(descriptors[0])
            assert not finished.wait(0.02)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="ETag"):
            job.result()
    assert not list(tmp_path.iterdir())


def test_empty_source_is_still_version_checked(monkeypatch, tmp_path, config):
    calls = []

    def handler(request):
        assert "Range" not in request.headers
        calls.append(request)
        return httpx.Response(
            200,
            headers={"ETag": '"v1"', "Content-Length": "0"},
            stream=httpx.ByteStream(b""),
        )

    mock_client(monkeypatch, handler)
    with transfer.Transport(*config) as transport:
        assert transport.fetch(manifest(b""), tmp_path / "empty")
    assert len(calls) == 1
    assert (tmp_path / "empty").read_bytes() == b""


def test_partial_transport_retry_overwrites_block_and_writes_small_chunks(
    monkeypatch, tmp_path, config
):
    calls, sizes = [], []
    real_write = transfer.write

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield b"WR"
            raise httpx.ReadError("body interrupted")

    def handler(request):
        calls.append(request)
        value = response(request, b"abcd")
        if len(calls) == 1:
            value.stream = Interrupted()
        return value

    def write(fd, chunk, offset):
        sizes.append(len(chunk))
        return real_write(fd, chunk, offset)

    mock_client(monkeypatch, handler)
    monkeypatch.setattr(transfer, "write", write)
    with transfer.Transport(*config) as transport:
        assert transport.fetch(manifest(b"abcd"), tmp_path / "file")
    assert len(calls) == 2
    assert max(sizes) <= config[1].read_bytes
    assert (tmp_path / "file").read_bytes() == b"abcd"
