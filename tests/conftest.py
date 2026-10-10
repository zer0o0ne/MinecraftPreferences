"""Fixtures shared by every block: the repository root and a fake HTTP server."""

import json
import select
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


class Gone(Exception):
    """The client closed the connection while its request was held."""


@dataclass
class Request:
    method: str
    path: str
    headers: object
    body: object
    t: float
    sock: socket.socket
    gone: bool = False

    def hold(self, seconds: float) -> None:
        """Delay the reply by `seconds`; raise Gone as soon as the client disconnects."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            if select.select([self.sock], [], [], min(left, 0.02))[0]:
                try:
                    closed = not self.sock.recv(1, socket.MSG_PEEK)
                except OSError:
                    closed = True
                if closed:
                    self.gone = True
                    raise Gone
                time.sleep(min(left, 0.02))

    def cancelled(self) -> bool:
        """Whether the client closes the held request's connection within 2 s."""
        end = time.monotonic() + 2.0
        while not self.gone and time.monotonic() < end:
            time.sleep(0.01)
        return self.gone


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        self.answer(None)

    def do_POST(self) -> None:
        self.answer(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

    def answer(self, body) -> None:
        fake = self.server.fake
        req = Request(
            self.command,
            self.path,
            self.headers,
            body,
            time.monotonic(),
            self.connection,
        )
        with fake.lock:
            fake.requests.append(req)
            fake.active += 1
            fake.max_active = max(fake.max_active, fake.active)
        try:
            reply = fake.app(req)
        except Gone:
            reply = None
        finally:
            with fake.lock:
                fake.active -= 1
        if reply is None:
            self.close_connection = True
            return
        status, payload = reply
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server(ThreadingHTTPServer):
    daemon_threads, block_on_close, request_queue_size = True, False, 1024

    def handle_error(self, request, client_address) -> None:
        pass


class FakeHTTP:
    """HTTP/1.0 on 127.0.0.1; `app(request)` -> (status, JSON or bytes), or None to drop the connection."""

    def __init__(self, app):
        self.app, self.requests, self.active, self.max_active = app, [], 0, 0
        self.lock, self.thread = threading.Lock(), None
        self.server = Server(("127.0.0.1", 0), Handler, bind_and_activate=False)
        self.server.fake = self
        self.server.server_bind()
        self.port = self.server.server_address[1]

    def listen(self) -> None:
        """Accept connections; until then the bound port refuses them."""
        self.server.server_activate()
        self.thread = threading.Thread(
            target=self.server.serve_forever, args=(0.01,), daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        if self.thread:
            self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def http():
    """Factory of FakeHTTP servers, listening unless `listen=False`; closed after the test."""
    fakes = []

    def start(app, listen: bool = True) -> FakeHTTP:
        fakes.append(FakeHTTP(app))
        if listen:
            fakes[-1].listen()
        return fakes[-1]

    yield start
    for fake in fakes:
        fake.close()


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture
def refusing_proxy(http, monkeypatch) -> None:
    """Proxy variables name a port that refuses connections; NO_PROXY is unset."""
    port = http(lambda req: None, listen=False).port
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, f"http://127.0.0.1:{port}")
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
