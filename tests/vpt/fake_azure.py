"""Fake Azure Blob server of [R2] in specs/vpt-gui-frames.md for tests/vpt: indices with
ETags; every file answers HEAD with its Content-Length and a Range GET with 206, exactly the
bytes asked (cut at the file's end) and Content-Range; 404 if absent."""

import io
import re
import threading
import time

from vpt_checks import part_ranges

HOLD = 3.0  # s a held reply waits; longer than every http_timeout of these tests


def request_key(method: str, path: str, headers) -> tuple:
    """A request as faults and stalls name it: (method, path, Range or None)."""
    return method, path, headers.get("Range")


class FakeAzure:
    """Serves `self.data(path)` (bytes, None: 404) and `self.etags` (subset -> ETag of
    /idx/<subset>.json, None: no ETag) of the subclass; `part` reads `self.fields`.
    `faults`: request key -> its first replies, each a status, (status, body), "drop" (no
    reply), "hold" (until the client gives up), "bare" (no Content-Length) or "cut" (the
    body one byte short of its Content-Length); `stalls`: request key -> s waited before
    the reply and again before its body; `redirects`: path -> Location of a 3xx reply."""

    def __init__(self, http):
        self.server = http(self.app, listen=False)
        self.host = f"http://127.0.0.1:{self.server.port}"
        self.faults, self.stalls, self.redirects = {}, {}, {}
        self.note = threading.local()  # Content-Length change of the reply being sent
        handler = self.server.server.RequestHandlerClass
        self.server.server.RequestHandlerClass = self.handler(handler)

    def handler(self, base):
        """`base` answering HEAD too, with the reply headers of `headers`."""
        azure = self

        class Handler(base):
            def do_HEAD(self) -> None:
                self.answer(None)

            def send_response(self, code, message=None) -> None:
                self.status = code
                super().send_response(code, message)

            def send_header(self, keyword, value) -> None:
                if keyword == "Content-Length":
                    change = getattr(azure.note, "change", 0)
                    if change is None:
                        return
                    value = self.length = int(value) + change
                super().send_header(keyword, value)

            def end_headers(self) -> None:
                for keyword, value in azure.headers(self).items():
                    self.send_header(keyword, value)
                super().end_headers()
                key = request_key(self.command, self.path, self.headers)
                time.sleep(azure.stalls.get(key, 0))
                if self.command == "HEAD":  # a HEAD reply has no body
                    self.wfile = io.BytesIO()

        return Handler

    def span(self, path: str, headers) -> tuple[int, int] | None:
        """Byte range asked for: inclusive (first, last), last cut at the file's end."""
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", headers.get("Range") or "")
        if not match or self.data(path) is None:
            return None
        last = len(self.data(path)) - 1
        return int(match[1]), min(int(match[2] or last), last)

    def fault(self, req):
        key = request_key(req.method, req.path, req.headers)
        n = self.asked().count(key) - 1
        faults = self.faults.get(key, [])
        return faults[n] if n < len(faults) else None

    def app(self, req):
        self.note.change = 0
        fault = self.fault(req)
        req.hold(self.stalls.get(request_key(req.method, req.path, req.headers), 0))
        if fault == "hold":
            req.hold(HOLD)
        elif fault == "drop":
            return None
        elif isinstance(fault, int):
            return fault, b""
        elif isinstance(fault, tuple):
            return fault
        data, span = self.data(req.path), self.span(req.path, req.headers)
        if data is None:
            return 404, b""
        status, body = (206, data[span[0] : span[1] + 1]) if span else (200, data)
        self.note.change = {"bare": None, "cut": 1}.get(fault, 0)
        return status, body[:-1] if fault == "cut" else body

    def headers(self, handler) -> dict:
        path, data = handler.path, self.data(handler.path)
        subset = re.fullmatch(r"/idx/(.+)\.json", path)
        out = {}
        if 300 <= handler.status < 400 and path in self.redirects:
            out["Location"] = self.redirects[path]
        if subset and self.etags.get(subset[1]) is not None:
            out["ETag"] = self.etags[subset[1]]
        if data is not None:
            out["Accept-Ranges"] = "bytes"
        span = self.span(path, handler.headers)
        if handler.status == 206 and span and hasattr(handler, "length"):
            end = span[0] + handler.length - 1
            out["Content-Range"] = f"bytes {span[0]}-{end}/{len(data)}"
        return out

    def get(self, path: str) -> tuple:
        """Request key of a GET of the whole file."""
        return "GET", path, None

    def head(self, path: str) -> tuple:
        return "HEAD", path, None

    def part(self, path: str, j: int, q: int | None = None) -> tuple:
        """Request key of part j of (f) with fetch_streams q (default: the fields')."""
        q = q or self.fields["fetch_streams"]
        return "GET", path, part_ranges(len(self.data(path)), q)[j]

    def asked(self, path: str | None = None) -> list[tuple]:
        """Request keys received (for `path` only, if given), in arrival order."""
        keys = [request_key(r.method, r.path, r.headers) for r in self.server.requests]
        return [k for k in keys if path in (None, k[1])]
