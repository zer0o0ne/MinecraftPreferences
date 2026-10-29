"""A fake Maven and engine host on the shared `http` fixture, the config of specs/engine.md
pointing at it with the synthetic jars of jars.py, and runs of `main` of
`mcpref.engine.prepare`. Replies are written raw to the request's socket, the connection
closed after each, so that a reply can redirect, cut its body or carry any header (the
shared server answers only a status and a JSON or bytes body)."""

import hashlib
import io
import random
import zipfile
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

from mcpref.engine.prepare import main

from .engine_world import JAR, TINY, write_config
from .jars import MODULES, expected, lwjgl_jars, source_jar

PREFIX = "mcpref.engine.prepare: "
VERSION = "3.9.1"
MEMBER = "engine/build/libs/mcprec-6.13.jar"
SOURCE = source_jar()
JARS = lwjgl_jars()  # Notation: for each module, classes then natives
ENTRIES = expected(SOURCE, JARS)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def engine_zip(member: bytes) -> bytes:
    """engine.zip of [E]: `engine/.vscode/settings.json` (40 kB of noise) and S."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("engine/.vscode/settings.json", random.Random(5).randbytes(40_000))
        z.writestr(MEMBER, member)
    return buf.getvalue()


ZIP = engine_zip(SOURCE)


def send(req, status: int, body: bytes = b"", length=None, headers=None) -> None:
    """A raw HTTP/1.1 reply; a `length` above len(body) cuts the body."""
    phrase = HTTPStatus(status).phrase if status in set(HTTPStatus) else "Unassigned"
    lines = [f"HTTP/1.1 {status} {phrase}", "Connection: close",
             f"Content-Length: {len(body) if length is None else length}"]  # fmt: skip
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    req.sock.sendall("\r\n".join(lines).encode() + b"\r\n\r\n" + body)


def ok(site, req) -> None:
    send(req, 200, site.zip)


def status(code: int):
    body = b"" if code in (204, 304) else b"body of the reply"
    return lambda site, req: send(req, code, body)


def cut(n: int):
    """200 with the whole length announced, the connection dropped after n bytes."""
    return lambda site, req: send(req, 200, site.zip[:n], length=len(site.zip))


def held(seconds: float):
    """The reply held `seconds` (abandoned when the client disconnects), then 200."""

    def reply(site, req) -> None:
        req.hold(seconds)
        ok(site, req)

    return reply


def encoded(site, req) -> None:
    """200 announcing gzip whose body is not gzip: a DecodingError ([H])."""
    send(req, 200, b"not gzip at all", headers={"Content-Encoding": "gzip"})


@dataclass
class Run:
    code: int
    out: str
    err: str


class Site:
    """tmp/repo — the repository root; ms — `minestudio_dir`; tmp/engine.yaml — the
    config. /hf/engine.zip redirects to `location` (the second path /cdn/engine.zip);
    /cdn/engine.zip answers with the replies of `cdn` in order, then 200 with `zip`, and
    records Z's bytes as each of its requests arrives (`seen`, None when absent). A jar's
    path answers the statuses of `busy[path]` first."""

    def __init__(self, http, root: Path, tmp: Path, capsys):
        self.capsys, self.path, self.ms = capsys, tmp / "engine.yaml", root / "ms"
        self.jar, self.download = self.ms / JAR, self.ms / "download"
        self.zip, self.cdn, self.seen, self.busy = ZIP, [], [], {}
        self.server = http(self.app)
        base = f"http://127.0.0.1:{self.server.port}"
        self.engine_url = f"{base}/hf/engine.zip"
        self.location = f"{base}/cdn/engine.zip"
        lwjgl = f"{base}/maven2/org/lwjgl"
        self.urls = [f"{lwjgl}/{m}/{VERSION}/{m}-{VERSION}{suffix}" for m in MODULES
                     for suffix in (".jar", "-natives-linux-arm64.jar")]  # fmt: skip
        self.served = {u[len(base) :]: data for u, (data, _) in zip(self.urls, JARS)}
        shas = [sha256(data) for data, _ in JARS]
        self.fields = TINY | {
            "minestudio_dir": "ms", "engine_url": self.engine_url, "engine_size": len(ZIP),
            "engine_sha256": sha256(ZIP), "lwjgl_url": lwjgl, "lwjgl_version": VERSION,
            "lwjgl_sha256": {m: {"classes": shas[2 * i], "natives": shas[2 * i + 1]}
                             for i, m in enumerate(MODULES)},
            "http_timeout": 5.0, "http_retries": 2, "retry_delay": 0.0,
        }  # fmt: skip

    def app(self, req) -> None:
        if req.path == "/hf/engine.zip":
            send(req, 302, headers={"Location": self.location})
        elif req.path == "/cdn/engine.zip":
            z = self.download / "engine.zip"
            self.seen.append(z.read_bytes() if z.exists() else None)
            (self.cdn.pop(0) if self.cdn else ok)(self, req)
        elif self.busy.get(req.path):
            send(req, self.busy[req.path].pop(0), b"busy")
        elif req.path in self.served:
            send(req, 200, self.served[req.path])
        else:
            send(req, 404, b"not found")

    def run(self, argv=None) -> Run:
        write_config(self.path, self.fields)
        code = main([str(self.path)] if argv is None else argv)
        out, err = self.capsys.readouterr()
        return Run(code, out, err)

    def paths(self) -> list[str]:
        return [r.path for r in self.server.requests]

    def attempts(self) -> int:
        """Requests of the engine's second path."""
        return self.paths().count("/cdn/engine.zip")

    def jar_paths(self) -> list[str]:
        return list(self.served)

    def entries(self) -> list:
        """(name, data) of J."""
        with zipfile.ZipFile(self.jar) as z:
            return [(i.filename, z.read(i)) for i in z.infolist()]


def expected_entries() -> list:
    """(name, data) of the oracle's patched jar of the site's S and jars."""
    return [(info.filename, data) for info, data in ENTRIES]
