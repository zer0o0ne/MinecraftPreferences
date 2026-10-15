"""Fake jar, fake server replies, run site and recorders for tests of the texture download of
specs/gui-screens.md (Interface), for the needed set T of the `catalogue` fixture."""

import contextlib
import hashlib
import io
import os
import sys
import warnings
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "shards"))
from store_disk import contents, fd_path, is_read_only

from mcpref.gui.textures import main

TEXTURES = "assets/minecraft/textures/"
T = {  # T of the `catalogue` fixture (its blits' textures, font/ascii.png) -> member bytes
    "font/ascii.png": b"ascii",
    "gui/book.png": b"book",
    "gui/container/inventory.png": b"inventory",
    "gui/widgets.png": b"widgets",
}
DIRS = dict.fromkeys(["font", "gui", "gui/container"])  # parents of T under out_dir
OTHER = [  # jar members outside T
    (TEXTURES + "gui/title/minecraft.png", b"under gui/, named by no blit"),
    (TEXTURES + "font/unicode_page_00.png", b"another font page"),
    (TEXTURES + "guide/x.png", b"prefix gui without the slash"),
    (TEXTURES + "block/stone.png", b"not gui"),
    ("assets/minecraft/lang/en_us.json", b"{}"),
    ("net/minecraft/client/main/Main.class", b"class"),
]
ENV_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY")
ENV_VARS += ("all_proxy", "NO_PROXY", "no_proxy", "SSLKEYLOGFILE")
HOLD = 3.0  # s a "hold" reply waits; longer than every http_timeout of the tests


def jar(members: list) -> bytes:
    """A zip of (name, bytes) pairs in this order; a repeated name is stored twice."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zipfile: "Duplicate name"
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in members:
                z.writestr(name, data)
    return buf.getvalue()


def members(textures: dict) -> list:
    return [(TEXTURES + p, data) for p, data in textures.items()]


JAR = jar(members(T) + OTHER)


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def url(server) -> str:
    return f"http://127.0.0.1:{server.port}/client.jar"


def scripted(*items) -> Callable:
    """App answering the n-th request with items[n] (the last repeats): jar bytes,
    a status with an empty body, "drop" (no reply) or "hold" (HOLD s, then JAR)."""
    calls = []

    def app(req):
        item = items[min(len(calls), len(items) - 1)]
        calls.append(req)
        if item == "hold":
            req.hold(HOLD)
            item = JAR
        if item == "drop":
            return None
        return (item, b"") if isinstance(item, int) else (200, item)

    return app


@dataclass
class Site:
    """`out`: absolute out_dir, free of symlinks and of `..`; cwd elsewhere; no proxy."""

    http: Callable
    catalogue: dict
    write_config: Callable
    out: Path

    def config(self, address: str, **textures) -> str:
        fields = {"url": address, "sha1": sha1(JAR), "out_dir": str(self.out)}
        self.catalogue["textures"].update(fields | textures)
        return str(self.write_config(self.catalogue))

    def serve(self, *items, listen: bool = True):
        return self.http(scripted(*items), listen=listen)

    def run(self, body: bytes = JAR):
        """`main` against a server answering `body` (sha1 of `body` configured)."""
        server = self.serve(body)
        return main([self.config(url(server), sha1=sha1(body))]), server

    def put(self, files: dict) -> None:
        """Files present under out_dir before a run."""
        for rel, data in files.items():
            (self.out / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.out / rel).write_bytes(data)


def make_site(http, catalogue, write_config, tmp_path, monkeypatch) -> Site:
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return Site(http, catalogue, write_config, tmp_path.resolve() / "out")


def assert_failed(code: int, err: str, out: Path) -> None:
    """Exit 1 with a one-line message on stderr; no file under out_dir (make_dirs ran)."""
    assert code == 1
    assert len(err.splitlines()) == 1 and err.strip(), err
    assert contents(out) == DIRS


@dataclass(frozen=True)
class Sync:
    """One `os.fsync`: the path of its fd, whether the server got a request by then,
    the entries under out_dir then (relative), whether the fd is read-only."""

    path: Path
    requested: bool
    entries: frozenset
    read_only: bool


def record_syncs(monkeypatch, out: Path, server) -> list[Sync]:
    """Replace `os.fsync` by a recorder of `Sync` entries that then calls the real one."""
    log, real = [], os.fsync

    def fsync(fd):
        state = frozenset(contents(out))
        log.append(Sync(fd_path(fd), bool(server.requests), state, is_read_only(fd)))
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return log


_WRITES: list[list[Path]] = []
_HOOKED: list[bool] = []


def _audit(event: str, args: tuple) -> None:
    """Audit hook: an `open` with write access appends its absolute path to _WRITES[-1]."""
    if event != "open" or not _WRITES or isinstance(args[0], int):
        return
    flags = args[2]
    if flags & os.O_ACCMODE != os.O_RDONLY or flags & (os.O_CREAT | os.O_TRUNC):
        _WRITES[-1].append(Path(os.path.abspath(os.fsdecode(args[0]))))


@contextlib.contextmanager
def opened_for_writing():
    """Yields the list of files this process opens for writing within the block,
    bytecode caches excluded (the hook stays installed, idle outside the block)."""
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED.append(True)
    _WRITES.append([])
    try:
        yield _WRITES[-1]
    finally:
        found = _WRITES.pop()
        found[:] = [p for p in found if "__pycache__" not in p.parts]
