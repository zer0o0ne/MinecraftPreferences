"""src/mcpref/gui/textures.py against specs/gui-screens.md (Interface): fake HTTP server, small zip."""

import dataclasses
import hashlib
import io
import itertools
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from mcpref.gui.textures import TexturesConfig, load_textures_config, main

TEXTURES = "assets/minecraft/textures/"
EXTRACTED = {  # path under out_dir -> bytes: every jar member under gui/, and font/ascii.png
    "gui/container/inventory.png": b"inventory",
    "gui/widgets.png": b"widgets",
    "gui/book.png": b"book",
    "gui/title/minecraft.png": b"named by no blit",
    "font/ascii.png": b"ascii",
}
KEPT = {  # jar members that are not extracted
    TEXTURES + "font/unicode_page_00.png": b"other font page",
    TEXTURES + "block/stone.png": b"not gui",
    TEXTURES + "guide/x.png": b"prefix gui without the slash",
    "assets/minecraft/lang/en_us.json": b"{}",
    "net/minecraft/client/main/Main.class": b"class",
}
NAMED = [
    "gui/container/inventory.png",
    "gui/widgets.png",
    "gui/book.png",
    "font/ascii.png",
]
KEYS = ("url", "sha1", "out_dir", "http_timeout", "http_retries", "retry_delay")
PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
PROXY_VARS += ("ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")
HOLD = 3.0  # s a "hold" reply waits; longer than every http_timeout below


def make_jar() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as jar:
        for name, data in EXTRACTED.items():
            jar.writestr(TEXTURES + name, data)
        for name, data in KEPT.items():
            jar.writestr(name, data)
    return buf.getvalue()


JAR = make_jar()


def files(root: Path) -> dict:
    found = root.rglob("*") if root.exists() else []
    return {
        p.relative_to(root).as_posix(): p.read_bytes() for p in found if p.is_file()
    }


def named(catalogue: dict) -> set:
    """Textures named by a blit, and font/ascii.png."""
    members = [m for c in catalogue["classes"].values() for m in c["members"].values()]
    return {b["texture"] for m in members for b in m["blits"]} | {"font/ascii.png"}


def url(server) -> str:
    return f"http://127.0.0.1:{server.port}/client.jar"


def scripted(*items):
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


def assert_failed(code: int, err: str, out: Path) -> None:
    """Exit 1 with a one-line message on stderr; nothing left in out_dir."""
    assert code == 1
    assert len(err.splitlines()) == 1 and err.strip(), err
    assert files(out) == {}


def gaps(server) -> list:
    return [b.t - a.t for a, b in itertools.pairwise(server.requests)]


@pytest.fixture
def out(tmp_path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def configure(catalogue, write_config, out, tmp_path, monkeypatch, repo):
    """Config for a URL with out_dir = `out` relative to the repository root.

    cwd is chosen so that out_dir resolved against it lands elsewhere in tmp_path;
    proxy variables are unset.
    """
    rel = os.path.relpath(out, repo)
    cwd = tmp_path.joinpath(*["cwd"] * (Path(rel).parts.count("..") + 1))
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)

    def make(address: str, **textures) -> str:
        sha1 = hashlib.sha1(JAR).hexdigest()
        fields = {"url": address, "sha1": sha1, "out_dir": rel} | textures
        catalogue["textures"].update(fields)
        return str(write_config(catalogue))

    return make


def test_load_textures_config_reads_textures_keys(
    catalogue, write_config, repo, tmp_path, monkeypatch
):
    """out_dir: absolute Path, resolved against the repository root, not the cwd."""
    monkeypatch.chdir(tmp_path)
    cfg = load_textures_config(write_config(catalogue))
    assert isinstance(cfg, TexturesConfig)
    assert {f.name for f in dataclasses.fields(cfg)} == set(KEYS)
    expected = catalogue["textures"] | {"out_dir": repo / "data/textures"}
    assert {k: getattr(cfg, k) for k in KEYS} == expected
    assert isinstance(cfg.out_dir, Path) and cfg.out_dir.is_absolute()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.url = "x"


def test_approved_config_names_the_1_16_5_client_jar(repo):
    """[J]: official 1.16.5 client jar."""
    cfg = load_textures_config(repo / "configs/gui-screens.yaml")
    assert cfg.sha1 == "37fd3c903861eeff3bc24b71eed48f828b5269c8"


@pytest.mark.parametrize("where", ["top", "textures"])
def test_textures_unknown_key_raises(catalogue, write_config, where):
    (catalogue if where == "top" else catalogue["textures"])["bogus"] = 1
    with pytest.raises(ValueError):
        load_textures_config(write_config(catalogue))


@pytest.mark.parametrize("key", KEYS)
def test_textures_missing_key_raises(catalogue, write_config, key):
    del catalogue["textures"][key]
    with pytest.raises(ValueError):
        load_textures_config(write_config(catalogue))


def test_downloads_checks_and_extracts_then_deletes_jar(http, configure, out):
    server = http(scripted(JAR))
    assert main([configure(url(server))]) == 0
    assert files(out) == EXTRACTED
    assert [(r.method, r.path) for r in server.requests] == [("GET", "/client.jar")]


def test_sha1_mismatch_deletes_jar_and_fails(http, configure, out, capsys):
    server = http(scripted(JAR))
    config = configure(url(server), sha1=hashlib.sha1(b"other").hexdigest())
    assert_failed(main([config]), capsys.readouterr().err, out)
    assert len(server.requests) == 1


def test_skips_when_named_textures_present(http, configure, out, catalogue):
    assert sorted(named(catalogue)) == sorted(NAMED)
    for name in NAMED:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b"present")
    server = http(scripted(JAR))
    assert main([configure(url(server))]) == 0
    assert server.requests == []
    assert files(out) == dict.fromkeys(NAMED, b"present")


@pytest.mark.parametrize("missing", NAMED)
def test_downloads_when_a_named_texture_is_missing(http, configure, out, missing):
    for name in set(NAMED) - {missing}:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b"present")
    server = http(scripted(JAR))
    assert main([configure(url(server))]) == 0
    assert len(server.requests) == 1
    assert files(out) == EXTRACTED


@pytest.mark.parametrize("first", [500, 503, "drop", "hold"])
def test_transient_failure_is_retried_after_delay(http, configure, out, first):
    server = http(scripted(first, JAR))
    config = configure(url(server), http_timeout=0.3, retry_delay=0.2)
    assert main([config]) == 0
    assert files(out) == EXTRACTED
    assert len(server.requests) == 2 and gaps(server)[0] >= 0.2
    assert first != "hold" or server.requests[0].cancelled()


@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("failure", [500, "drop"])
def test_retries_exhausted_fail(http, configure, out, capsys, retries, failure):
    server = http(scripted(failure))
    config = configure(url(server), http_retries=retries, retry_delay=0.1)
    assert_failed(main([config]), capsys.readouterr().err, out)
    assert len(server.requests) == retries + 1
    assert all(gap >= 0.1 for gap in gaps(server))


def test_refused_connection_is_retried(http, configure, out):
    server = http(scripted(JAR), listen=False)
    timer = threading.Timer(0.3, server.listen)
    config = configure(url(server), http_retries=100, retry_delay=0.05)
    timer.start()
    code = main([config])
    timer.join()
    assert code == 0
    assert len(server.requests) == 1 and files(out) == EXTRACTED


@pytest.mark.parametrize("status", [302, 403, 404])
def test_other_statuses_fail_at_once(http, configure, out, capsys, status):
    server = http(scripted(status))
    assert_failed(main([configure(url(server))]), capsys.readouterr().err, out)
    assert len(server.requests) == 1


def test_redirect_is_not_followed(http, configure, out, capsys):
    target = http(scripted(JAR))
    redirect = http(scripted(302), listen=False)

    class Located(redirect.server.RequestHandlerClass):
        def end_headers(self) -> None:
            self.send_header("Location", url(target))
            super().end_headers()

    redirect.server.RequestHandlerClass = Located
    redirect.listen()
    assert_failed(main([configure(url(redirect))]), capsys.readouterr().err, out)
    assert (len(redirect.requests), target.requests) == (1, [])


def test_proxy_from_environment_is_used(http, configure, out, monkeypatch):
    """Library mapping: client defaults, trust_env=True."""
    proxy = http(scripted(JAR))
    config = configure("http://textures.invalid/client.jar")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
    assert main([config]) == 0
    assert [r.path for r in proxy.requests] == ["http://textures.invalid/client.jar"]
    assert files(out) == EXTRACTED


@pytest.mark.parametrize("reply, code", [(JAR, 0), (404, 1)], ids=["ok", "404"])
def test_module_entry_point(http, configure, out, reply, code):
    """Run as `python -m mcpref.gui.textures <config>`: exit code and stderr."""
    config = configure(url(http(scripted(reply))))
    cmd = [sys.executable, "-m", "mcpref.gui.textures", config]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if code:
        assert_failed(proc.returncode, proc.stderr, out)
    else:
        assert proc.returncode == 0 and files(out) == EXTRACTED
