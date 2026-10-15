"""src/mcpref/gui/textures.py against specs/gui-screens.md (Interface): the config, and the
download (fake HTTP server serving a small zip; retries, statuses, sha1, proxy, entry point)."""

import dataclasses
import itertools
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from textures_site import (
    DIRS,
    JAR,
    T,
    assert_failed,
    contents,
    make_site,
    sha1,
    url,
)

from mcpref.gui.textures import TexturesConfig, load_textures_config, main

KEYS = ("url", "sha1", "out_dir", "http_timeout", "http_retries", "retry_delay")
WRITTEN = DIRS | T  # out_dir after a complete run from an empty out_dir


@pytest.fixture
def site(http, catalogue, write_config, tmp_path, monkeypatch):
    return make_site(http, catalogue, write_config, tmp_path, monkeypatch)


def gaps(server) -> list:
    return [b.t - a.t for a, b in itertools.pairwise(server.requests)]


def needed(classes: dict) -> set:
    """T: the textures named by a blit of any member, and font/ascii.png."""
    members = [m for c in classes.values() for m in c["members"].values()]
    return {b["texture"] for m in members for b in m["blits"]} | {"font/ascii.png"}


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


def test_approved_catalogue_needs_at_most_25_textures(classes):
    """Progress bars: none, as T holds at most 25 files."""
    assert len(needed(classes)) <= 25


def test_catalogue_fixture_needs_t(catalogue):
    assert needed(catalogue["classes"]) == set(T)


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


def test_relative_out_dir_resolved_against_repository_root(
    site, repo, tmp_path, monkeypatch
):
    """The cwd is chosen so that out_dir resolved against it lands elsewhere in tmp_path."""
    rel = os.path.relpath(site.out, repo)
    cwd = tmp_path.joinpath(*["cwd"] * (Path(rel).parts.count("..") + 1))
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)
    server = site.serve(JAR)
    assert main([site.config(url(server), out_dir=rel)]) == 0
    assert contents(site.out) == WRITTEN


def test_sha1_mismatch_fails_and_writes_nothing(site, capsys):
    server = site.serve(JAR)
    config = site.config(url(server), sha1=sha1(b"other"))
    assert_failed(main([config]), capsys.readouterr().err, site.out)
    assert len(server.requests) == 1


@pytest.mark.parametrize("first", [500, 503, "drop", "hold"])
def test_transient_failure_is_retried_after_delay(site, first):
    server = site.serve(first, JAR)
    config = site.config(url(server), http_timeout=0.3, retry_delay=0.2)
    assert main([config]) == 0
    assert contents(site.out) == WRITTEN
    assert len(server.requests) == 2 and gaps(server)[0] >= 0.2
    assert first != "hold" or server.requests[0].cancelled()


@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("failure", [500, "drop"])
def test_retries_exhausted_fail(site, capsys, retries, failure):
    server = site.serve(failure)
    config = site.config(url(server), http_retries=retries, retry_delay=0.1)
    assert_failed(main([config]), capsys.readouterr().err, site.out)
    assert len(server.requests) == retries + 1
    assert all(gap >= 0.1 for gap in gaps(server))


def test_refused_connection_is_retried(site):
    server = site.serve(JAR, listen=False)
    timer = threading.Timer(0.3, server.listen)
    config = site.config(url(server), http_retries=100, retry_delay=0.05)
    timer.start()
    code = main([config])
    timer.join()
    assert code == 0
    assert len(server.requests) == 1 and contents(site.out) == WRITTEN


@pytest.mark.parametrize("status", [302, 403, 404])
def test_other_statuses_fail_at_once(site, capsys, status):
    server = site.serve(status)
    assert_failed(main([site.config(url(server))]), capsys.readouterr().err, site.out)
    assert len(server.requests) == 1


def test_redirect_is_not_followed(site, capsys):
    target = site.serve(JAR)
    redirect = site.serve(302, listen=False)

    class Located(redirect.server.RequestHandlerClass):
        def end_headers(self) -> None:
            self.send_header("Location", url(target))
            super().end_headers()

    redirect.server.RequestHandlerClass = Located
    redirect.listen()
    config = site.config(url(redirect))
    assert_failed(main([config]), capsys.readouterr().err, site.out)
    assert (len(redirect.requests), target.requests) == (1, [])


def test_proxy_from_environment_is_used(site, monkeypatch):
    """Library mapping: client defaults, trust_env=True."""
    proxy = site.serve(JAR)
    config = site.config("http://textures.invalid/client.jar")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
    assert main([config]) == 0
    assert [r.path for r in proxy.requests] == ["http://textures.invalid/client.jar"]
    assert contents(site.out) == WRITTEN


@pytest.mark.parametrize("reply, code", [(JAR, 0), (404, 1)], ids=["ok", "404"])
def test_module_entry_point(site, reply, code):
    """Run as `python -m mcpref.gui.textures <config>`: exit code and stderr."""
    config = site.config(url(site.serve(reply)))
    cmd = [sys.executable, "-m", "mcpref.gui.textures", config]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if code:
        assert_failed(proc.returncode, proc.stderr, site.out)
    else:
        assert proc.returncode == 0 and contents(site.out) == WRITTEN
