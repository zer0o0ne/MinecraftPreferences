"""Independent jar lifecycle oracles for the atomic progress telemetry installer."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from mcpref.runtime.engine import telemetry


@pytest.fixture
def engine(tmp_path, monkeypatch):
    jar = tmp_path / "engine.jar"
    sources = {telemetry.ENV: b"pristine server", "Gameplay.class": b"untouched"}
    monkeypatch.setattr(
        telemetry,
        "SOURCE_HASHES",
        {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
    )
    with zipfile.ZipFile(jar, "w") as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
        archive.writestr("asset.png", b"original asset")
    return jar


def compiled(monkeypatch, engine_bytes=b"patched server"):
    calls = []

    def compile_once(jar, work, original):
        assert original == b"pristine server"
        calls.append(jar)
        return {telemetry.ENV: engine_bytes, telemetry.HELPER: b"read-only observer"}

    monkeypatch.setattr(telemetry, "_compile", compile_once)
    return calls


def contents(jar):
    with zipfile.ZipFile(jar) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        return {name: archive.read(name) for name in archive.namelist()}


def replace_entry(jar, name, data):
    entries = contents(jar) | {name: data}
    with zipfile.ZipFile(jar, "w") as archive:
        for key, value in entries.items():
            archive.writestr(key, value)


def test_install_preserves_gameplay_and_reuses_verified_output(engine, monkeypatch):
    before = contents(engine)
    calls = compiled(monkeypatch)
    assert telemetry.install(engine)
    after = contents(engine)
    assert after[telemetry.ENV] == b"patched server"
    assert after[telemetry.ORIGINAL] == before[telemetry.ENV]
    assert after["Gameplay.class"] == before["Gameplay.class"]
    assert after["asset.png"] == before["asset.png"]
    marker = json.loads(after[telemetry.MARKER])
    for name in (telemetry.ENV, telemetry.HELPER):
        assert marker["classes"][name] == hashlib.sha256(after[name]).hexdigest()
    installed = engine.read_bytes()
    assert not telemetry.install(engine)
    assert engine.read_bytes() == installed
    assert calls == [engine]


def test_upgrade_starts_from_pristine_class(engine, monkeypatch):
    compiled(monkeypatch)
    assert telemetry.install(engine)
    monkeypatch.setattr(telemetry, "_implementation", lambda: "new source revision")
    calls = compiled(monkeypatch, b"new patched server")
    assert telemetry.install(engine)
    assert contents(engine)[telemetry.ENV] == b"new patched server"
    assert contents(engine)[telemetry.ORIGINAL] == b"pristine server"
    assert calls == [engine]


@pytest.mark.parametrize("failure", ["compiler", "publication"])
def test_failure_leaves_original_jar_and_no_staging(engine, monkeypatch, failure):
    before = engine.read_bytes()
    compiled(monkeypatch)

    def fail(*args):
        raise RuntimeError("injected failure")

    if failure == "compiler":
        monkeypatch.setattr(telemetry, "_compile", fail)
    else:
        monkeypatch.setattr(telemetry, "publish", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        telemetry.install(engine)
    assert engine.read_bytes() == before
    assert list(engine.parent.iterdir()) == [engine]


@pytest.mark.parametrize("name", [telemetry.ENV, "Gameplay.class"])
def test_rejects_unexpected_engine_before_compiling(engine, monkeypatch, name):
    replace_entry(engine, name, b"another engine build")
    calls = compiled(monkeypatch)
    before = engine.read_bytes()
    with pytest.raises(RuntimeError, match="unexpected source"):
        telemetry.install(engine)
    assert engine.read_bytes() == before
    assert not calls


@pytest.mark.parametrize("name", [telemetry.ENV, telemetry.HELPER])
def test_rejects_tampered_installed_classes(engine, monkeypatch, name):
    compiled(monkeypatch)
    telemetry.install(engine)
    replace_entry(engine, name, b"modified installed code")
    before = engine.read_bytes()
    with pytest.raises(RuntimeError, match="modified installed class"):
        telemetry.install(engine)
    assert engine.read_bytes() == before


def test_rejects_unmarked_helper(engine, monkeypatch):
    calls = compiled(monkeypatch)
    replace_entry(engine, telemetry.HELPER, b"unknown helper")
    with pytest.raises(RuntimeError, match="unmarked telemetry"):
        telemetry.install(engine)
    assert not calls


def test_compiler_failure_is_not_published(engine, monkeypatch):
    before = engine.read_bytes()

    def fail_command(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="javac rejected source")

    monkeypatch.setattr(telemetry.subprocess, "run", fail_command)
    with pytest.raises(RuntimeError, match="javac rejected source"):
        telemetry.install(engine)
    assert engine.read_bytes() == before
    assert list(engine.parent.iterdir()) == [engine]


@pytest.mark.parametrize("changed", [True, False])
def test_cli_resolves_configuration_and_reports_result(monkeypatch, capsys, changed):
    monkeypatch.setattr(telemetry, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(telemetry, "engine_jar", lambda cfg: cfg["path"])
    jars = []
    monkeypatch.setattr(telemetry, "install", lambda jar: jars.append(jar) or changed)
    assert telemetry.main(["engine.yaml"]) == 0
    assert jars == [Path("engine.yaml")]
    assert ("installed" if changed else "already done") in capsys.readouterr().err


def test_cli_reports_compiler_diagnostics(monkeypatch, capsys):
    def fail(path):
        raise RuntimeError("javac failed: incompatible engine method")

    monkeypatch.setattr(telemetry, "load_config", fail)
    assert telemetry.main(["engine.yaml"]) == 1
    assert "incompatible engine method" in capsys.readouterr().err
