"""`main` of `mcpref.engine.prepare` (specs/engine.md (0)-(6)) with the fake host of
prepare_site.py: J built from the jars, D removed at (0) and (6), (1) without any request,
sha256 mismatches, J never half-written, a rerun after a failure, the arguments."""

import subprocess
import sys

import pytest
from progress_bars import assert_aborted

from mcpref.engine.prepare import main

from .engine_world import TINY, write_config
from .prepare_site import PREFIX, ZIP, engine_zip, expected_entries, sha256

WRONG = "f" * 64


def test_builds_the_jar(site):
    """The 2|M| jars in order, then the engine through its redirect; J holds S's entries
    not in R, then A; D removed; nothing on stdout."""
    r = site.run()
    assert (r.code, r.out) == (0, ""), r.err
    assert site.paths() == site.jar_paths() + ["/hf/engine.zip", "/cdn/engine.zip"]
    assert site.entries() == expected_entries()
    assert not site.download.exists()
    assert sorted(p.name for p in site.jar.parent.iterdir()) == ["mcprec-6.13.jar"]


def test_identity_encoding_on_both_paths(site):
    """`Accept-Encoding: identity` on the request and on the redirected one."""
    assert site.run().code == 0
    engine = [r for r in site.server.requests if r.path.endswith("/engine.zip")]
    assert [r.headers.get_all("Accept-Encoding") for r in engine] == [["identity"]] * 2


@pytest.mark.parametrize("index", [0, 7, 13])  # lwjgl classes, lwjgl-stb natives, last
def test_lwjgl_sha256_mismatch(site, index):
    """(2): the jar's URL, its sha256 and the expected one; the later jars and the engine
    not requested; nothing written: D absent, no J."""
    module = list(site.fields["lwjgl_sha256"])[index // 2]
    site.fields["lwjgl_sha256"][module]["natives" if index % 2 else "classes"] = WRONG
    r = site.run()
    assert_aborted(r.code, r.err)
    data = site.served[site.jar_paths()[index]]
    line = f"{site.urls[index]}: sha256 {sha256(data)} != {WRONG}"
    assert r.err.splitlines()[-1] == f"{PREFIX}ValueError: {line}"
    assert site.paths() == site.jar_paths()[: index + 1]
    assert not site.download.exists() and not site.jar.exists()


def test_engine_sha256_mismatch(site):
    """(3): `engine_url` as configured (not the redirect target); no J."""
    site.fields["engine_sha256"] = WRONG
    r = site.run()
    assert_aborted(r.code, r.err)
    line = f"{site.engine_url}: sha256 {sha256(ZIP)} != {WRONG}"
    assert r.err.splitlines()[-1] == f"{PREFIX}ValueError: {line}"
    assert not site.jar.exists()


def test_already_done(site):
    """(0), (1): a leftover D removed; J present: one line, exit 0, no request, J kept."""
    site.jar.parent.mkdir(parents=True)
    site.jar.write_bytes(b"an earlier engine")
    (site.download / "engine/build/libs").mkdir(parents=True)
    (site.download / "engine.zip").write_bytes(b"partial")
    r = site.run()
    assert (r.code, r.out, r.err) == (0, "", "engine: already done\n")
    assert site.paths() == []
    assert site.jar.read_bytes() == b"an earlier engine"
    assert not site.download.exists()


def test_leftover_download_removed_at_start(site):
    """(0): D removed before (2): a failing (2) leaves no D."""
    (site.download / "engine").mkdir(parents=True)
    (site.download / "stale.txt").write_text("stale")
    site.fields["lwjgl_sha256"]["lwjgl"]["classes"] = WRONG
    assert site.run().code == 1
    assert not site.download.exists()


def test_leftover_download_replaced(site):
    """(0), (3): a leftover D with a stale Z and other files does not reach J; D removed."""
    site.download.mkdir(parents=True)
    (site.download / "engine.zip").write_bytes(engine_zip(b"not a jar"))
    (site.download / "stale.txt").write_text("stale")
    r = site.run()
    assert r.code == 0, r.err
    assert site.seen == [None]  # removed before the engine was fetched
    assert site.entries() == expected_entries() and not site.download.exists()


def test_leftover_part_overwritten(site):
    """(5): a leftover part_path(J) is overwritten, then published as J."""
    site.jar.parent.mkdir(parents=True)
    part = site.jar.with_name(site.jar.name + ".part")
    part.write_bytes(b"x" * 100_000)
    assert site.run().code == 0
    assert site.entries() == expected_entries() and not part.exists()


def test_jar_never_half_written(site):
    """(4)-(5) failing on a member that is no zip: exit 1, no J."""
    site.zip = engine_zip(b"PK not a jar")
    site.fields["engine_sha256"] = sha256(site.zip)
    r = site.run()
    assert_aborted(r.code, r.err)
    assert r.err.splitlines()[-1].startswith(f"{PREFIX}BadZipFile: ")
    assert not site.jar.exists()


def test_rerun_after_failure(site):
    """A rerun starts again at (0) and completes."""
    site.fields["engine_sha256"] = WRONG
    assert site.run().code == 1
    site.fields["engine_sha256"] = sha256(ZIP)
    r = site.run()
    assert r.code == 0, r.err
    assert site.entries() == expected_entries() and not site.download.exists()


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    """One positional config path; otherwise argparse's usage, exit 2."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "usage: python -m mcpref.engine.prepare" in capsys.readouterr().err


def test_python_m(tmp_path):
    """`python -m mcpref.engine.prepare <config>`: a config error → exit 1 and the line,
    before anything is touched."""
    path = write_config(tmp_path / "c.yaml", TINY | {"engine_size": 0})
    proc = subprocess.run(
        [sys.executable, "-m", "mcpref.engine.prepare", str(path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert proc.returncode == 1 and proc.stdout == ""
    lines = proc.stderr.splitlines()
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith(f"{PREFIX}ValueError: engine_size 0 is not ")
