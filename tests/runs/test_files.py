"""`finished_run` (1), `sha256` (2) and `write_json` (3) of src/mcpref/runs/files.py against
specs/runs.md; the helpers its revision removes elsewhere. (4): test_files_fingerprints.py."""

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from mcpref.runs.files import finished_run, sha256, write_json

WRITTEN = "./data/run_as_written/"  # as written in a config: kept verbatim in messages
MARKERS = {"complete": True, "clips": 4, "frames": {"ok": 2, "failed": 1}}  # the table


def write_run(directory: Path, run) -> None:
    (directory / "run.json").write_text(json.dumps(run, indent=2) + "\n")


FINISHED = [("complete", True), ("clips", 0), ("clips", 7), ("frames", {})]
FINISHED += [("frames", {"ok": 3, "failed": 1})]


@pytest.mark.parametrize("marker, value", FINISHED)
def test_finished_run_returns_the_parsed_run_file(tmp_path, marker, value):
    """(1): the marker holds neither false nor null (0 and {} count as present): the
    parsed run.json, every key kept."""
    run = {"config": {"seed": 0}, "sha256": {"config": "0a" * 32}, marker: value}
    write_run(tmp_path, run)
    got = finished_run(tmp_path, WRITTEN, marker)
    assert got == run and type(got) is dict


NOT_FINISHED = {  # run.json contents for the marker m; None: no file
    "no file": lambda m: None,
    "only the other markers": lambda m: {k: v for k, v in MARKERS.items() if k != m},
    "marker only in sha256": lambda m: {"sha256": {m: MARKERS[m]}},
    "false": lambda m: {**MARKERS, m: False},
    "null": lambda m: {**MARKERS, m: None},
}


@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize("case", NOT_FINISHED)
def test_finished_run_rejects(tmp_path, case, marker):
    """(1) fails: RuntimeError(f"{written} is not complete"), `written` verbatim."""
    run = NOT_FINISHED[case](marker)
    if run is not None:
        write_run(tmp_path, run)
    with pytest.raises(RuntimeError) as error:
        finished_run(tmp_path, WRITTEN, marker)
    assert str(error.value) == f"{WRITTEN} is not complete"


def test_finished_run_on_a_missing_directory(tmp_path):
    """(1): no {directory} at all is a missing file too."""
    with pytest.raises(RuntimeError, match=r"^w/frames is not complete$"):
        finished_run(tmp_path / "absent", "w/frames", "frames")


def test_finished_run_propagates_json_errors(tmp_path):
    """Domain: a run.json that is not JSON is out of domain; json.loads' error propagates."""
    (tmp_path / "run.json").write_text('{"complete": tr')
    with pytest.raises(json.JSONDecodeError):
        finished_run(tmp_path, WRITTEN, "complete")


VECTORS = {  # FIPS 180-2 test vectors; one megabyte spans any read buffer
    b"": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    b"abc": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq": (
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    ),
    b"a" * 10**6: "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
}


@pytest.mark.parametrize("data", VECTORS, ids=["empty", "abc", "448 bits", "1e6 a"])
def test_sha256_known_vectors(tmp_path, data):
    """(2): lowercase hex SHA-256 of the file's bytes."""
    (tmp_path / "f.bin").write_bytes(data)
    assert sha256(tmp_path / "f.bin") == VECTORS[data]


def test_sha256_of_bytes_not_text(tmp_path):
    """(2): the bytes as stored: CR LF and non-UTF-8 bytes untranslated."""
    data = b"a\r\nb\xff\x00\r"
    (tmp_path / "f.bin").write_bytes(data)
    assert sha256(tmp_path / "f.bin") == hashlib.sha256(data).hexdigest()


JSON_TEXT = {  # (3) by hand: indent 2, keys in order, non-ASCII escaped, final newline
    "nested": (
        {"b": [1, 2.5], "a": {"é": None, "t": True}, "s": 'x"y'},
        (
            '{\n  "b": [\n    1,\n    2.5\n  ],\n  "a": {\n    "\\u00e9": null,\n'
            '    "t": true\n  },\n  "s": "x\\"y"\n}\n'
        ),
    ),
    "empty": ({}, "{}\n"),
}


@pytest.mark.parametrize("case", JSON_TEXT)
def test_write_json_bytes(tmp_path, case):
    value, text = JSON_TEXT[case]
    write_json(tmp_path / "run.json", value)
    assert (tmp_path / "run.json").read_bytes() == text.encode()


def test_write_json_replaces_a_leftover_part_and_the_old_file(tmp_path):
    """(3) by specs/shards.md (4): a leftover <path>.part truncated, published as path."""
    path = tmp_path / "run.json"
    Path(f"{path}.part").write_bytes(b"x" * 100)
    path.write_bytes(b"an older run file, longer than the new one")
    write_json(path, {"complete": True})
    assert path.read_bytes() == b'{\n  "complete": true\n}\n'
    assert os.listdir(tmp_path) == ["run.json"]


def test_write_json_creates_the_parent(tmp_path):
    write_json(tmp_path / "a/b/run.json", {})
    assert (tmp_path / "a/b/run.json").read_bytes() == b"{}\n"


def test_write_json_syncs_before_and_after_the_rename(tmp_path, monkeypatch):
    """(3) by specs/shards.md (4), (3): <path>.part synced before path exists, the parent
    synced last, once path exists."""
    base = tmp_path.resolve()
    path, log, real = base / "run.json", [], os.fsync

    def fsync(fd):
        log.append((Path(os.readlink(f"/proc/self/fd/{fd}")), path.exists()))
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    write_json(path, {"clips": 0})
    assert (Path(f"{path}.part"), False) in log and log[-1] == (base, True)


REMOVED = [("mcpref.gui.label_files", n) for n in ("sha256", "write_json")]
REMOVED += [("mcpref.gui.label_files", "check_fingerprints")]
REMOVED += [("mcpref.recognition.stored", "complete_run")]
REMOVED += [("mcpref.pretrain.index", "counted_run")]
REMOVED += [("mcpref.vpt.config", n) for n in ("check_rerun", "write_run")]


@pytest.mark.parametrize("module, name", REMOVED)
def test_removed_helpers(module, name):
    """Interface: moved to runs/files.py, replaced by finished_run, or dead: absent."""
    assert not hasattr(importlib.import_module(module), name)
