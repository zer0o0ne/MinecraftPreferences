"""(3) `publish`, (4) `write_file` and make(p) `make_dirs` of specs/shards.md."""

import os
from pathlib import Path

import pytest
from store_disk import (
    Sync,
    ancestors,
    fail_syncs,
    fd_path,
    published,
    record_syncs,
    snapshot,
)

from mcpref.shards.store import (
    complete_shards,
    make_dirs,
    publish,
    write_file,
    write_shard,
)

OLD = [None, b"old content, longer than the new one"]


@pytest.mark.parametrize("old", OLD, ids=["new path", "existing path"])
def test_publish_renames_part_to_path(base, old):
    part, path = base / "clip.yuv.part", base / "clip.yuv"
    part.write_bytes(b"new")
    inode = part.stat().st_ino
    if old is not None:
        path.write_bytes(old)
    publish(part, path)
    assert path.read_bytes() == b"new" and path.stat().st_ino == inode
    assert os.listdir(base) == ["clip.yuv"]


@pytest.mark.parametrize("old", OLD, ids=["new path", "existing path"])
def test_publish_syncs_file_then_parent(base, monkeypatch, old):
    part, path = base / "clip.yuv.part", base / "clip.yuv"
    part.write_bytes(b"new")
    if old is not None:
        path.write_bytes(old)
    log = record_syncs(monkeypatch, lambda: published(path, b"new"))
    publish(part, path)
    assert log == [Sync(part, False), Sync(base, True)]


def test_publish_failed_sync_of_part_publishes_nothing(base, monkeypatch):
    part, path = base / "clip.yuv.part", base / "clip.yuv"
    part.write_bytes(b"new")
    fail_syncs(monkeypatch, lambda p: p == part)
    with pytest.raises(OSError):
        publish(part, path)
    assert os.listdir(base) == ["clip.yuv.part"] and part.read_bytes() == b"new"


@pytest.mark.parametrize("old", OLD, ids=["new path", "existing path"])
def test_write_file_writes_path(base, old):
    path = base / "f.bin"
    if old is not None:
        path.write_bytes(old)
    write_file(path, b"data")
    assert path.read_bytes() == b"data"
    assert os.listdir(base) == ["f.bin"]


def test_write_file_truncates_leftover_part(base):
    path, part = base / "f.bin", base / "f.bin.part"
    part.write_bytes(b"leftover, longer than data")
    with open(part, "rb") as held:  # keeps the leftover's inode from being reused
        write_file(path, b"data")
        assert path.read_bytes() == b"data"
        assert path.stat().st_ino == os.fstat(held.fileno()).st_ino
    assert os.listdir(base) == ["f.bin"]


def test_write_file_creates_missing_parents(base):
    path = base / "a" / "b" / "f.bin"
    write_file(path, b"data")
    assert path.read_bytes() == b"data"
    assert os.listdir(base) == ["a"] and os.listdir(base / "a" / "b") == ["f.bin"]


@pytest.mark.parametrize("parent_exists", [False, True])
def test_write_file_syncs(base, monkeypatch, parent_exists):
    parent = base / "a" / "b"
    path = parent / "f.bin"
    if parent_exists:
        parent.mkdir(parents=True)
        path.write_bytes(b"old")
    log = record_syncs(monkeypatch, lambda: published(path, b"data"))
    write_file(path, b"data")
    assert {s.path for s in log[:-2]} == ancestors(parent)
    assert not any(s.after for s in log[:-2]) and all(s.read_only for s in log)
    assert log[-2:] == [Sync(parent / "f.bin.part", False), Sync(parent, True)]


@pytest.mark.parametrize("existing", [None, "a", "a/b", "a/b/c"])
def test_make_dirs_creates_and_syncs_p_and_all_ancestors(base, monkeypatch, existing):
    p = base / "a" / "b" / "c"
    if existing is not None:
        (base / existing).mkdir(parents=True)
    log = record_syncs(monkeypatch, p.is_dir)
    make_dirs(p)
    assert p.is_dir()
    assert {s.path for s in log} == ancestors(p)
    assert all(s.after and s.read_only for s in log)  # synced once p exists


def test_make_dirs_on_existing_p_changes_nothing(base):
    p = base / "a" / "b"
    p.mkdir(parents=True)
    (p / "f.bin").write_bytes(b"f")
    before = snapshot(base)
    make_dirs(p)
    assert snapshot(base) == before


def test_make_dirs_of_relative_p_syncs_its_ancestors_up_to_root(base, monkeypatch):
    monkeypatch.chdir(base)
    log = record_syncs(monkeypatch)
    make_dirs(Path("a/b"))
    assert (base / "a" / "b").is_dir()
    assert {s.path for s in log} == ancestors(base / "a" / "b")


def call_publish(base):
    (base / "f.part").write_bytes(b"f")
    publish(base / "f.part", base / "f")


def call_write_shard(base):
    with write_shard(base / "D", 0):
        pass


def call_complete_shards(base):
    (base / "D").mkdir()
    complete_shards(base / "D")


CALLS = {
    "make_dirs": lambda base: make_dirs(base / "a"),
    "write_file": lambda base: write_file(base / "f", b"f"),
    "publish": call_publish,
    "write_shard": call_write_shard,
    "complete_shards": call_complete_shards,
}


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_sync_closes_fd_when_fsync_raises(base, monkeypatch, call):
    failed, error = fail_syncs(monkeypatch, lambda p: True)
    with pytest.raises(OSError) as caught:
        call(base)
    assert caught.value is error
    [(fd, path)] = failed
    assert fd_path(fd) != path
