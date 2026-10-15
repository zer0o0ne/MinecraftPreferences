"""(1) of specs/shards.md: writing shard k of D with `write_shard`."""

import os

import pytest
from store_disk import (
    ancestors,
    contents,
    fail_syncs,
    populate,
    record_syncs,
    snapshot,
    write_tree,
)

from mcpref.shards.store import write_shard


def test_write_shard_renames_staged_shard_with_its_tree(base):
    D = base / "D"
    with write_shard(D, 0) as staged:
        assert staged == D / "000000.part"
        assert staged.is_dir() and os.listdir(staged) == []
        tree = write_tree(staged, base)
        inode = staged.stat().st_ino
    assert contents(D / "000000") == tree
    assert (D / "000000").stat().st_ino == inode  # renamed, not copied
    assert os.listdir(D) == ["000000"]


def test_write_shard_after_previous_leaves_it_untouched(base):
    D = base / "D"
    populate(D, ["000000/"])
    shard_0 = snapshot(D / "000000")
    with write_shard(D, 1) as staged:
        assert staged == D / "000001.part"
        (staged / "f.bin").write_bytes(b"f")
    assert sorted(os.listdir(D)) == ["000000", "000001"]
    assert contents(D / "000001") == {"f.bin": b"f"}
    assert snapshot(D / "000000") == shard_0


def test_write_shard_creates_D_with_its_parents(base):
    D = base / "x" / "y" / "D"
    with write_shard(D, 0) as staged:
        (staged / "f.bin").write_bytes(b"f")
    assert os.listdir(base) == ["x"] and os.listdir(base / "x" / "y") == ["D"]
    assert contents(D) == {"000000": None, "000000/f.bin": b"f"}


@pytest.mark.parametrize("error", [RuntimeError("caller"), KeyboardInterrupt()])
def test_write_shard_caller_raising_leaves_staged_shard_as_it_is(base, error):
    D = base / "D"
    with pytest.raises(type(error)) as caught, write_shard(D, 0) as staged:
        write_tree(staged, base)
        left = snapshot(staged)
        raise error
    assert caught.value is error
    assert os.listdir(D) == ["000000.part"]
    assert snapshot(staged) == left


def test_write_shard_replaces_staged_shard_of_raising_caller(base):
    D = base / "D"
    with pytest.raises(RuntimeError), write_shard(D, 0) as staged:
        write_tree(staged, base)
        raise RuntimeError
    with write_shard(D, 0) as staged:
        assert staged.is_dir() and os.listdir(staged) == []
        (staged / "new.bin").write_bytes(b"new")
    assert os.listdir(D) == ["000000"]
    assert contents(D / "000000") == {"new.bin": b"new"}


@pytest.mark.parametrize(
    "leftover", ["000000.part/", "000000.part"], ids=["dir", "file"]
)
def test_write_shard_removes_leftover_staged_entry(base, leftover):
    D = base / "D"
    populate(D, [leftover])
    if leftover.endswith("/"):
        populate(D / "000000.part", ["sub/", "y.bin"])
    with write_shard(D, 0) as staged:
        assert staged.is_dir() and os.listdir(staged) == []
        (staged / "new.bin").write_bytes(b"new")
    assert os.listdir(D) == ["000000"]
    assert contents(D / "000000") == {"new.bin": b"new"}


CHECKS = [  # entries of D (None: D missing), k, exception, message
    (None, 1, ValueError, "{D}: shard 000000 missing before 000001"),
    (["000001.part/"], 1, ValueError, "{D}: shard 000000 missing before 000001"),
    (["000000", "000001.part"], 1, ValueError, "{D}: shard 000000 missing before 000001"),
    (["000000.part/"], 1, ValueError, "{D}: shard 000000 missing before 000001"),
    (["000008/", "000010.part/"], 10, ValueError, "{D}: shard 000009 missing before 000010"),
    (["000000/", "000000.part/"], 0, FileExistsError, "{D}/000000 exists"),
    (["000000", "000000.part"], 0, FileExistsError, "{D}/000000 exists"),
    (["000000/", "000001/", "000001.part/"], 1, FileExistsError, "{D}/000001 exists"),
    (["000000/", "000001"], 1, FileExistsError, "{D}/000001 exists"),
    (["000001/", "000001.part/"], 1, ValueError, "{D}: shard 000000 missing before 000001"),
]  # fmt: skip


@pytest.mark.parametrize("names, k, exception, message", CHECKS)
def test_write_shard_checks_change_nothing(base, names, k, exception, message):
    D = base / "D"
    if names is not None:
        populate(D, names)
    before, ran = snapshot(base), False
    with pytest.raises(exception) as caught, write_shard(D, k):
        ran = True
    assert type(caught.value) is exception
    assert str(caught.value) == message.format(D=D)
    assert not ran and snapshot(base) == before


@pytest.mark.parametrize("names, k", [(None, 0), ([], 0), (["000000/"], 1)])
def test_write_shard_syncs(base, monkeypatch, names, k):
    D = base / "a" / "D"
    if names is not None:
        populate(D, names)
    name = ["000000", "000001"][k]
    log = record_syncs(monkeypatch, lambda: os.path.lexists(D / name))
    with write_shard(D, k) as staged:
        tree = write_tree(staged, base)
    shard = {staged} | {staged / rel for rel in tree}
    assert {s.path for s in log if not s.after} == ancestors(D) | shard
    assert [s.path for s in log if s.after] == [D]
    assert all(s.read_only for s in log)


def test_write_shard_failed_sync_leaves_no_complete_shard(base, monkeypatch):
    D = base / "D"
    failed, error = fail_syncs(monkeypatch, lambda p: p.name == "c.bin")
    with pytest.raises(OSError) as caught, write_shard(D, 0) as staged:
        write_tree(staged, base)
    assert caught.value is error and len(failed) == 1
    assert not os.path.lexists(D / "000000")
