"""`prune` of src/mcpref/pretrain/checkpoints.py against specs/pretrain.md (Interface): the
`checkpoint.pt` of each checkpoint 0 .. last - k that still has one deleted, then its
directory synced; nothing else touched."""

import os
from pathlib import Path

import pytest

from mcpref.pretrain.checkpoints import prune

CASES = {  # checkpoints, last, k, those without checkpoint.pt before, deleted
    "last 4, k 2": (5, 4, 2, (), [0, 1, 2]),
    "last 4, k 4": (5, 4, 4, (), [0]),
    "last - k = -1": (5, 4, 5, (), []),
    "checkpoints after last untouched": (5, 2, 1, (), [0, 1]),
    "some already deleted": (5, 4, 2, (0, 1), [2]),
    "a gap already deleted": (6, 5, 1, (1, 3), [0, 2, 4]),
    "one checkpoint": (1, 0, 1, (), []),
}


@pytest.mark.parametrize("n, last, k, without, deleted", CASES.values(), ids=CASES)
def test_prune(tmp_path, monkeypatch, n, last, k, without, deleted):
    directory = tmp_path / "checkpoints"
    for j in range(n):
        d = directory / f"{j:06d}"
        d.mkdir(parents=True)
        (d / "losses.json").write_bytes(b"[]\n")
        if j not in without:
            (d / "checkpoint.pt").write_bytes(b"x")
    synced, real = [], os.fsync

    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        synced.append((path, (path / "checkpoint.pt").exists()))
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    prune(directory, last, k)
    for j in range(n):
        d = directory / f"{j:06d}"
        assert (d / "checkpoint.pt").exists() == (j not in without and j not in deleted)
        assert (d / "losses.json").read_bytes() == b"[]\n"
        assert {p.name for p in d.iterdir()} <= {"checkpoint.pt", "losses.json"}
    for j in deleted:
        assert (directory / f"{j:06d}", False) in synced
