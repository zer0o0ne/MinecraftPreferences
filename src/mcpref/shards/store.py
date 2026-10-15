"""Sharded datasets on disk, written atomically and durably: specs/shards.md."""

import contextlib
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

PART = ".part"


def name(k: int) -> str:
    """name(k): six digits, more when k >= 10**6."""
    return f"{k:06d}"


def index(entry: str) -> int | None:
    """The k with name(k) == entry, else None."""
    if entry.isascii() and entry.isdigit() and name(int(entry)) == entry:
        return int(entry)
    return None


def shard_path(D: Path, k: int) -> Path:
    """Complete shard k: D/name(k)."""
    return D / name(k)


def part_path(path: Path) -> Path:
    """The temporary name <path>.part of `path`; the staged shard of a complete one."""
    return Path(f"{path}{PART}")


def is_directory(p: Path) -> bool:
    """p is a directory, symlinks not followed."""
    return p.is_dir() and not p.is_symlink()


def sync(p: Path) -> None:
    """sync(p): fsync of p opened read-only; closed also when fsync raises."""
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_dirs(p: Path) -> None:
    """make(p): missing directories down to p created; p and every ancestor up to / synced."""
    p.mkdir(parents=True, exist_ok=True)
    absolute = p.absolute()
    for q in (absolute, *absolute.parents):
        sync(q)


def check(D: Path, k: int) -> None:
    """(1).1: shard k - 1 complete, no entry named name(k)."""
    if k > 0 and not is_directory(shard_path(D, k - 1)):
        raise ValueError(f"{D}: shard {name(k - 1)} missing before {name(k)}")
    if os.path.lexists(shard_path(D, k)):
        raise FileExistsError(f"{shard_path(D, k)} exists")


def remove(p: Path) -> None:
    """(1).3: a leftover staged entry removed: rmtree for a directory, unlink otherwise."""
    if is_directory(p):
        shutil.rmtree(p)
    elif os.path.lexists(p):
        os.unlink(p)


def sync_tree(p: Path) -> None:
    """(1).4: every regular file and directory under p, at any depth, and p synced."""
    for child in p.iterdir():
        if is_directory(child):
            sync_tree(child)
        else:
            sync(child)
    sync(p)


@contextlib.contextmanager
def write_shard(D: Path, k: int) -> Iterator[Path]:
    """(1): yields the staged shard k; on a normal return it is synced, renamed to
    D/name(k) and D synced; on an exception it is left as it is."""
    check(D, k)
    make_dirs(D)
    shard = shard_path(D, k)
    staged = part_path(shard)
    remove(staged)
    staged.mkdir()
    yield staged
    sync_tree(staged)
    os.rename(staged, shard)
    sync(D)


def is_staged(entry: str) -> bool:
    """entry is name(k).part for some k."""
    return entry.endswith(PART) and index(entry[: -len(PART)]) is not None


def shard_indices(D: Path) -> set[int]:
    """(2): the k of D's complete shards; the smallest foreign entry raises."""
    ks = set()
    for entry in sorted(os.listdir(D)):
        if is_staged(entry):
            continue
        k = index(entry)
        if k is None or not is_directory(D / entry):
            raise ValueError(f"{D}: foreign entry {entry}")
        ks.add(k)
    return ks


def complete_shards(D: Path) -> int:
    """(2): n of D, 0 when D is missing; the smallest missing shard raises; D synced."""
    if not D.exists():
        return 0
    ks = shard_indices(D)
    for j in range(len(ks)):
        if j not in ks:
            raise ValueError(f"{D}: shard {name(j)} missing")
    sync(D)
    return len(ks)


def publish(part: Path, path: Path) -> None:
    """(3): `part` synced, renamed to `path` (an existing file replaced), the parent synced."""
    sync(part)
    os.rename(part, path)
    sync(path.parent)


def write_file(path: Path, data: bytes) -> None:
    """(4): make of the parent; `data` written to <path>.part, closed; published as `path`."""
    make_dirs(path.parent)
    part = part_path(path)
    part.write_bytes(data)
    publish(part, path)
