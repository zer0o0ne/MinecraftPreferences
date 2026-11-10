"""Disk helpers for tests of specs/shards.md: trees, snapshots, fake `os.fsync`."""

import errno
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TREE = {
    "a.bin": b"a",
    "empty.bin": b"",
    "sub": None,
    "sub/b.bin": b"bb",
    "sub/deep": None,
    "sub/deep/c.bin": b"ccc",
    "emptydir": None,
}  # relative path -> file bytes, None for a directory; parents before children


@dataclass(frozen=True)
class Sync:
    """One `os.fsync` call: the path of its fd, `after()` at the call, fd read-only."""

    path: Path
    after: bool
    read_only: bool = True


def ancestors(p: Path) -> set[Path]:
    """`p` and every directory above it up to `/`."""
    return {p, *p.parents}


def fd_path(fd: int) -> Path | None:
    """The path an open fd of this process refers to; None if `fd` is not open."""
    try:
        return Path(os.readlink(f"/proc/self/fd/{fd}"))
    except FileNotFoundError:
        return None


def is_read_only(fd: int) -> bool:
    """Whether `fd` was opened with access mode `O_RDONLY`."""
    info = Path(f"/proc/self/fdinfo/{fd}").read_text()
    flags = int(info.split("flags:")[1].split()[0], 8)
    return flags & os.O_ACCMODE == os.O_RDONLY


def record_syncs(monkeypatch, after: Callable[[], bool] = lambda: False) -> list[Sync]:
    """Replace `os.fsync` by a recorder of `Sync` entries that then calls the real one."""
    log, real = [], os.fsync

    def fsync(fd):
        log.append(Sync(fd_path(fd), after(), is_read_only(fd)))
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return log


def fail_syncs(monkeypatch, fails: Callable[[Path], bool]):
    """Replace `os.fsync`: raise `error` for an fd whose path satisfies `fails`.

    Returns (list of (fd, path) of the failed calls, error)."""
    error = OSError(errno.EIO, "fake fsync failure")
    failed, real = [], os.fsync

    def fsync(fd):
        path = fd_path(fd)
        if fails(path):
            failed.append((fd, path))
            raise error
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return failed, error


def published(path: Path, data: bytes) -> bool:
    """Whether `path` is a file holding `data`."""
    return path.is_file() and path.read_bytes() == data


def entries(root: Path) -> list[Path]:
    """Every entry under `root`, symlinks not followed."""
    out = []
    for top, dirs, files in os.walk(root):
        out += [Path(top, name) for name in dirs + files]
    return sorted(out)


def contents(root: Path) -> dict[str, bytes | None]:
    """Entries under `root` by relative path: file bytes, None for a directory."""
    return {
        p.relative_to(root).as_posix(): None if p.is_dir() else p.read_bytes()
        for p in entries(root)
    }


def snapshot(root: Path) -> dict[Path, tuple]:
    """`root` and every entry under it: kind, inode, mtime, bytes of a regular file."""
    out = {}
    for p in [root, *entries(root)]:
        st = p.lstat()
        data = p.read_bytes() if stat.S_ISREG(st.st_mode) else None
        out[p] = (stat.S_IFMT(st.st_mode), st.st_ino, st.st_mtime_ns, data)
    return out


def populate(D: Path, names: list[str]) -> None:
    """Create `D` and in it: `name/` a directory holding x.bin, `name` a file."""
    D.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name.endswith("/"):
            (D / name).mkdir()
            (D / name / "x.bin").write_bytes(b"x")
        else:
            (D / name).write_bytes(name.encode())


def write_tree(staged: Path, outside: Path) -> dict[str, bytes | None]:
    """Write TREE into `staged` with plain calls, then rename in a file from `outside`."""
    for rel, data in TREE.items():
        if data is None:
            (staged / rel).mkdir()
        else:
            (staged / rel).write_bytes(data)
    (outside / "moved.bin").write_bytes(b"moved")
    os.rename(outside / "moved.bin", staged / "sub" / "moved.bin")
    return {**TREE, "sub/moved.bin": b"moved"}
