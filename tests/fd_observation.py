"""Observe real file descriptors in durability tests on macOS and Linux."""

import errno
import fcntl
import os
import sys
from pathlib import Path


def fd_path(fd: int) -> Path | None:
    """Resolve an open descriptor; return None only when it is no longer open."""
    try:
        if sys.platform == "darwin":
            value = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
            return Path(os.fsdecode(value.split(b"\0", 1)[0]))
        return Path(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError as error:
        if error.errno not in (errno.EBADF, errno.ENOENT):
            raise
        return None


def is_read_only(fd: int) -> bool:
    """Observe the descriptor's access mode without reopening its path."""
    return fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
