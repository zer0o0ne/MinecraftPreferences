"""One writer per output directory, including nested calls in the same thread."""

import fcntl
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from mcpref.infrastructure.shards.store import make_dirs

_LOCAL = threading.local()


@contextmanager
def exclusive_run(root: Path):
    """Keep the sibling lock file: unlinking it would allow two locked inodes."""
    root = root.resolve()
    pid = os.getpid()
    held = getattr(_LOCAL, "held", {})
    if getattr(_LOCAL, "pid", pid) != pid:
        for handle in held.values():
            handle.close()
        held = {}
    _LOCAL.pid, _LOCAL.held = pid, held
    if root in held:
        yield
        return
    make_dirs(root.parent)
    with root.with_name(f"{root.name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is already using {root}") from None
        held[root] = handle
        try:
            yield
        finally:
            del held[root]
