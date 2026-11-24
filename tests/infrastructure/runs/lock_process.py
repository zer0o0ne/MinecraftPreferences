"""A separate process holding a real output lock until released or killed."""

import subprocess
import sys
from contextlib import contextmanager

HOLDER = """
import sys
from contextlib import ExitStack
from pathlib import Path
from mcpref.infrastructure.runs.lock import exclusive_run
from mcpref.infrastructure.shards.store import write_shard
root = Path(sys.argv[1])
with ExitStack() as stack:
    stack.enter_context(exclusive_run(root))
    staged = None
    if len(sys.argv) > 2:
        staged = stack.enter_context(write_shard(root / 'checkpoints', 0))
        (staged / 'checkpoint.pt').write_bytes(b'first weights')
    print('locked', flush=True)
    sys.stdin.readline()
    if staged is not None:
        (staged / 'metadata.json').write_bytes(b'first archive')
"""


@contextmanager
def holding(root, shard=False):
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(root)] + (["shard"] if shard else []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline() == "locked\n"
        yield process
    finally:
        if process.poll() is None:
            process.communicate(input="release\n", timeout=2)
        else:
            process.communicate(timeout=2)


def attempt(root, shard=False):
    script = """
import sys
from pathlib import Path
from mcpref.infrastructure.cli.failure import failed
from mcpref.infrastructure.runs.lock import exclusive_run
from mcpref.infrastructure.shards.store import write_shard
root = Path(sys.argv[1])
try:
    with exclusive_run(root):
        if len(sys.argv) > 2:
            with write_shard(root / 'checkpoints', 0) as staged:
                (staged / 'checkpoint.pt').write_bytes(b'second weights')
                (staged / 'metadata.json').write_bytes(b'second archive')
        else:
            (root / 'checkpoint.pt').write_bytes(b'new checkpoint')
except RuntimeError as error:
    sys.exit(failed('pretrain', error))
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(root)] + (["shard"] if shard else []),
        capture_output=True,
        check=False,
        text=True,
        timeout=3,
    )
