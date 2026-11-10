"""Progress reporting for resumable shard processing."""

import contextlib
import sys
from collections.abc import Callable, Iterator

from tqdm import tqdm


@contextlib.contextmanager
def shard_bar(
    desc: str, n_shards: int, h: int, total: int, initial: int, unit: str
) -> Iterator[Callable[..., object]]:
    """Interface Progress bars: the bar of a set of n_shards continued at shard h, or its
    line (`nothing to do`: no shard, or no item left; `already done`: every shard complete)."""
    if n_shards == 0 or (h < n_shards and total == initial):
        print(f"{desc}: nothing to do", file=sys.stderr)
        yield lambda n=1: None
    elif h == n_shards:
        print(f"{desc}: already done", file=sys.stderr)
        yield lambda n=1: None
    else:
        with tqdm(total=total, initial=initial, desc=desc, unit=unit) as bar:
            yield bar.update
