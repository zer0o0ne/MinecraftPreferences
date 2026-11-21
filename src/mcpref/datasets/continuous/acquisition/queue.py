"""Bounded FIFO lookahead; sources run concurrently, publication stays ordered."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import chain

from mcpref.config import REPO
from mcpref.datasets.continuous.acquisition.http import Transport
from mcpref.datasets.continuous.acquisition.space import Space, source_bytes
from mcpref.datasets.continuous.sources import acquire


class Acquisition:
    def __init__(self, pipe, cfg, settings, plan, first_fragment):
        self.pipe, self.cfg, self.settings = pipe, cfg, settings
        self.pieces = chain.from_iterable(
            pieces for k, _, pieces in plan if k >= first_fragment
        )
        self.pending, self.next_piece = deque(), None
        self.queued_bytes = 0
        self.space = Space(REPO / cfg.out_dir, cfg.disk_reserve_bytes)
        self.stack = ExitStack()

    def __enter__(self):
        self.transport = self.stack.enter_context(
            Transport(self.pipe.cfg, self.settings)
        )
        self.workers = ThreadPoolExecutor(self.settings.workers)
        self.stack.callback(self.workers.shutdown, wait=True, cancel_futures=True)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def fill(self, required=False):
        while len(self.pending) < self.settings.prefetch_sources:
            if self.next_piece is None:
                self.next_piece = next(self.pieces, None)
            if self.next_piece is None:
                return
            piece, size = self.next_piece, source_bytes(self.next_piece)
            if size > self.settings.prefetch_bytes:
                if required and not self.pending:
                    raise ValueError(
                        "Source exceeds prefetch_bytes; increase the transfer budget"
                    )
                return
            if self.queued_bytes + size > self.settings.prefetch_bytes:
                return
            allowance = size + self.settings.processing_reserve_bytes
            if not self.space.available(allowance):
                if not required or self.pending:
                    return
                self.space.check(size)
            self.space.check(size)
            job = self.workers.submit(
                acquire, self.pipe, self.cfg, piece, self.transport
            )
            self.space.track(job, size)
            self.queued_bytes += size
            self.pending.append((piece["piece_id"], size, job))
            self.next_piece = None

    def get(self, piece):
        self.fill(required=True)
        if not self.pending or self.pending[0][0] != piece["piece_id"]:
            raise ValueError("Acquisition order differs from the frozen plan")
        _, size, job = self.pending[0]
        result = job.result()
        self.pending.popleft()
        self.queued_bytes -= size
        self.fill()
        return result
