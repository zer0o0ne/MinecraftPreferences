"""Fake classifier gamma of specs/perception-data.md (Notation): it finds each frame it is
given by its RGB view, vpt-gui-frames (5) by the exact oracle of tests/vpt/test_yuv.py,
answers the class of collect_world.Ep.cls and records its calls."""

import threading
import time

import numpy as np
from vpt.test_yuv import exact_rgb, upsampled

from .collect_world import H, W


def rgb(frame: bytes) -> np.ndarray:
    """vpt-gui-frames (5) of a raw frame, by tests/vpt/test_yuv.py."""
    y = np.frombuffer(frame[: W * H], np.uint8).reshape(H, W).astype(np.int64)
    u, v = (np.frombuffer(frame[W * H + k * W * H // 4 :][: W * H // 4], np.uint8)
            .reshape(H // 2, W // 2).astype(np.int64) for k in (0, 1))  # fmt: skip
    return exact_rgb(y, upsampled(u, H, W), upsampled(v, H, W))


class Classifier:
    """gamma: finds each frame by its RGB view and answers Ep.cls; records each call's
    frames as (stem, t); raises `self.raised` at call `fail_at`; holds each call `pause`
    s."""

    def __init__(self, world, fail_at: int | None = None, pause: float = 0.0):
        eps = {e.stem: e for e in world.eps.values()}
        self.index = {rgb(fr).tobytes(): (s, t) for s, e in eps.items()
                      for t, fr in enumerate(e.frames())}  # fmt: skip
        assert len(self.index) == sum(len(e.script) + 1 for e in eps.values())
        self.eps, self.fail_at, self.pause = eps, fail_at, pause
        self.calls, self.bad, self.lock, self.active, self.max_active = [], [], threading.Lock(), 0, 0  # fmt: skip

    def __call__(self, frames: list) -> list[str]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.pause)
            if len(self.calls) == self.fail_at:
                self.calls.append("raised")
                self.raised = RuntimeError("classifier broke")
                raise self.raised
            ok = isinstance(frames, list) and frames and all(
                isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.shape == (H, W, 3)
                for x in frames)  # fmt: skip
            ids = [self.index.get(x.tobytes()) for x in frames] if ok else [None]
            self.calls.append(ids)
            if None in ids:
                self.bad.append(frames)
                raise AssertionError("not a list of RGB views of known frames")
            return [self.eps[s].cls(t) for s, t in ids]
        finally:
            with self.lock:
                self.active -= 1

    def by_stem(self) -> dict:
        """stem -> the frames of each of its calls, in order (every call within one)."""
        out = {}
        for ids in self.calls:
            assert ids != "raised" and len({s for s, _ in ids}) == 1, ids
            out.setdefault(ids[0][0], []).append([t for _, t in ids])
        return out
