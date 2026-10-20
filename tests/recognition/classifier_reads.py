"""The fake gui-masks output of the (L) tests of specs/gui-classifier.md and recorders of
the reads of `load_classifier` and gamma: the files opened (`open` audit events) and the
arrays read from .npz files (`NpzFile.__getitem__`; Library mapping: `numpy.load`)."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from classifier_world import Masks, shard_arrays

W, H = 6, 4
THRESHOLDS = {"agreement": (3, "0.5", "0.1"), "macro_recall": (5, "0.75", "0")}


class Opens:
    """Real paths of the files opened while active."""

    active = None

    def __init__(self):
        self.paths = []


def audit(event: str, args: tuple) -> None:
    rec = Opens.active
    if rec is not None and event == "open" and isinstance(args[0], (str, bytes, Path)):
        rec.paths.append(Path(os.path.realpath(os.fsdecode(args[0]))))


sys.addaudithook(audit)  # for the life of the process; idle without an active recorder


@contextmanager
def opens():
    rec = Opens()
    Opens.active = rec
    try:
        yield rec
    finally:
        Opens.active = None


@contextmanager
def array_reads():
    """Keys read from NpzFile objects while active."""
    keys, real = [], np.lib.npyio.NpzFile.__getitem__

    def getitem(self, key):
        keys.append(key)
        return real(self, key)

    np.lib.npyio.NpzFile.__getitem__ = getitem
    try:
        yield keys
    finally:
        np.lib.npyio.NpzFile.__getitem__ = real


def pair(rng, h: int = H, w: int = W) -> tuple:
    mask = rng.random((h, w)) < 0.5
    mask[0, 0] = True
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mask


def world(tmp_path, repo) -> Masks:
    """inv: m1, m3 with (d) pairs, m2 without; chest; empty: no pair; craft: a member named
    like chest's and one with a non-ASCII name."""
    rng = np.random.default_rng(5)
    classes = {"inv": {"m1": pair(rng), "m2": None, "m3": pair(rng)}}
    classes |= {"chest": {"main": pair(rng)}, "empty": {"x": None}}
    classes["craft"] = {"main": pair(rng), "é1": pair(rng)}
    return Masks(tmp_path / "w", repo, classes, THRESHOLDS, W, H).write()


def rewrite(masks: Masks, k: str, member: str, new: tuple) -> None:
    """Replaces the (d) pair of `member` of class k in its shard."""
    masks.classes[k][member] = new
    s = list(masks.classes).index(k)
    path = masks.out / "templates/catalogue" / f"{s:06d}" / "pairs.npz"
    np.savez(path, **shard_arrays(masks.classes[k], W, H))
