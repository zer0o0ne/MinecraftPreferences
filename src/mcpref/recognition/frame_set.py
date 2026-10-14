"""Notation and (13) of specs/gui-masks.md: the labelled frames F, their parts, Q and the
folds of the training episodes."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mcpref.gui.label_files import INVALID, NO_ANSWER, ok_frames, read_jsonl


def read_frames(frames_dir: Path, labels_dir: Path) -> tuple[list[str], list[dict]]:
    """Notation: episodes of all `ok` records, and those records joined by id with
    their label y_f (key `label`)."""
    ok = [f for _, f in ok_frames(frames_dir)]
    labels = {r["id"]: r["label"] for r in read_jsonl(labels_dir / "labels.jsonl")}
    frames = [f | {"label": labels[f["id"]]} for f in ok if f["id"] in labels]
    return [f["relpath"] for f in ok], frames


def labelled_parts(frames: list[dict], parts: dict[str, int]) -> list[list[dict]]:
    """Notation: F^tr, F^va, F^te (parts 0, 1, 2) without `invalid`, `no_answer`."""
    kept = [f for f in frames if f["label"] not in (INVALID, NO_ANSWER)]
    return [[f for f in kept if parts[f["relpath"]] == i] for i in range(3)]


def frame_values(frames: Sequence[dict]) -> tuple[list[str], list[str], list[bool]]:
    """Notation: the id, the label y_f and the flag g_f of each frame, in order."""
    ids, y = [f["id"] for f in frames], [f["label"] for f in frames]
    return ids, y, [f["line"]["isGuiInventory"] for f in frames]


def qualified(train: list[dict], names: list[str], min_episodes: int) -> list[str]:
    """Notation Q: classes whose training frames span >= min_episodes episodes."""
    spans = {k: len({f["relpath"] for f in train if f["label"] == k}) for k in names}
    return [k for k in names if spans[k] >= min_episodes]


def folds(relpaths: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """(13): episode e_{pi[r]} gets fold r mod K, pi = permutation(n) of
    Generator(PCG64(seed)); relpaths = e_0 ... e_{n-1}."""
    pi = np.random.Generator(np.random.PCG64(seed)).permutation(len(relpaths))
    return {relpaths[i]: r % k for r, i in enumerate(pi)}
