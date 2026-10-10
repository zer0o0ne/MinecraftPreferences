"""Notation of specs/gui-masks.md: the labelled frames F, their parts and Q."""

from pathlib import Path

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


def qualified(train: list[dict], names: list[str], min_episodes: int) -> list[str]:
    """Notation Q: classes whose training frames span >= min_episodes episodes."""
    spans = {k: len({f["relpath"] for f in train if f["label"] == k}) for k in names}
    return [k for k in names if spans[k] >= min_episodes]
