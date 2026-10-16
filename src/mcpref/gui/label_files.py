"""Interface Consumes and Produces of specs/gui-labels.md: frames read, label values,
files written."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mcpref.shards.store import shard_path, write_file, write_shard
from mcpref.vpt.gui_frames import yuv420p_to_rgb

OTHER = "other"
INVALID = "invalid"
NO_ANSWER = "no_answer"
LABELS = "labels.json"
INVENTORY = {"inventory", "inventory_book"}
VPT_PAIRS = {
    "inventory_flag_true": (True, True),
    "inventory_flag_false": (True, False),
    "other_flag_true": (False, True),
    "other_flag_false": (False, False),
}


def frame_size(frames_dir: Path) -> tuple[int, int]:
    """(1): `width`, `height` of the config stored in {frames_dir}/run.json."""
    frames_config = json.loads((frames_dir / "run.json").read_text())["config"]
    return frames_config["width"], frames_config["height"]


def run_counts(path: Path, written: Path | str) -> dict[str, int]:
    """(4), Notation; specs/gui-masks.md (R): the `ok`, `failed` counts of the frames'
    run.json at `path`, 0 when absent; abort while it lacks them (`written` shown)."""
    run = json.loads(path.read_text())
    if "frames" not in run:
        raise RuntimeError(f"{written} lacks the counts: the frames are not complete")
    return {status: run["frames"].get(status, 0) for status in ("ok", "failed")}


def frame_counts(frames_dir: Path) -> dict[str, int]:
    """(4), Notation: the counts of {frames_dir}/run.json; abort while it lacks them or
    has no `ok` frame."""
    path = frames_dir / "run.json"
    counts = run_counts(path, path)
    if counts["ok"] < 1:
        raise RuntimeError(f"no ok frame in {path}")
    return counts


def yuv_path(frames_dir: Path, frame: dict) -> Path:
    """Notation: {frames_dir}/frames/<id>.yuv; the frame is `ok` iff it exists."""
    return frames_dir / "frames" / f"{frame['id']}.yuv"


def rgb_view(frames_dir: Path, frame: dict, size: tuple[int, int]) -> np.ndarray:
    """(1): x_f, the RGB view (vpt-gui-frames (5)) of frames/<id>.yuv."""
    return yuv420p_to_rgb(yuv_path(frames_dir, frame).read_bytes(), *size)


def sha256(path: Path) -> str:
    """(4): SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(q: list[str | np.ndarray]) -> list[str]:
    """Interface run.json: the prompt with the image replaced by "<image>"."""
    return [part if isinstance(part, str) else "<image>" for part in q]


def write_json(path: Path, value: dict) -> None:
    """(4); specs/gui-masks.md: a JSON file written by specs/shards.md (4)."""
    write_file(path, (json.dumps(value, indent=2) + "\n").encode())


def write_run(out: Path, header: dict) -> None:
    """(4): {out_dir}/run.json by specs/shards.md (4)."""
    write_json(out / "run.json", header)


def check_fingerprints(path: Path, fingerprints: dict, out_dir: str) -> None:
    """(R) of specs/gui-masks.md, specs/gui-stack.md: an existing run.json `path` holds
    `fingerprints` (top-level keys or keys of its `sha256`), compared in order."""
    if path.exists():
        stored = json.loads(path.read_text())
        for name, value in fingerprints.items():
            if (stored | stored["sha256"])[name] != value:
                raise RuntimeError(
                    f"{name} changed since the run started; delete {out_dir}"
                )


def read_labels(labels_dir: Path, k: int) -> dict:
    """Interface Produces: labels/<name(k)>/labels.json."""
    return json.loads((shard_path(labels_dir, k) / LABELS).read_text())


def write_labels(labels_dir: Path, k: int, records: list[dict], end: int) -> None:
    """(4): label shard k by specs/shards.md (1), its records and `end`."""
    with write_shard(labels_dir, k) as staged:
        (staged / LABELS).write_text(json.dumps({"labels": records, "end": end}))


def resume(labels_dir: Path, h: int) -> tuple[int, int]:
    """(4): the records of the h complete label shards, read one at a time, and `end` of
    label shard h - 1 (0 when h = 0)."""
    records, end = 0, 0
    for k in range(h):
        shard = read_labels(labels_dir, k)
        records, end = records + len(shard["labels"]), shard["end"]
    return records, end


def labelled(record: dict) -> bool:
    """(5); specs/gui-masks.md Notation: a record with a class label (`other` included),
    not `invalid` or `no_answer`."""
    return record["label"] not in (INVALID, NO_ANSWER)


def write_summary(out: Path, H: int) -> None:
    """(4'), (5): summary.json from the H label shards read one at a time, plain write;
    the VPT table counts (y_f in {inventory, inventory_book}, g_f) over class labels."""
    labels, pairs = Counter(), Counter()
    for k in range(H):
        for r in read_labels(out / "labels", k)["labels"]:
            labels[r["label"]] += 1
            if labelled(r):
                pairs[r["label"] in INVENTORY, r["gui_inventory"]] += 1
    vpt_check = {key: pairs[pair] for key, pair in VPT_PAIRS.items()}
    summary = {"labels": labels, "vpt_check": vpt_check}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
