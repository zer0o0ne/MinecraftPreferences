"""Interface Consumes and Produces of specs/gui-labels.md: frames read, label values,
files written."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mcpref.vpt.gui_frames import yuv420p_to_rgb

OTHER = "other"
INVALID = "invalid"
NO_ANSWER = "no_answer"


def read_jsonl(path: Path) -> list[dict]:
    """One JSON object per line."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def frame_size(frames_dir: Path) -> tuple[int, int]:
    """(1): `width`, `height` of the config stored in {frames_dir}/run.json."""
    frames_config = json.loads((frames_dir / "run.json").read_text())["config"]
    return frames_config["width"], frames_config["height"]


def ok_frames(frames_dir: Path) -> list[tuple[int, dict]]:
    """Notation, (2): `ok` records with their 0-based positions over all records."""
    frames = enumerate(read_jsonl(frames_dir / "frames.jsonl"))
    return [(position, f) for position, f in frames if f["status"] == "ok"]


def rgb_view(frames_dir: Path, frame: dict, size: tuple[int, int]) -> np.ndarray:
    """(1): x_f, the RGB view (vpt-gui-frames (5)) of frames/<id>.yuv."""
    buf = (frames_dir / "frames" / f"{frame['id']}.yuv").read_bytes()
    return yuv420p_to_rgb(buf, *size)


def sha256(path: Path) -> str:
    """(4): SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rerun(path: Path, hashes: dict[str, str]) -> None:
    """(4): an existing run.json must hold the same three SHA-256 values."""
    if path.exists() and json.loads(path.read_text())["sha256"] != hashes:
        raise RuntimeError(f"{path} holds other SHA-256 values: {hashes}")


def finished(out: Path) -> dict[str, dict]:
    """(4): records of earlier runs, by id."""
    path = out / "labels.jsonl"
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def shown(q: list[str | np.ndarray]) -> list[str]:
    """Interface run.json: the prompt with the image replaced by "<image>"."""
    return [part if isinstance(part, str) else "<image>" for part in q]


def write_atomic(path: Path, text: str) -> None:
    """(4): a temporary file, then a rename."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_outputs(
    out: Path, header: dict, records: list[dict], vpt_check: dict[str, int]
) -> None:
    """Interface Produces: run.json, labels.jsonl, summary.json."""
    write_atomic(out / "run.json", json.dumps(header, indent=2) + "\n")
    write_atomic(out / "labels.jsonl", "".join(json.dumps(r) + "\n" for r in records))
    labels = Counter(r["label"] for r in records)
    summary = {"labels": labels, "vpt_check": vpt_check}
    write_atomic(out / "summary.json", json.dumps(summary, indent=2) + "\n")
