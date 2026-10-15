"""Interface Produces, (5)-(7) of specs/perception-data.md: records, staged and
committed files, the replay of the committed episodes."""

import dataclasses
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mcpref.clips.sample import Clip
from mcpref.gui.label_files import read_jsonl, sha256
from mcpref.vpt.episodes import Examined, episode_name, version_of

RECORD = ("subset", "relpath", "reason", "options", "starts_session")
RECORD += ("overlay_start", "valid_starts", "clips")
PROCESSED = (None, "video", "no_clip")


@dataclass(frozen=True)
class Episode:
    """(6): episode r ready to commit: its record (`clips` set at the commit), options
    file to store or None, the records of its staged clips in the order taken."""

    r: int
    record: dict
    options_file: bytes | None
    clips: list[dict]


def run_header(
    cfg: object, files: Mapping[str, Path], etags: dict, classifier_id: str
) -> dict:
    """(7): the config as written, the index ETags, the SHA-256 of `files`, gamma_id."""
    hashes = {k: sha256(path) for k, path in files.items()}
    header = {"config": dataclasses.asdict(cfg), "index_etags": etags, "sha256": hashes}
    return header | {"classifier": classifier_id}


def episode_record(subset: str, relpath: str, reason: str | None, **computed) -> dict:
    """Interface episodes.jsonl: a record, every field null unless computed."""
    fields = {"subset": subset, "relpath": relpath, "reason": reason} | computed
    return dict.fromkeys(RECORD) | fields


def clip_id(relpath: str, first: int) -> str:
    """Interface clips.jsonl: id = <version>_<stem>_<first state line>."""
    return f"{episode_name(relpath)}_{first}"


def clip_record(
    subset: str, relpath: str, clip: Clip, classes: Mapping[int, str]
) -> dict:
    """Interface clips.jsonl: a clip's record; per state its class, null when its GUI
    flag is false."""
    return {
        "id": clip_id(relpath, clip.states[0]),
        "subset": subset,
        "relpath": relpath,
        "version": version_of(relpath),
        "states": clip.states,
        "actions": clip.actions,
        "n": clip.n,
        "gui": clip.gui,
        "classes": [classes[t] if g else None for t, g in zip(clip.states, clip.gui)],
        "pitch": clip.pitch,
    }


def staging(out: Path, r: int) -> Path:
    """(2), (5): {out_dir}/staging/<r>, the staged clips of the episode drawn at r
    (its video: staging/<r>.mp4)."""
    return out / "staging" / str(r)


def line_starts(jsonl: bytes) -> list[int]:
    """Notation: the byte offset in J_e of each line l_t, then len(J_e) (twice when a
    final \\n ends the last line)."""
    ends = np.flatnonzero(np.frombuffer(jsonl, np.uint8) == ord("\n")) + 1
    return [0, *ends.tolist(), len(jsonl)]


def stage_clips(
    out: Path,
    r: int,
    relpath: str,
    clips: Sequence[Clip],
    frames: Mapping[int, bytes],
    jsonl: bytes,
) -> None:
    """(5): per clip, frames.yuv (its state frames in state order) and lines.jsonl (the
    lines of its span as in J_e, each with its \\n) under staging/<r>/<id>.part."""
    starts = line_starts(jsonl)
    for c in clips:
        part = staging(out, r) / f"{clip_id(relpath, c.states[0])}.part"
        part.mkdir(parents=True)
        (part / "frames.yuv").write_bytes(b"".join(frames[t] for t in c.states))
        span = jsonl[starts[c.states[0]] : starts[c.states[-1] + 1]]
        (part / "lines.jsonl").write_bytes(span)


def remove(path: Path) -> None:
    """(5)-(7): a staging directory removed, if present."""
    if path.exists():
        shutil.rmtree(path)


def append(path: Path, records: Sequence[dict]) -> None:
    """(6): records appended as JSON lines; the file closed (flushed)."""
    with path.open("a") as f:
        f.write("".join(json.dumps(record) + "\n" for record in records))


def store_options(out: Path, relpath: str, options_file: bytes) -> None:
    """Interface: options/<version>_<stem>-options.json, byte-identical."""
    path = out / "options" / f"{episode_name(relpath)}-options.json"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(options_file)


def commit(out: Path, ep: Episode, left: int) -> dict:
    """(6): the options file, the first `left` clips moved to clips/ and appended to
    clips.jsonl, the record listing their ids (returned); the staging of r removed."""
    taken = ep.clips[:left]
    if ep.options_file is not None:
        store_options(out, ep.record["relpath"], ep.options_file)
    (out / "clips").mkdir(exist_ok=True)
    for c in taken:
        (staging(out, ep.r) / f"{c['id']}.part").rename(out / "clips" / c["id"])
    append(out / "clips.jsonl", taken)
    ids = [c["id"] for c in taken] if ep.record["reason"] is None else None
    record = ep.record | {"clips": ids}
    append(out / "episodes.jsonl", [record])
    remove(staging(out, ep.r))
    return record


def replay(out: Path) -> list[dict]:
    """(7): the committed records of episodes.jsonl; clips.jsonl and clips/ cut to their
    clips (a commit interrupted before its record); staging emptied."""
    path, clips = out / "episodes.jsonl", out / "clips.jsonl"
    records = read_jsonl(path) if path.exists() else []
    ids = {i for record in records for i in record["clips"] or ()}
    if clips.exists():
        lines = clips.read_bytes().splitlines(keepends=True)
        clips.write_bytes(b"".join(x for x in lines if json.loads(x)["id"] in ids))
    if (out / "clips").exists():
        for d in (out / "clips").iterdir():
            if d.name not in ids:
                shutil.rmtree(d)
    remove(out / "staging")
    return records


def committed_clips(records: Sequence[dict]) -> int:
    """(6): c, the clips committed with the records."""
    return sum(len(record["clips"] or ()) for record in records)


def replayed(
    records: Sequence[dict], examine: Callable[[str, str], Examined]
) -> Callable[[str, str], Examined]:
    """(7): `examine` answering the first len(records) draws with the recorded reasons,
    nothing fetched; `video`, `no_clip` and null were valid for the plan."""
    recorded = iter(records)

    def examined(basedir: str, relpath: str) -> Examined:
        record = next(recorded, None)
        if record is None:
            return examine(basedir, relpath)
        reason = None if record["reason"] in PROCESSED else record["reason"]
        return Examined(reason, None, None, None, None)

    return examined
