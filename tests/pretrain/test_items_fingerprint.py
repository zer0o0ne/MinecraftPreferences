"""`fingerprint` and `new_hash` of src/mcpref/pretrain/items.py against the transcription of
(4) of specs/train-index.md (index_oracle.py; specs/pretrain.md Interface): the directory
as written, resolved against the repository root; the complete shards only, in order;
`update()` after each."""

import hashlib
import json

import pytest

from mcpref.pretrain.items import fingerprint, new_hash

from .index_oracle import CLIP_FIELDS, field, sha, shard_files
from .index_run import repo_root
from .index_world import HAND, clips_run, write_clips, write_labels


def fields_of(run: dict) -> dict:
    """{dotted name: value} of the clip-defining fields, in their order."""
    return {f: field(run, f) for f in CLIP_FIELDS}


def call(tmp_path, monkeypatch, directory: str, file: str, fields) -> tuple[str, int]:
    """fingerprint from a working directory where `directory` leads nowhere; the count of
    update() calls."""
    calls = []
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    with repo_root(tmp_path / "repo"):
        got = fingerprint(directory, file, fields, lambda: calls.append(1))
    return got, len(calls)


@pytest.mark.parametrize("shards", [HAND.clips, HAND.pair_clips, []], ids=["C", "Q_c", "none"])  # fmt: skip
def test_clips_dataset(tmp_path, monkeypatch, shards):
    """records.json with the fields: SHA-256 of one json.dumps of the fields, then the
    records.json of each complete shard (not the staged one); update() once per shard."""
    run = clips_run(5, {"classifier": "d:été"})
    write_clips(tmp_path / "repo/w/clips", shards, run)
    got, updates = call(
        tmp_path, monkeypatch, "w/clips", "records.json", fields_of(run)
    )
    head = json.dumps(fields_of(run)).encode("utf-8")
    records = shard_files(tmp_path / "repo/w/clips/shards", "records.json")
    assert got == sha(head + b"".join(records))
    assert updates == len(shards) == len(records)


@pytest.mark.parametrize("shards", [HAND.pairs, []], ids=["pairs", "none"])
def test_labels_dataset(tmp_path, monkeypatch, shards):
    """pairs.json, fields None: SHA-256 of the pairs.json of the complete shards."""
    write_labels(tmp_path / "repo/w/labels", shards, {"complete": True})
    got, updates = call(tmp_path, monkeypatch, "w/labels", "pairs.json", None)
    pairs = shard_files(tmp_path / "repo/w/labels/shards", "pairs.json")
    assert got == sha(b"".join(pairs)) and updates == len(shards)


def test_directory_as_written(tmp_path, monkeypatch):
    """`w/clips/` (as written) is the same dataset."""
    run = clips_run(5)
    write_clips(tmp_path / "repo/w/clips", HAND.clips, run)
    a, _ = call(tmp_path, monkeypatch, "w/clips/", "records.json", fields_of(run))
    b, _ = call(tmp_path, monkeypatch, "w/clips", "records.json", fields_of(run))
    assert a == b


def test_new_hash():
    """The hash before the shards: json.dumps(fields) fed when not None; a fresh object."""
    fields = {"config.width": 640, "classifier": "é"}
    h = new_hash(fields)
    assert h.hexdigest() == sha(json.dumps(fields).encode("utf-8"))
    h.update(b"records")
    assert h.hexdigest() == sha(json.dumps(fields).encode("utf-8") + b"records")
    assert new_hash(fields).hexdigest() == sha(json.dumps(fields).encode("utf-8"))
    assert new_hash(None).hexdigest() == hashlib.sha256().hexdigest()
