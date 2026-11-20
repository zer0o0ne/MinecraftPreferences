"""Linear source reads and crash-safe publication of pending diagnostic events."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from diagnostics.core.test_artifacts import pending
from diagnostics.core.world import write_config
from mcpref.diagnostics.core import artifacts, runtime, state


@pytest.fixture
def queued(tmp_path, monkeypatch):
    for module in (artifacts, runtime, state):
        monkeypatch.setattr(module, "REPO", tmp_path)
    config = write_config(tmp_path / "diagnostics.yaml")
    run = runtime.DiagnosticRun.load(config)
    run.register("mcts", watermark=0, next=0)
    for index in range(8):
        source = f"step-{index}.pt"
        (tmp_path / source).write_bytes(bytes([index]) * 128)
        record = pending(f"step:{index}") | {"kind": "mcts"}
        run.stage("mcts", index, record["id"], index + 1, [source], record)
    return run, config


def test_recovery_reads_sources_and_outboxes_linearly(queued, monkeypatch):
    run, _ = queued
    reads = Counter()
    original = Path.read_bytes

    def read(path):
        reads[path] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    run.recover("mcts")
    source_reads = sum(n for path, n in reads.items() if path.name.startswith("step-"))
    outbox_reads = sum(n for path, n in reads.items() if "outbox" in path.parts)
    assert source_reads == 8
    assert outbox_reads <= 16


def assert_complete(run, root):
    value = json.loads((run.out_dir / "run.json").read_bytes())
    channel = value["channels"]["mcts"]
    assert (channel["next"], channel["records"], len(channel["events"])) == (8, 8, 8)
    assert [event["id"] for event in channel["events"]] == [
        f"step:{i}" for i in range(8)
    ]
    expected = b"".join(
        f"step-{i}.pt".encode() + b"\0" + bytes([i]) * 128 for i in range(8)
    )
    assert value["sha256"]["inputs"]["mcts"] == hashlib.sha256(expected).hexdigest()
    index = json.loads((run.out_dir / "index.json").read_bytes())
    assert [record["id"] for record in index["records"]] == [
        f"step:{i}" for i in range(8)
    ]
    assert not list(run.outbox("mcts"))
    assert all(
        (root / f"step-{i}.pt").read_bytes() == bytes([i]) * 128 for i in range(8)
    )


@pytest.mark.parametrize("failure", ["record", "identity", "save", "remove"])
def test_recovery_after_interrupted_batch_preserves_records(
    queued, monkeypatch, failure
):
    run, config = queued
    original_write = runtime.write_record
    original_remove = run.remove

    def fail(*args, **kwargs):
        raise OSError("interrupted publication")

    def write(directory, kind, index, record):
        original_write(directory, kind, index, record)
        if index == 3:
            fail()

    def remove(paths):
        original_remove(paths[:2])
        fail()

    with monkeypatch.context() as patch:
        if failure == "record":
            patch.setattr(runtime, "write_record", write)
        elif failure == "identity":
            patch.setattr(run, "identity", fail)
        elif failure == "save":
            patch.setattr(run, "save", fail)
        else:
            patch.setattr(run, "remove", remove)
        with pytest.raises(OSError, match="interrupted publication"):
            run.recover("mcts")
    saved = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in run.out_dir.glob("mcts/*/metrics.json")
    }
    resumed = runtime.DiagnosticRun.load(config)
    resumed.register("mcts", watermark=0, next=0)
    resumed.recover("mcts")
    assert_complete(resumed, config.parent)
    assert all(
        (path.stat().st_mtime_ns, path.read_bytes()) == before
        for path, before in saved.items()
    )


def test_recovery_rebuilds_identity_with_previously_accepted_events(queued):
    run, config = queued
    run.accept("mcts", "step:0")
    resumed = runtime.DiagnosticRun.load(config)
    resumed.register("mcts", watermark=0, next=1)
    resumed.recover("mcts")
    assert_complete(resumed, config.parent)


def test_changed_retained_input_is_rejected_before_recovery(queued):
    run, config = queued
    run.accept("mcts", "step:0")
    (config.parent / "step-0.pt").write_bytes(b"changed")
    resumed = runtime.DiagnosticRun.load(config)
    with pytest.raises(RuntimeError, match="source inputs changed"):
        resumed.register("mcts", watermark=0, next=1)
    assert len(list(run.outbox("mcts"))) == 7


def test_grouped_event_retains_record_order_and_discards_unpublished_source(queued):
    run, config = queued
    extra = pending("extra") | {"kind": "mcts"}
    run.stage("mcts", 8, "step:0", 1, ["step-0.pt"], extra)
    (config.parent / "step-7.pt").unlink()
    run.recover("mcts")
    records = list(artifacts.iter_records(run.out_dir, "mcts"))
    assert [record["id"] for record in records] == [
        "step:0",
        "extra",
        *(f"step:{i}" for i in range(1, 7)),
    ]
    assert run.state["counts"]["mcts"] == {"events": 7, "records": 8}
    assert run.state["channels"]["mcts"]["next"] == 7
    assert not list(run.outbox("mcts"))


def test_conflicting_record_after_crash_is_not_silently_accepted(queued, monkeypatch):
    run, _ = queued

    def fail():
        raise OSError("interrupted before metadata")

    with monkeypatch.context() as patch:
        patch.setattr(run, "save", fail)
        with pytest.raises(OSError):
            run.recover("mcts")
    path = run.out_dir / "outbox/mcts/000003.json"
    value = json.loads(path.read_bytes())
    value["record"]["joined"] = {"changed": True}
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="diagnostic event step:3 differs"):
        run.recover("mcts")
    assert path.exists()
