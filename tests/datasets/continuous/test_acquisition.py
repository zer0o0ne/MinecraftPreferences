"""Concurrent source acquisition preserves ordered, bounded publication."""

import json
from concurrent.futures import Future
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace as NS

import pytest

from mcpref.datasets.continuous.acquisition import queue, space


def piece(index):
    return {
        "piece_id": str(index),
        "mp4": {"bytes": 1},
        "jsonl": {"bytes": 1},
        "nominal_steps": 1,
    }


def setup(monkeypatch, tmp_path, acquire, free=1000, **overrides):
    monkeypatch.setattr(queue, "Transport", lambda *args: nullcontext(object()))
    monkeypatch.setattr(queue, "acquire", acquire)
    monkeypatch.setattr(space.shutil, "disk_usage", lambda path: NS(free=free))
    settings = NS(
        workers=2, prefetch_sources=2, prefetch_bytes=6, processing_reserve_bytes=20
    )
    settings.__dict__.update(overrides)
    cfg = NS(out_dir=str(tmp_path), disk_reserve_bytes=10)
    return NS(cfg=NS(required_options={})), cfg, settings


def test_download_overlaps_verification_and_publication_is_fifo(monkeypatch, tmp_path):
    from mcpref.datasets.continuous import run

    downloading, verified = Event(), Event()
    order = []

    def acquire(pipe, cfg, source, transport):
        path = tmp_path / source["piece_id"]
        path.mkdir()
        (path / "options.json").write_text("{}")
        if source["piece_id"] == "1":
            downloading.set()
            assert verified.wait(2)
        return path, {"files": {}}

    def verify(pipe, path, rows, fingerprint):
        if path.name == "0":
            assert downloading.wait(2), "Next source must download during verification"
            verified.set()
        order.append(path.name)
        return {"reason": None}

    monkeypatch.setattr(run, "verify", verify)
    monkeypatch.setattr(run, "options_ok", lambda *args: True)
    args = setup(monkeypatch, tmp_path, acquire)
    pieces = [piece(0), piece(1)]
    rows = NS(append_source=lambda *args: 0)
    with queue.Acquisition(*args, [(0, {}, pieces)], 0) as downloads:
        result = run.prepare(args[0], downloads, pieces, rows, {})
    assert order == ["0", "1"]
    assert [s["piece"] for s in result] == pieces


def test_large_fragment_streams_through_small_queue_and_skips_completed(
    monkeypatch, tmp_path
):
    called = []

    def acquire(pipe, cfg, source, transport):
        called.append(source["piece_id"])
        return source["piece_id"]

    args = setup(monkeypatch, tmp_path, acquire)
    pieces = [piece(i) for i in range(6)]
    plan = [(0, {}, [piece(99)]), (1, {}, pieces)]
    with queue.Acquisition(*args, plan, 1) as downloads:
        downloads.fill()
        assert downloads.queued_bytes == 6
        assert len(downloads.pending) == 2
        for source in pieces:
            assert downloads.get(source) == source["piece_id"]
            assert downloads.queued_bytes <= 6
            assert len(downloads.pending) <= 2
    assert sorted(called) == [str(i) for i in range(6)]


def test_prefetch_yields_disk_workspace_but_required_source_can_progress(
    monkeypatch, tmp_path
):
    calls = []
    args = setup(
        monkeypatch, tmp_path, lambda p, c, s, t: calls.append(s["piece_id"]), free=14
    )
    with queue.Acquisition(*args, [(0, {}, [piece(0), piece(1)])], 0) as downloads:
        downloads.fill()
        assert not downloads.pending
        downloads.get(piece(0))
        assert not downloads.pending
    assert calls == ["0"]


@pytest.mark.parametrize(
    "free,budget,match", [(12, 6, "Disk reserve"), (1000, 2, "prefetch_bytes")]
)
def test_impossible_source_fails_without_submitting(
    monkeypatch, tmp_path, free, budget, match
):
    calls = []
    args = setup(
        monkeypatch,
        tmp_path,
        lambda *args: calls.append(args),
        free=free,
        prefetch_bytes=budget,
    )
    with (
        queue.Acquisition(*args, [(0, {}, [piece(0)])], 0) as downloads,
        pytest.raises((RuntimeError, ValueError), match=match),
    ):
        downloads.get(piece(0))
    assert not calls


def test_failure_waits_for_writers_and_does_not_return_later_sources(
    monkeypatch, tmp_path
):
    second = Event()
    finished = []

    def acquire(pipe, cfg, source, transport):
        if source["piece_id"] == "0":
            assert second.wait(2)
            raise ValueError("bad source")
        finished.append(source["piece_id"])
        second.set()

    args = setup(monkeypatch, tmp_path, acquire)
    with (
        pytest.raises(ValueError, match="bad source"),
        queue.Acquisition(*args, [(0, {}, [piece(0), piece(1)])], 0) as downloads,
    ):
        downloads.get(piece(0))
    assert finished == ["1"]


def test_publisher_counts_inflight_writes_but_not_finished_files(monkeypatch, tmp_path):
    monkeypatch.setattr(space.shutil, "disk_usage", lambda path: NS(free=35))
    budget = space.Space(tmp_path, 10)
    job = Future()
    budget.track(job, 20)
    with pytest.raises(RuntimeError, match="Disk reserve"):
        budget.check(6)
    job.set_result(None)
    budget.check(25)
    assert budget.pending == 0


def test_transfer_config_validates_its_own_limits(tmp_path):
    from mcpref.datasets.continuous.acquisition.settings import load_transfer

    values = {
        "workers": 2,
        "prefetch_sources": 3,
        "prefetch_bytes": 100,
        "processing_reserve_bytes": 20,
        "connections": 4,
        "file_connections": 2,
        "block_bytes": 8,
        "read_bytes": 4,
        "block_timeout": 30,
        "keepalive_expiry": 60,
    }
    path = tmp_path / "transfer.yaml"
    path.write_text(json.dumps(values))
    assert load_transfer(path).workers == 2
    for key, value in [
        ("workers", 4),
        ("file_connections", 5),
        ("read_bytes", 9),
        ("block_timeout", 0),
    ]:
        path.write_text(json.dumps(values | {key: value}))
        with pytest.raises(ValueError, match=key):
            load_transfer(path)


def test_completion_during_disk_snapshot_keeps_its_reservation(monkeypatch, tmp_path):
    job = Future()
    budget = space.Space(tmp_path, 10)
    budget.track(job, 20)

    def snapshot_then_finish_write(path):
        before_write = NS(free=35)
        job.set_result(None)
        return before_write

    monkeypatch.setattr(space.shutil, "disk_usage", snapshot_then_finish_write)
    assert not budget.available(20)
