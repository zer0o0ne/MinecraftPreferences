"""Payload retention follows live references and published checkpoints, across epochs."""

import json
import shutil

import pytest
import torch

from mcpref.infrastructure.shards.store import complete_shards, shard_path, write_shard
from mcpref.training.checkpoints import prune
from mcpref.training.pretrain.continuous import archive_gc
from mcpref.training.pretrain.continuous.archive import Archive


def archive(root, epoch=1):
    return Archive(root / "archive" / f"epoch-{epoch}" / "shards", 1, 1)


def checkpoint(root, store, k, epoch=1, legacy=False):
    state = store.snapshot()
    with write_shard(root / "checkpoints", k) as staged:
        torch.save(
            {"position": {"epoch": epoch}, "archive": state}, staged / "checkpoint.pt"
        )
        header = {} if legacy else {"epoch": epoch, "archive": state}
        (staged / "metadata.json").write_text(json.dumps(header))
    return state


def retired(store, key):
    store.put(key, torch.tensor([[float(key)]]), 1)
    store.consume(f"edge-{key}", key)


def payloads(store):
    return {int(p.parent.name) for p in store.root.glob("*/histories.pt")}


@pytest.mark.parametrize("legacy", [False, True])
def test_live_and_checkpoint_histories_survive_recent_floor(tmp_path, legacy):
    store = archive(tmp_path)
    try:
        store.put("old", torch.tensor([[13.0]]), 1)
        previous = checkpoint(tmp_path, store, 0, legacy=legacy)
        store.consume("edge-old", "old")
        store.put("live", torch.tensor([[7.0]]), 1)
        for key in ("2", "3", "4"):
            retired(store, key)
        checkpoint(tmp_path, store, 1)
        archive_gc.collect(tmp_path, store, 2)
        assert payloads(store) == {0, 1, 3, 4}
        assert complete_shards(store.root) == 5
        copied = tmp_path / "recovered"
        shutil.copytree(store.root.parent, copied)
        recovered = Archive(copied / "shards", 1, 1)
        try:
            recovered.restore(previous)
            assert recovered.get("old", torch.zeros(1, 1)).item() == 13.0
        finally:
            recovered.close()
        prune(tmp_path / "checkpoints", 1, 1)
        archive_gc.collect(tmp_path, store, 2)
        assert payloads(store) == {1, 3, 4}
        assert not (store.root.parent / "metadata/000000/state.sqlite").exists()
    finally:
        store.close()


def test_unused_payloads_are_collected_and_append_and_restore_still_work(tmp_path):
    store = archive(tmp_path)
    try:
        for key in ("0", "1", "2", "3"):
            retired(store, key)
        checkpoint(tmp_path, store, 0)
        archive_gc.collect(tmp_path, store, 2)
        assert payloads(store) == {2, 3}
        assert not store.live
        store.put("new", torch.tensor([[19.0]]), 1)
        state = checkpoint(tmp_path, store, 1)
        archive_gc.collect(tmp_path, store, 2)
        assert payloads(store) == {3, 4}
        store.restore(state)
        assert store.get("new", torch.zeros(1, 1)).item() == 19.0
        assert complete_shards(store.root) == 5
    finally:
        store.close()


def test_old_epoch_freed_only_after_its_last_checkpoint_is_pruned(tmp_path):
    old = archive(tmp_path)
    old.put("old", torch.tensor([[11.0]]), 1)
    previous = checkpoint(tmp_path, old, 0)
    old.consume("edge-old", "old")
    old.close()
    current = archive(tmp_path, 2)
    try:
        checkpoint(tmp_path, current, 1, epoch=2)
        archive_gc.collect(tmp_path, current, 2)
        assert payloads(old) == {0}
        recovered = archive(tmp_path)
        try:
            recovered.restore(previous)
            assert recovered.get("old", torch.zeros(1, 1)).item() == 11.0
        finally:
            recovered.close()
        prune(tmp_path / "checkpoints", 1, 1)
        archive_gc.collect(tmp_path, current, 2)
        assert not payloads(old)
        assert not list(old.root.parent.glob("metadata/*/state.sqlite"))
        assert not (old.root.parent / "working.sqlite").exists()
        assert complete_shards(old.root) == 1
    finally:
        current.close()


@pytest.mark.parametrize("damage", ["checksum", "snapshot", "payload"])
def test_invalid_protected_artifact_aborts_before_any_deletion(tmp_path, damage):
    store = archive(tmp_path)
    try:
        store.put("old", torch.ones(1, 1), 1)
        checkpoint(tmp_path, store, 0)
        store.consume("edge-old", "old")
        for key in ("1", "2"):
            retired(store, key)
        checkpoint(tmp_path, store, 1)
        path = store.root.parent / "metadata/000000/state.sqlite"
        if damage == "checksum":
            path.write_bytes(path.read_bytes() + b"changed")
        else:
            if damage == "payload":
                path = shard_path(store.root, 0) / "histories.pt"
            path.unlink()
        before = payloads(store)
        with pytest.raises((RuntimeError, FileNotFoundError)):
            archive_gc.collect(tmp_path, store, 1)
        assert payloads(store) == before
        assert (store.root.parent / "metadata/000001/state.sqlite").exists()
    finally:
        store.close()


def test_interrupted_gc_is_idempotent_and_preserves_checkpoint(tmp_path, monkeypatch):
    store = archive(tmp_path)
    try:
        for key in ("0", "1", "2"):
            retired(store, key)
        store.put("live", torch.tensor([[23.0]]), 1)
        state = checkpoint(tmp_path, store, 0)
        with monkeypatch.context() as patch:

            def stop(_):
                raise InterruptedError("after first unlink")

            patch.setattr(archive_gc, "sync", stop)
            with pytest.raises(InterruptedError):
                archive_gc.collect(tmp_path, store, 1)
        store.restore(state)
        archive_gc.collect(tmp_path, store, 1)
        assert payloads(store) == {3}
        assert store.get("live", torch.zeros(1, 1)).item() == 23.0
    finally:
        store.close()


def test_restore_rejects_missing_live_payload(tmp_path):
    store = archive(tmp_path)
    try:
        store.put("live", torch.ones(1, 1), 1)
        state = store.snapshot()
        (shard_path(store.root, 0) / "histories.pt").unlink()
        with pytest.raises(RuntimeError, match="missing latent archive payload"):
            store.restore(state)
    finally:
        store.close()
