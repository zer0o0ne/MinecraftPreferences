"""Archive snapshots recover together with checkpoints and keep metadata on disk."""

import pytest
import torch

from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.training.pretrain.continuous.archive import Archive

EMPTY = {"next_shard": 0, "live": {}, "seen": [], "consumed": []}


def test_restore_discards_uncheckpointed_appends_and_preserves_checkpoint_histories(
    tmp_path,
):
    root = tmp_path / "archive/shards"
    archive = Archive(root, 1, 1)
    archive.restore(EMPTY)
    a = torch.tensor([[1.0, 2.0]], requires_grad=True)
    archive.put("a", a, 2)
    saved = archive.snapshot()
    referenced = (shard_path(root, 0) / "histories.pt").read_bytes()
    archive.put("b", torch.ones(1, 2), 1)
    archive.consume("edge-a-b", "a")
    archive.close()

    restored = Archive(root, 1, 1)
    try:
        restored.restore(saved)
        assert complete_shards(root) == 1
        assert (shard_path(root, 0) / "histories.pt").read_bytes() == referenced
        assert set(restored.seen) == {"a"}
        assert not restored.consumed and "b" not in restored.live
        assert restored.live["a"]["remaining"] == 2
        value = restored.get("a", a)
        assert torch.equal(value, a) and not value.requires_grad
        restored.consume("edge-a-b", "a")
        restored.consume("edge-a-c", "a")
        assert not restored.live
        assert set(restored.consumed) == {"edge-a-b", "edge-a-c"}
    finally:
        restored.close()


def test_published_archive_snapshot_without_checkpoint_is_rolled_back(tmp_path):
    archive = Archive(tmp_path / "shards", 2, 1)
    try:
        archive.restore(EMPTY)
        archive.put("first", torch.tensor([[3.0]]), 1)
        committed = archive.snapshot()
        archive.consume("later-edge", "first")
        archive.put("later", torch.tensor([[7.0]]), 1)
        uncommitted = archive.snapshot()
        assert uncommitted["metadata"]["index"] == 1
        archive.restore(committed)
        assert complete_shards(tmp_path / "metadata") == 1
        assert complete_shards(tmp_path / "shards") == 1
        assert set(archive.live) == {"first"}
        assert not archive.consumed
        assert archive.get("first", torch.zeros(1, 1)).item() == 3
    finally:
        archive.close()


def test_retirement_is_exact_and_membership_metadata_does_not_accumulate_in_python(
    tmp_path,
):
    archive = Archive(tmp_path / "shards", 3, 2)
    try:
        archive.restore(EMPTY)
        for i in range(400):
            archive.put(str(i), torch.ones(1, 1), 1)
            archive.consume(f"edge-{i}", str(i))
        assert not archive.live and not archive.buffer and not archive.cache
        assert len(archive.seen) == len(archive.consumed) == 400
        assert not isinstance(archive.seen, (set, dict, list))
        assert not isinstance(archive.consumed, (set, dict, list))
        state = archive.snapshot()
        assert set(state) == {"next_shard", "metadata"}
        assert state["next_shard"] == 0
        archive.restore(state)
        assert len(archive.consumed) == 400
        archive.put("remaining", torch.ones(1, 1), 2)
        with pytest.raises(RuntimeError, match="consumed twice"):
            archive.consume("edge-0", "remaining")
        assert archive.live["remaining"]["remaining"] == 2
    finally:
        archive.close()


def test_snapshots_are_immutable(tmp_path):
    archive = Archive(tmp_path / "shards", 2, 1)
    try:
        archive.restore(EMPTY)
        value = torch.tensor([[1.0]])
        archive.put("a", value, 2)
        value.add_(100)
        states = [archive.snapshot()]
        archive.consume("e1", "a")
        states.append(archive.snapshot())
        archive.consume("e2", "a")
        states.append(archive.snapshot())
        archive.restore(states[1])
        assert archive.live["a"]["remaining"] == 1
        assert archive.get("a", value).item() == 1
        assert archive.consumed == {"e1"}
    finally:
        archive.close()
