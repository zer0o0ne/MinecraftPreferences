"""Tensor payload corruption must fail before persisted training state is consumed."""

import struct
import zipfile

import pytest
import torch
from training.pretrain.continuous.diagnostics.hooks_world import setup, training
from training.pretrain.continuous.engine_world import (
    Reader,
    config,
    context,
    modules,
    observation_step,
)
from training.pretrain.continuous.test_resume import world

from mcpref.training.pretrain.continuous import checkpoints, loop, steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.diagnostics.reference_store import References
from mcpref.training.pretrain.continuous.schedule import epoch_updates


def corrupt(path, kind):
    if kind == "missing":
        path.unlink()
        return
    if kind == "truncated":
        with path.open("r+b") as stream:
            stream.truncate(40)
        return
    with zipfile.ZipFile(path) as archive:
        record = next(item for item in archive.infolist() if "/data/" in item.filename)
    with path.open("r+b") as stream:
        stream.seek(record.header_offset + 26)
        name_size, extra_size = struct.unpack("<HH", stream.read(4))
        offset = record.header_offset + 30 + name_size + extra_size
        stream.seek(offset)
        changed = stream.read(1)[0] ^ 1
        stream.seek(offset)
        stream.write(bytes([changed]))
    # torch.load does not validate ZIP tensor-record CRCs itself.
    torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("damage", ["byteflip", "missing", "truncated"])
def test_archive_get_rejects_damaged_histories(tmp_path, damage):
    archive = Archive(tmp_path / "shards", 1, 1)
    try:
        archive.put("a", torch.tensor([[1.0, 2.0]]), 1)
        path = tmp_path / "shards/000000/histories.pt"
        corrupt(path, damage)
        if damage == "byteflip":
            loaded = torch.load(path, weights_only=True)["a"]
            assert loaded[0, 0].item() != 1.0
        with pytest.raises(RuntimeError, match="tensor payload"):
            archive.get("a", torch.zeros(1, 2))
        assert not archive.cache
    finally:
        archive.close()


@pytest.mark.parametrize("damage", ["byteflip", "missing", "truncated"])
def test_restore_rejects_damaged_checkpoint_before_loading_weights(
    tmp_path, monkeypatch, damage
):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg, index = config(epochs=1), world(1)
    ctx = context(tmp_path, cfg, modules(), index, epoch_updates(index, cfg))
    loop.train(ctx)
    path = max(tmp_path.glob("checkpoints/*/checkpoint.pt"))
    corrupt(path, damage)
    before = [
        parameter.clone()
        for module in ctx.modules.values()
        for parameter in module.parameters()
    ]
    with pytest.raises(RuntimeError, match="tensor payload"):
        checkpoints.restore(ctx)
    after = [
        parameter
        for module in ctx.modules.values()
        for parameter in module.parameters()
    ]
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["byteflip", "missing", "truncated"])
def test_reference_reuse_rejects_damaged_initial_tensors(tmp_path, monkeypatch, damage):
    fixture, index = setup(tmp_path, monkeypatch)
    try:
        ctx = training(tmp_path, fixture, index, "reference")
        ctx.step = 0
        References(ctx, None, "unused").start()
        corrupt(ctx.root / "reference/000000/initial.pt", damage)
        reference = References(ctx, None, "unused")
        with pytest.raises(RuntimeError, match="tensor payload"):
            reference.start()
        assert reference.initial is None
    finally:
        index.close()


@pytest.mark.parametrize("damage", ["byteflip", "missing", "truncated"])
def test_model_load_rejects_damaged_export(tmp_path, damage):
    cfg, index = config(epochs=1), world(1)
    ctx = context(tmp_path, cfg, modules(), index, epoch_updates(index, cfg))
    checkpoints.write_model(ctx)
    path = tmp_path / "model.pt"
    corrupt(path, damage)
    with pytest.raises(RuntimeError, match="tensor payload"):
        checkpoints.load_model(path, ctx.modules)
