"""Checkpoint diagnostics own CPU snapshots, never live training tensor storage."""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import pytest
import torch

from mcpref.training.pretrain.continuous.diagnostics import hooks
from mcpref.training.pretrain.continuous.events import WindowResult


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_pending_copies_training_storage_before_decorating(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    storage = torch.arange(16.0, device=device, requires_grad=True)
    result = WindowResult(
        weighted={"phi": storage[2] * 2},
        sums={"phi_total": storage[3]},
        ce_rows=[{"logits": storage[4:8].reshape(1, 4)}],
        phi_rows=[{"z_norm": storage[8:10]}],
        shift_rows=[{"zero_shift": storage[10]}],
        preference_rows=[{"s": storage[11], "t": storage[12]}],
        gradients={"norm": storage[13]},
        updates={"norm": storage[14]},
    )
    pending = hooks.Diagnostics()
    pending.pending, pending.gradients = [], None
    pending.reference = SimpleNamespace(fit=None)

    def decorate(context, snapshot, fit):
        assert snapshot is not result
        assert all(value.device.type == "cpu" for value in tensors(snapshot))
        assert all(not value.requires_grad for value in tensors(snapshot))
        return {"loss": float(snapshot.weighted["phi"])}, snapshot.preference_rows

    monkeypatch.setattr(hooks, "decorate", decorate)
    pending.after_update(SimpleNamespace(step=7), result)
    with torch.no_grad():
        storage.fill_(-1)
        result.weighted["phi"].fill_(-2)
    step, snapshot, record, rows = pending.pending[0]
    assert step == 7 and record == {"loss": 4.0}
    assert snapshot.sums["phi_total"].item() == 3.0
    assert snapshot.ce_rows[0]["logits"].tolist() == [[4.0, 5.0, 6.0, 7.0]]
    assert [value.item() for value in tensors(rows)] == [11.0, 12.0]
    assert snapshot.gradients["norm"].item() == 13.0
    assert snapshot.updates["norm"].item() == 14.0
