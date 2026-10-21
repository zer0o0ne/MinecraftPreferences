"""src/mcpref/training/devices.py against specs/pretrain.md (Interface: `cpu_copies`,
`device`, moved unchanged from `stacking/fit.py` and `stacking/config.py`)."""

import pytest
import torch

import mcpref.stacking.config
import mcpref.stacking.fit
from mcpref.training.devices import cpu_copies, device


def test_cpu_copies_are_detached_copies_on_the_cpu():
    """Same values and dtypes, on the CPU, no grad, no storage shared with the parameters."""
    params = [
        torch.nn.Parameter(
            torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.float64)
        ),
        torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float32)),
    ]
    got = cpu_copies(params)
    assert type(got) is list and len(got) == 2
    for p, c in zip(params, got):
        assert c.device == torch.device("cpu") and c.dtype == p.dtype
        assert not c.requires_grad and torch.equal(c, p.detach())
    with torch.no_grad():
        params[0].add_(1.0)
    assert torch.equal(
        got[0], torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize("text", ["cpu", "cuda", "cuda:1", "meta"])
def test_device_accepts_what_torch_device_accepts(text):
    """A string torch.device accepts gives that device; no GPU needed."""
    assert device(text) == torch.device(text)


@pytest.mark.parametrize("value", ["gpu", "", "cuda:x", 0, None, ["cpu"]])
def test_device_rejects_others(value):
    """Not a string, or a string torch.device rejects: ValueError."""
    with pytest.raises(ValueError):
        device(value)


@pytest.mark.parametrize(
    "module, name, moved",
    [
        (mcpref.stacking.fit, "cpu_copies", cpu_copies),
        (mcpref.stacking.config, "device", device),
    ],
)
def test_moved_from_gui_stack(module, name, moved):
    """The function lives in training/devices.py; its old module keeps no copy of it."""
    assert moved.__module__ == "mcpref.training.devices"
    assert getattr(module, name, moved) is moved
