"""Devices of the training loops (specs/gui-stack.md, specs/pretrain.md): CPU copies
of tensors, the config's device."""

from collections.abc import Sequence

import torch


def cpu_copies(params: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Interface checkpoints, model.pt: detached copies of the parameters on the CPU."""
    return [p.detach().to("cpu", copy=True) for p in params]


def device(value: object) -> torch.device:
    """Notation: a string accepted by torch.device."""
    if isinstance(value, str):
        try:
            return torch.device(value)
        except RuntimeError as error:
            raise ValueError(f"device {value!r}: {error}") from error
    raise ValueError(f"device {value!r} is not a string")
