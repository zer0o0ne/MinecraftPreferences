"""Functional recurrent values from specs/pretrain-continuous.md."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GDNState:
    conv: Tensor
    recurrent: Tensor


@dataclass(frozen=True)
class ContinuousState:
    layers: tuple[GDNState, ...]
    origin: Mapping


def detach_state(
    state: ContinuousState, *, device: torch.device | str | None = None
) -> ContinuousState:
    """Keep stream values across an update, with no gradient to its old window."""
    layers = tuple(
        GDNState(s.conv.detach().to(device), s.recurrent.detach().to(device))
        for s in state.layers
    )
    return ContinuousState(layers, dict(state.origin))
