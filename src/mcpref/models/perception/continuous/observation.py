"""Differentiable and inference observation steps for continuous pretraining."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from mcpref.models.perception.continuous.encoding import (
    Prepared,
    decode,
    encode,
    vision_metadata,
)
from mcpref.models.perception.continuous.state import ContinuousState
from mcpref.models.perception.meta import meta_text
from mcpref.models.perception.phi import Phi
from mcpref.models.perception.tokens import image_inputs, positions, segment


def prepare(phi: Phi, frame: np.ndarray, line: Mapping, origin: Mapping) -> Prepared:
    """Prepare one already composited RGB observation without model outputs."""
    pixels, grid = image_inputs(phi.processor, [frame])
    meta = meta_text(phi.cfg, line, origin)
    ids = segment(phi.tokenizer, phi.sp, grid[0], meta, phi.cfg.marker)
    pos = positions(phi.backbone, ids, grid, phi.sp)
    device = phi.head[0].weight.device
    tokens = torch.tensor([ids], dtype=torch.int64, device=device)
    return Prepared(
        pixels.to(device),
        grid,
        tokens,
        pos.to(device),
        (tokens == phi.sp.image).unsqueeze(-1),
        ids.count(phi.sp.image),
        vision_metadata(phi, grid),
        dict(origin),
    )


def observation_step(
    phi: Phi, frame: np.ndarray, line: Mapping, state: ContinuousState | None = None
) -> tuple[Tensor, ContinuousState]:
    """Read one RGB observation (cursor already drawn) with its fragment's origin."""
    prepared = prepare(phi, frame, line, line if state is None else state.origin)
    return decode(phi, encode(phi, prepared), prepared, state)


class ContinuousPhiStream:
    """Inference with the same observation semantics and explicit fragment resets."""

    def __init__(self, phi: Phi):
        self.phi = phi
        self.state = None

    def restart(self) -> None:
        self.state = None

    @torch.no_grad()
    def step(self, frame: np.ndarray, line: Mapping) -> Tensor:
        z, self.state = observation_step(self.phi, frame, line, self.state)
        return z
