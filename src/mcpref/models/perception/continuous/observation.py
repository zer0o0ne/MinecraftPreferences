"""Differentiable and inference observation steps for continuous pretraining."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.state import ContinuousState
from mcpref.models.perception.meta import meta_text
from mcpref.models.perception.phi import Phi
from mcpref.models.perception.tokens import image_inputs, positions, segment


def observation_step(
    phi: Phi, frame: np.ndarray, line: Mapping, state: ContinuousState | None = None
) -> tuple[Tensor, ContinuousState]:
    """Read one RGB observation (cursor already drawn) with its fragment's origin."""
    origin = dict(line) if state is None else state.origin
    pixels, grid = image_inputs(phi.processor, [frame])
    meta = meta_text(phi.cfg, line, origin)
    ids = segment(phi.tokenizer, phi.sp, grid[0], meta, phi.cfg.marker)
    pos = positions(phi.backbone, ids, grid, phi.sp)
    device = phi.head[0].weight.device
    tokens = torch.tensor([ids], dtype=torch.int64, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        backbone = phi.backbone
        embeds = backbone.get_input_embeddings()(tokens)
        features = backbone.get_image_features(
            pixels.to(device), grid.to(device), return_dict=True
        )
        images = torch.cat(features.pooler_output, dim=0).to(embeds)
        image_mask, _ = backbone.get_placeholder_mask(
            tokens, inputs_embeds=embeds, image_features=images
        )
        embeds = embeds.masked_scatter(image_mask, images)
        hidden, updated = text_step(
            backbone.language_model,
            embeds,
            pos.to(device),
            () if state is None else state.layers,
        )
        z = phi.head(hidden[0, -1])
    return z, ContinuousState(updated, origin)


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
