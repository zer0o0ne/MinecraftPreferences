"""Reusable observation inputs and differentiable stages of continuous phi."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.state import ContinuousState
from mcpref.models.perception.phi import Phi


@dataclass(frozen=True)
class Prepared:
    pixels: Tensor
    grid: Tensor
    tokens: Tensor
    positions: Tensor
    image_mask: Tensor
    image_count: int
    vision: dict[str, Tensor | int | None]
    origin: Mapping


def vision_metadata(phi: Phi, grid: Tensor) -> dict[str, Tensor | int | None]:
    """Compute weight-independent metadata on CPU; SDPA lengths stay on CPU."""
    visual = phi.backbone.visual
    device = phi.head[0].weight.device
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.spatial_merge_size,
        padding="border",
    )
    position_ids = get_vision_position_ids(
        grid, visual.spatial_merge_size, include_temporal=False
    )
    cu_seqlens, max_seqlen = get_vision_attention_seqlens(
        grid, visual.config, merge_temporal=False
    )
    return {
        "image_interp_indices": indices.to(device),
        "image_interp_weights": weights.to(device),
        "image_position_ids": position_ids.to(device),
        "image_cu_seqlens": cu_seqlens,
        "image_max_seqlen": max_seqlen,
    }


def encode(phi: Phi, prepared: Prepared, embeddings: Tensor | None = None) -> Tensor:
    """Vision features and token embeddings, with their parameter gradients."""
    with torch.autocast(prepared.tokens.device.type, dtype=torch.bfloat16):
        backbone = phi.backbone
        embeds = (
            backbone.get_input_embeddings()(prepared.tokens)
            if embeddings is None
            else embeddings
        )
        features = backbone.get_image_features(
            prepared.pixels, prepared.grid, return_dict=True, **prepared.vision
        )
        images = torch.cat(features.pooler_output, dim=0).to(embeds)
        if prepared.image_count * embeds.shape[-1] != images.numel():
            raise ValueError(
                "Image features and image tokens do not match, "
                f"tokens: {prepared.image_count}, features: {images.shape[0]}"
            )
        return embeds.masked_scatter(prepared.image_mask, images)


def decode(
    phi: Phi,
    embeddings: Tensor,
    prepared: Prepared,
    state: ContinuousState | None = None,
) -> tuple[Tensor, ContinuousState]:
    """Observation-local text attention and continuing GDN memory."""
    with torch.autocast(embeddings.device.type, dtype=torch.bfloat16):
        hidden, updated = text_step(
            phi.backbone.language_model,
            embeddings,
            prepared.positions,
            () if state is None else state.layers,
        )
        z = phi.head(hidden[0, -1])
    return z, ContinuousState(updated, prepared.origin)
