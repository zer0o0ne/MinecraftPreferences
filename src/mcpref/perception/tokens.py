"""(3)-(5) of specs/perception.md: the image inputs, the segment of a state and the
M-RoPE positions of a sequence."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from transformers import (
    PretrainedConfig,
    PreTrainedTokenizerBase,
    Qwen2VLImageProcessor,
    Qwen3_5Model,
)


@dataclass(frozen=True)
class Specials:
    """Notation: b, p, e (vision start, image, vision end token ids) and m_s."""

    start: int
    image: int
    end: int
    merge: int


def specials(config: PretrainedConfig) -> Specials:
    """Notation: b, p, e and m_s from B's config."""
    return Specials(
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
        config.vision_config.spatial_merge_size,
    )


def image_inputs(
    processor: Qwen2VLImageProcessor, frames: list[np.ndarray]
) -> tuple[Tensor, Tensor]:
    """(4): pixel_values (the rows of all frames in order) and image_grid_thw."""
    out = processor(frames, input_data_format="channels_last", return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Notation: tau_tok(text)."""
    return tokenizer(text, add_special_tokens=False, split_special_tokens=False)[
        "input_ids"
    ]


def segment(
    tokenizer: PreTrainedTokenizerBase,
    sp: Specials,
    grid: Tensor,
    meta: str,
    marker: str,
) -> list[int]:
    """(3): sigma_s = [b] + [p]^N_s + [e] + tau_tok(mu_s) + tau_tok(omega)."""
    N = int(grid.prod()) // sp.merge**2
    return [
        sp.start,
        *[sp.image] * N,
        sp.end,
        *encode(tokenizer, meta),
        *encode(tokenizer, marker),
    ]


def positions(
    backbone: Qwen3_5Model, ids: list[int], grid: Tensor, sp: Specials
) -> Tensor:
    """(5): Pi of the sequence `ids` with the grids of its frames, shape (3, 1, |ids|)."""
    input_ids = torch.tensor([ids], dtype=torch.int64)
    mm_token_type_ids = (input_ids == sp.image).to(torch.int64)
    return backbone.get_rope_index(
        input_ids=input_ids, mm_token_type_ids=mm_token_type_ids, image_grid_thw=grid
    )[0]
