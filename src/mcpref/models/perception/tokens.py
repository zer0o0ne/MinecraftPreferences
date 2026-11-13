"""(3)-(5) of specs/perception.md: the image inputs, the segment of a state and the
M-RoPE positions of a sequence."""

from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import torch
from torch import Tensor
from transformers import (
    PretrainedConfig,
    PreTrainedTokenizerBase,
    Qwen2VLImageProcessor,
    Qwen3_5Model,
)

from mcpref.infrastructure.profiling.scopes import measured, qwen_batch


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


@measured("qwen.preprocess")
def batch_inputs(
    backbone: Qwen3_5Model,
    tokenizer: PreTrainedTokenizerBase,
    processor: Qwen2VLImageProcessor,
    sp: Specials,
    states: list[tuple[list[np.ndarray], list[str]]],
    marker: str,
) -> tuple[dict[str, Tensor], list[list[int]]]:
    """Right-pad independent sequences; preserve each clip's M-RoPE and image order."""
    ids, pixels, grids, pos, readouts = [], [], [], [], []
    for frames, metas in states:
        pixel, grid = image_inputs(processor, frames)
        segments = [segment(tokenizer, sp, g, mu, marker) for g, mu in zip(grid, metas)]
        sequence = [token for part in segments for token in part]
        ids.append(torch.tensor(sequence, dtype=torch.int64))
        pixels.append(pixel)
        grids.append(grid)
        pos.append(positions(backbone, sequence, grid, sp))
        readouts.append([n - 1 for n in accumulate(map(len, segments))])
    qwen_batch([len(row) for row in ids], sum(len(frames) for frames, _ in states))
    lengths = torch.tensor([len(row) for row in ids])
    padded = torch.nn.utils.rnn.pad_sequence(
        ids, batch_first=True, padding_value=sp.end
    )
    mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "pixel_values": torch.cat(pixels),
        "image_grid_thw": torch.cat(grids),
        "position_ids": torch.cat(
            [
                torch.nn.functional.pad(p, (0, padded.shape[1] - p.shape[-1]))
                for p in pos
            ],
            dim=1,
        ),
    }, readouts
