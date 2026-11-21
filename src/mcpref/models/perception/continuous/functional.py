"""Observation-local attention and immutable GDN state; continuous spec, Stream."""

from functools import partial
from types import SimpleNamespace

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from mcpref.models.perception.continuous.state import GDNState


class _StateCache:
    """A single decoder invocation owns its writes, including checkpoint replay."""

    def __init__(self, index: int, conv: Tensor | None, recurrent: Tensor | None):
        self.previous = recurrent is not None
        self.layer = SimpleNamespace(
            conv_states={0: conv}, recurrent_states={0: recurrent}, record_past=True
        )
        self.layers = {index: self.layer}

    def has_previous_state(self, layer_idx: int, state_idx: int) -> bool:
        return self.previous

    def update_conv_state(
        self, values: Tensor, layer_idx: int, *, conv_kernel_size: int
    ) -> Tensor:
        previous = self.layer.conv_states[0]
        full = values if previous is None else torch.cat((previous, values), dim=-1)
        if full.shape[-1] < conv_kernel_size:
            full = F.pad(full, (conv_kernel_size - full.shape[-1], 0))
        self.layer.conv_states[0] = full[..., -conv_kernel_size:].clone()
        return full

    def update_recurrent_state(self, values: Tensor, layer_idx: int) -> Tensor:
        self.layer.recurrent_states[0] = values
        return values


def _layer(layer, hidden, conv, recurrent, *, rotary, mask):
    cache = None
    if layer.block_type == "linear_attention":
        cache = _StateCache(layer.linear_attn.layer_idx, conv, recurrent)
    hidden = layer.forward(
        hidden,
        position_embeddings=rotary,
        attention_mask=mask,
        past_key_values=cache,
    )
    if cache is None:
        return hidden, None, None
    return hidden, cache.layer.conv_states[0], cache.layer.recurrent_states[0]


def text_step(
    model: Qwen3_5TextModel,
    embeddings: Tensor,
    positions: Tensor,
    states: tuple[GDNState, ...],
) -> tuple[Tensor, tuple[GDNState, ...]]:
    """A whole observation; no previous full-attention key/value is supplied."""
    count = model.config.layer_types.count("linear_attention")
    if states and len(states) != count:
        raise ValueError("Continuous state does not match the model's GDN layers")
    rotary = model.rotary_emb(embeddings, positions)
    mask = create_causal_mask(
        config=model.config,
        inputs_embeds=embeddings,
        attention_mask=None,
        past_key_values=None,
    )
    hidden, updated, previous = embeddings, [], iter(states)
    for layer in model.layers:
        state = next(previous, None) if layer.block_type == "linear_attention" else None
        args = (
            (hidden, None, None)
            if state is None
            else (hidden, state.conv, state.recurrent)
        )
        call = partial(
            _layer,
            layer,
            rotary=rotary,
            mask=mask if layer.block_type == "full_attention" else None,
        )
        if layer.training and layer.gradient_checkpointing and torch.is_grad_enabled():
            hidden, conv, recurrent = checkpoint(call, *args, use_reentrant=False)
        else:
            hidden, conv, recurrent = call(*args)
        if conv is not None:
            updated.append(GDNState(conv, recurrent))
    return model.norm(hidden), tuple(updated)
