"""Eq. (6): restart-separated differentiable raw histories and eval adapters."""

import re
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import Tensor

from mcpref import REPO
from mcpref.mcts.tree import LatentHistory
from mcpref.perception.config import PerceptionConfig
from mcpref.perception.meta import meta_text
from mcpref.perception.phi import Phi
from mcpref.rl_pairs.types import Encoder, StateRef
from mcpref.sampling.config import SamplingConfig
from mcpref.sampling.history import append
from mcpref.sampling.shards import RawHistory, read_history


def encode_raw(
    phi: Phi,
    perception: PerceptionConfig,
    raw: RawHistory,
    states: int,
    action_count: int,
) -> LatentHistory:
    """Encode independent stream segments, then append preceding-action rows."""
    boundaries = [0]
    boundaries += [
        i for i in range(1, len(raw.states)) if (raw.restart + i) % states == 0
    ]
    boundaries.append(len(raw.states))
    latents = []
    for start, stop in pairwise(boundaries):
        segment = raw.states[start:stop]
        first = segment[0].line
        frames = [observation.frame for observation in segment]
        metas = [
            meta_text(perception, observation.line, first) for observation in segment
        ]
        latents.append(phi.frames_z(frames, metas))
    z = torch.cat(latents, dim=0)
    history = None
    for row, action in zip(z, raw.actions):
        history = append(history, row, action, action_count, states)
    if history is None:
        raise ValueError("a raw history must contain at least one state")
    return history


def encode_ref(
    ref: StateRef,
    phi: Phi,
    perception: PerceptionConfig,
    sampling: SamplingConfig,
    states: int,
    action_count: int,
) -> LatentHistory:
    """Validate a per-cycle episode reference and rebuild its full raw context."""
    match = re.fullmatch(re.escape(sampling.out_dir) + r"#([0-9]{6,})", ref.episode)
    if match is None:
        raise ValueError(f"invalid sampling episode {ref.episode!r}")
    raw = read_history(REPO / sampling.out_dir, ref.index, states)
    return encode_raw(phi, perception, raw, states, action_count)


def eval_encoder(
    phi: Phi,
    perception: PerceptionConfig,
    sampling: SamplingConfig,
    states: int,
    action_count: int,
) -> Encoder:
    """Return batch-1 Y values in reference order, restoring phi's caller mode."""

    def encode(refs: Sequence[StateRef]) -> Sequence[Tensor]:
        training = phi.training
        phi.eval()
        try:
            with torch.no_grad():
                return [
                    encode_ref(ref, phi, perception, sampling, states, action_count).Y
                    for ref in refs
                ]
        finally:
            phi.train(training)

    return encode
