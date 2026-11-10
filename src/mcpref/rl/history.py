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
from mcpref.sampling.shards import RawHistory, read_history


def encode_raw(
    phi: Phi,
    perception: PerceptionConfig,
    raw: RawHistory,
    states: int,
    action_count: int,
) -> LatentHistory:
    """Encode independent stream segments and construct the final action-history tail."""
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
    tail = raw.actions[-states:]
    Y = z[-states:].unsqueeze(0)
    C = z.new_zeros((1, len(tail), action_count + 1))
    columns = [action_count if action is None else action for action in tail]
    C[0, list(range(len(tail))), columns] = 1
    return LatentHistory(Y, C)


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
