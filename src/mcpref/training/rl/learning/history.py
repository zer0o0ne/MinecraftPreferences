"""Eq. (6): restart-separated differentiable raw histories and eval adapters."""

import re
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import Tensor

from mcpref.config import REPO
from mcpref.models.perception.config import PerceptionConfig
from mcpref.models.perception.meta import meta_text
from mcpref.models.perception.phi import Phi
from mcpref.runtime.mcts.tree import LatentHistory
from mcpref.runtime.sampling.config import SamplingConfig
from mcpref.runtime.sampling.shards import RawHistory, read_history
from mcpref.training.rl.pairs.types import Encoder, StateRef


def encode_raw(
    phi: Phi,
    perception: PerceptionConfig,
    raw: RawHistory,
    states: int,
    action_count: int,
) -> LatentHistory:
    """Encode independent stream segments and construct the final action-history tail."""
    latents = [
        phi.frames_z(frames, metas)
        for frames, metas in raw_segments(raw, perception, states)
    ]
    return latent_history(raw, torch.cat(latents, dim=0), states, action_count)


def raw_segments(raw: RawHistory, perception: PerceptionConfig, states: int) -> list:
    """Keep stream restart boundaries and each segment's own meta origin."""
    boundaries = [0]
    boundaries += [
        i for i in range(1, len(raw.states)) if (raw.restart + i) % states == 0
    ]
    boundaries.append(len(raw.states))
    segments = []
    for start, stop in pairwise(boundaries):
        segment = raw.states[start:stop]
        first = segment[0].line
        frames = [observation.frame for observation in segment]
        metas = [
            meta_text(perception, observation.line, first) for observation in segment
        ]
        segments.append((frames, metas))
    return segments


def latent_history(
    raw: RawHistory, z: Tensor, states: int, action_count: int
) -> LatentHistory:
    """Retain the action-aligned tail after joining independent stream segments."""
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
    raw = read_ref(ref, sampling, states)
    return encode_raw(phi, perception, raw, states, action_count)


def read_ref(ref: StateRef, sampling: SamplingConfig, states: int) -> RawHistory:
    match = re.fullmatch(re.escape(sampling.out_dir) + r"#([0-9]{6,})", ref.episode)
    if match is None:
        raise ValueError(f"invalid sampling episode {ref.episode!r}")
    return read_history(REPO / sampling.out_dir, ref.index, states)


def encode_refs(
    refs: Sequence[StateRef],
    phi: Phi,
    perception: PerceptionConfig,
    sampling: SamplingConfig,
    states: int,
    action_count: int,
) -> list[LatentHistory]:
    """Batch histories across refs, with separate rows for every stream restart."""
    raws = [read_ref(ref, sampling, states) for ref in refs]
    segments = [raw_segments(raw, perception, states) for raw in raws]
    latents = [[] for _ in refs]
    for part in range(max(map(len, segments))):
        owners = [i for i, parts in enumerate(segments) if part < len(parts)]
        encoded = phi.batch_frames_z([segments[i][part] for i in owners])
        for i, z in zip(owners, encoded):
            latents[i].append(z)
    return [
        latent_history(raw, torch.cat(z, dim=0), states, action_count)
        for raw, z in zip(raws, latents)
    ]


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
