"""Frozen observation-only inputs of block 33 producer hooks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PolicyObservations:
    logits: Tensor
    targets: Tensor
    masks: Tensor
    previous: Tensor
    actions: Tensor
    prior: Tensor
    width: int
    copy_eps: float
    calibration_bins: int


@dataclass(frozen=True)
class PpoObservations:
    ratios: Tensor
    advantages: Tensor
    behavior: Tensor
    policy: Tensor
    actions: Tensor
    clip: float
    kinds: tuple[str, ...]
    eligible: Tensor


@dataclass(frozen=True)
class ShiftObservations:
    predicted: Tensor
    actual: Tensor
    alpha: Tensor
    distance: Tensor
    constraint: Tensor
    dual: Tensor
    slack: float
    zero_skill: Tensor


@dataclass(frozen=True)
class UpdateObservations:
    gradients: Mapping[str, Tensor]
    updates: Mapping[str, Tensor | None]
    groups: Mapping[str, tuple[str, ...]]
    clip_coefficient: float | None
    learning_rate: float
    schedule_position: int
    logits: Tensor
    supported_actions: Tensor


@dataclass(frozen=True)
class MemoryEvent:
    kind: str
    real_states: int
    seeds: int
    temporal: int
    nearest: int
    beam: int
    candidate_occurrences: int
    unique_comparisons: int
    repeated_comparisons: int
    vertices: int
    capacity: int
    evictions: int
    unused: int
    empty_candidates: int
    episodes: tuple[str, ...]
    covered_episodes: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeSummary:
    episode: str
    first_times: Mapping[str, int | None]
    censored_at: int


@dataclass(frozen=True)
class EpisodeArtifacts:
    episode: str
    inventories: Sequence[Mapping[str, int]]
    previous: Sequence[EpisodeSummary]
