"""Typed producer context and pending/accepted metric record schema."""

from dataclasses import asdict, dataclass
from typing import TypedDict


class MetricRecord(TypedDict):
    version: int
    kind: str
    index: int | None
    id: str
    context: dict
    populations: dict[str, list[str]]
    metrics: dict
    joined: dict


@dataclass(frozen=True)
class ReferencePair:
    micro_batch: int
    pair_in_micro: int
    a: str
    b: str
    labels: dict
    baseline: float
    only_phi_post: float
    joint_post: float
    delta_only_phi: float
    delta_joint: float


@dataclass(frozen=True)
class ProducerContext:
    producer: str
    split: str
    condition: str
    model: str
    source: str
    filters: dict
    steps: tuple[int, ...]
    cluster_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]


def metric_record(
    kind: str,
    event_id: str,
    context: ProducerContext,
    metrics: dict,
    populations: dict[str, list[str]],
) -> MetricRecord:
    return {
        "version": 1,
        "kind": kind,
        "index": None,
        "id": event_id,
        "context": asdict(context),
        "populations": populations,
        "metrics": metrics,
        "joined": {},
    }
