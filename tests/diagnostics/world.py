"""Test-owned inputs and independent scalar oracles for block 33."""

import math
from pathlib import Path

import torch
import yaml

from mcpref.diagnostics.config import DiagnosticsConfig
from mcpref.diagnostics.targets import ProducerContext
from mcpref.pretrain.diagnostics import (
    GlobalFit,
    PreferenceRows,
    ReferencePair,
    Stratum,
)

STARTING = {
    "out_dir": "data/diagnostics",
    "seed": 0,
    "ci_level": 0.95,
    "bootstrap_samples": 20,
    "min_cluster_count": 2,
    "quantiles": [0.05, 0.5, 0.95],
}


def config(**changes) -> DiagnosticsConfig:
    values = STARTING | changes
    values["quantiles"] = tuple(values["quantiles"])
    return DiagnosticsConfig(**values)


def write_config(path: Path, **changes) -> Path:
    values = {
        key: value for key, value in (STARTING | changes).items() if value is not None
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path


def context(
    source="pair_labels/shards/000003/pairs.json",
    producer="acyclic-labels",
    condition="train",
    split="train",
    steps=(),
    clusters=("e0", "e1"),
) -> ProducerContext:
    return ProducerContext(
        producer=producer,
        split=split,
        condition=condition,
        model="test-model",
        source=source,
        filters={},
        steps=steps,
        cluster_ids=tuple(clusters),
        artifact_paths=(source,),
    )


def ref(episode: str, clip: str = "x") -> dict:
    return {"id": f"{clip}-{episode}", "episode": episode}


def vote(episode: str, labels: list[str | None]) -> dict:
    return {
        "a": ref(episode, "a"),
        "b": ref(f"{episode}-b", "b"),
        "requests": [
            {
                "order": order,
                "sample": sample,
                "label": "no_answer" if label is None else label,
            }
            for order in (0, 1)
            for sample, label in enumerate(labels)
        ],
    }


def label(episode: str, p: float | None, inc: int) -> dict:
    return {
        "a": ref(episode, "a"),
        "b": ref(episode, "b"),
        "labels": {"p": p, "inc": inc},
    }


def comparison(episode: str) -> dict:
    return {"a": ref(episode, "a"), "b": ref(episode, "b")}


def fit(inc=0.25, succ=0.5) -> GlobalFit:
    logit = lambda p: None if p is None or p in (0, 1) else math.log(p / (1 - p))
    comparable = 0 if succ is None else 2
    return GlobalFit(
        3,
        3 - comparable,
        comparable,
        None if succ is None else 1.0,
        inc,
        succ,
        logit(inc),
        logit(succ),
        (
            Stratum("0.5", "comparable", 0.5, comparable, 0),
            Stratum("inc", "inc", None, 3 - comparable, 0),
        ),
    )


def rows(s=(0.0, 1.0, 2.0), t=None, p=None, inc=None):
    t = (0.0,) * len(s) if t is None else t
    p = (0.0, 0.5, 1.0) if p is None and len(s) == 3 else p
    p = (0.5,) * len(s) if p is None else p
    inc = (0,) * len(s) if inc is None else inc
    return PreferenceRows(
        torch.tensor(s, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(inc, dtype=torch.uint8),
    )


def pair(delta_phi=-1.0, delta_joint=-2.0) -> ReferencePair:
    return ReferencePair(
        0, 0, "a", "b", {"p": 0.5, "inc": 0}, 3.0, 2.0, 1.0, delta_phi, delta_joint
    )


def bce(x: float, y: float) -> float:
    return max(x, 0.0) - x * y + math.log1p(math.exp(-abs(x)))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    h = 1 + q * (len(values) - 1)
    j = math.floor(h)
    alpha = h - j
    if alpha == 0:
        return ordered[j - 1]
    return (1 - alpha) * ordered[j - 1] + alpha * ordered[j]


def distribution(values: list[float], quantiles=(0.05, 0.5, 0.95)) -> dict:
    n = len(values)
    return {
        "n": n,
        "mean": None if n == 0 else sum(values) / n,
        "median": quantile(values, 0.5),
        "quantiles": {str(q): quantile(values, q) for q in quantiles},
        "positive": None if n == 0 else sum(x > 0 for x in values) / n,
        "negative": None if n == 0 else sum(x < 0 for x in values) / n,
        "zero": None if n == 0 else sum(x == 0 for x in values) / n,
    }
