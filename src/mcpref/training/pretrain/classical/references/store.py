"""Atomic fixed-reference metric and typed-pair shard writers."""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from mcpref.diagnostics.core.artifacts import write_pairs
from mcpref.diagnostics.core.types import ReferencePair
from mcpref.diagnostics.preference.suites import GlobalFit
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import shard_path, write_shard

GLOBAL_FIT = "global-fit.json"


def write_initial(
    directory: Path,
    selection: dict,
    baselines: GlobalFit,
    initial: dict,
    metrics: dict,
    pairs: Sequence[ReferencePair],
) -> None:
    """(18): atomically publish the new pairs-bearing shard-0 format."""
    shared = directory.parent / GLOBAL_FIT
    baseline_bytes = (
        shared.read_bytes() if shared.exists() else json_bytes(asdict(baselines))
    )
    with write_shard(directory, 0) as staged:
        (staged / "selection.json").write_bytes(json_bytes(selection))
        (staged / "baselines.json").write_bytes(baseline_bytes)
        torch.save(initial, staged / "initial.pt")
        (staged / "metrics.json").write_bytes(json_bytes(metrics))
        write_pairs(staged, pairs)


def write_metrics(
    directory: Path,
    k: int,
    metrics: dict,
    pairs: Sequence[ReferencePair] | None,
) -> None:
    """(21): write one later shard in the layout fixed by accepted shard 0."""
    legacy = not (shard_path(directory, 0) / "pairs.json").is_file()
    if legacy != (pairs is None):
        raise RuntimeError("reference pairs layout differs from shard 000000")
    with write_shard(directory, k) as staged:
        (staged / "metrics.json").write_bytes(json_bytes(metrics))
        if pairs is not None:
            write_pairs(staged, pairs)
