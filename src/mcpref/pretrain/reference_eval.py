"""Public fixed-reference evaluation interfaces of pretrain diagnostics."""

import torch

from mcpref.diagnostics.types import ReferencePair
from mcpref.pretrain.checkpoints import Modules
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.diagnostics import GlobalFit
from mcpref.pretrain.reference import ReferenceData
from mcpref.pretrain.reference_conditions import (
    evaluate_initial_event,
    evaluate_reference_event,
    evaluation_state,
)

__all__ = ["evaluate_initial", "evaluate_reference", "evaluation_state"]


def evaluate_initial(
    cfg: PretrainConfig,
    modules: Modules,
    selection: dict,
    baselines: GlobalFit,
    run: dict,
) -> tuple[
    dict[str, torch.Tensor | dict[str, torch.Tensor]], dict, list[ReferencePair]
]:
    """(18): step-0 latents/state, metrics and new typed pair artifacts."""
    initial, metrics, pairs, _ = evaluate_initial_event(
        cfg, modules, selection, baselines, run
    )
    return initial, metrics, pairs


def evaluate_reference(
    cfg: PretrainConfig,
    modules: Modules,
    reference: ReferenceData,
    run: dict,
    step: int,
) -> tuple[dict, list[ReferencePair] | None]:
    """(20)-(21): four conditions, with pairs only in the new layout."""
    metrics, pairs, _ = evaluate_reference_event(cfg, modules, reference, run, step)
    return metrics, pairs
