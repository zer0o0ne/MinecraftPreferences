"""Deterministic stratified selection shared by both pretraining modes."""

import math

import numpy as np

from mcpref.diagnostics.pretrain.diagnostics import GlobalFit
from mcpref.training.pretrain.config import TrainingConfig

PARTS = ("train", "validation")


def part_pairs(fit: GlobalFit, part: str) -> int:
    return sum(getattr(stratum, part) for stratum in fit.strata)


def allocations(fit: GlobalFit, part: str, pairs: int) -> dict[str, int]:
    """(17): largest-remainder allocation in catalogue order."""
    total = part_pairs(fit, part)
    if total < pairs:
        raise RuntimeError(f"{part} has only {total} pairs, reference needs {pairs}")
    exact = [pairs * getattr(stratum, part) / total for stratum in fit.strata]
    result = {
        stratum.key: math.floor(value) for stratum, value in zip(fit.strata, exact)
    }
    remaining = pairs - sum(result.values())
    order = sorted(
        range(len(fit.strata)),
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in order[:remaining]:
        result[fit.strata[index].key] += 1
    return result


def selected_ordinals(cfg: TrainingConfig, fit: GlobalFit) -> dict:
    """One local PCG64 stream in part/stratum order, with sorted source ordinals."""
    generator = np.random.Generator(np.random.PCG64([cfg.seed, 4]))
    selected = {}
    for part in PARTS:
        allocation = allocations(fit, part, cfg.reference_pairs)
        selected[part] = {}
        for stratum in fit.strata:
            count = getattr(stratum, part)
            chosen = generator.permutation(count)[: allocation[stratum.key]]
            selected[part][stratum.key] = sorted(chosen.tolist())
    return selected
