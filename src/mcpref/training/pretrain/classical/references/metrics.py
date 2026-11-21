"""Metric assembly for fixed-reference evaluation of pretrain diagnostics."""

import copy

from mcpref.diagnostics.pretrain.diagnostics import (
    GlobalFit,
    phi_suite,
    policy_suite,
)
from mcpref.diagnostics.pretrain.observation import (
    step_observation,
)
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.classical.config import PretrainConfig
from mcpref.training.pretrain.references.metrics import preference_part
from mcpref.training.pretrain.references.selection import PARTS


def empty_part(conditions: tuple[str, ...]) -> dict:
    return {
        "conditions": {
            condition: {"terms": [], "rows": []} for condition in conditions
        },
        "micro": [],
        "phi": {key: [] for key in ("push", "constraint", "dual", "norm", "total")},
        "pi": {key: [] for key in ("action", "shift", "total")},
    }


def add_preference(part: dict, condition: str, value: tuple[dict, object]) -> None:
    terms, rows = value
    part["conditions"][condition]["terms"].append(terms)
    part["conditions"][condition]["rows"].append(rows)


def preference_only(part: dict) -> dict:
    return {
        "losses": {key: part["losses"][key] for key in ("P", "P_inc", "P_succ")},
        "preference": copy.deepcopy(part["preference"]),
        "baselines": copy.deepcopy(part["baselines"]),
    }


def shared_initial(metrics: dict, conditions: tuple[str, ...]) -> dict:
    """(18): compute joint once and copy preference-only conditions at step 0."""
    result = {
        condition: {part: preference_only(metrics[part]["joint"]) for part in PARTS}
        for condition in conditions
    }
    result["joint"] = {part: metrics[part]["joint"] for part in PARTS}
    result["initial"] = copy.deepcopy(result["phi_only"])
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def finish_part(
    cfg: PretrainConfig,
    modules: Modules,
    fit: GlobalFit,
    part: dict,
    conditions: tuple[str, ...],
) -> dict:
    result = {
        condition: preference_part(cfg, fit, part["conditions"][condition])
        for condition in conditions
    }
    if "joint" not in conditions:
        return result
    joint = result["joint"]
    joint["losses"] |= {
        "phi": mean(part["phi"]["total"]),
        "phi_push": mean(part["phi"]["push"]),
        "phi_constraint": mean(part["phi"]["constraint"]),
        "phi_dual": mean(part["phi"]["dual"]),
        "phi_norm": mean(part["phi"]["norm"]),
        "pi": mean(part["pi"]["total"]),
        "pi_action": mean(part["pi"]["action"]),
        "pi_shift": mean(part["pi"]["shift"]),
    }
    observations = step_observation(part["micro"], 2 * cfg.reference_pairs)
    joint["pi"] = policy_suite(
        [observations.policy],
        calibration_bins=cfg.calibration_bins,
        copy_eps=cfg.action_copy_eps,
        full=True,
    )
    joint["phi"] = {
        key: mean(part["phi"][key]) for key in ("push", "constraint", "dual", "norm")
    }
    joint["phi"]["dual_lambda"] = modules["phi_loss"].log_lambda.exp().item()
    joint["phi"] |= phi_suite([observations.phi])
    return result
