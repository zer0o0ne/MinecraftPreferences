"""Metric assembly for fixed-reference evaluation of pretrain diagnostics."""

import copy

from mcpref.diagnostics.preference.suites import pool_preferences
from mcpref.diagnostics.pretrain.diagnostics import (
    GlobalFit,
    phi_suite,
    policy_suite,
    preference_baselines,
    preference_suite,
)
from mcpref.diagnostics.pretrain.observation import (
    preference_observation,
    step_observation,
)
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.references.select import PARTS


def preference_data(modules: Modules, out: tuple, labels: dict) -> tuple[dict, object]:
    terms = modules["P_loss"].terms(out, [labels])
    values = {key: value[0].item() for key, value in terms.items()}
    rows = preference_observation((out[0], out[1], [labels]))
    return values, rows


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


def preference_part(cfg: PretrainConfig, fit: GlobalFit, observations: dict) -> dict:
    rows = pool_preferences(observations["rows"])
    terms = observations["terms"]
    count = len(terms)
    losses = {
        "P": sum(value["total"] for value in terms) / count,
        "P_inc": sum(value["inc"] for value in terms) / count,
        "P_succ": sum(value["succ"] for value in terms) / count,
    }
    return {
        "losses": losses,
        "preference": preference_suite(
            rows,
            strata=fit.strata,
            calibration_bins=cfg.calibration_bins,
            saturation_threshold=cfg.saturation_threshold,
            inc_classification_threshold=cfg.inc_classification_threshold,
            quantiles=cfg.quantiles,
            full=True,
        ),
        "baselines": preference_baselines(rows, fit),
    }


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
