"""One ordinary optimizer step and its diagnostic record."""

import torch

from mcpref.pretrain.aggregate import WindowAccumulator
from mcpref.pretrain.checkpoints import Modules
from mcpref.pretrain.checks import Groups
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.diagnostics import (
    BASELINE_KEYS,
    GlobalFit,
    StepObservations,
    phi_suite,
    policy_suite,
    preference_baselines,
    preference_suite,
)
from mcpref.pretrain.items import Part
from mcpref.pretrain.loss import step_gradients
from mcpref.pretrain.update import gradient_stats, update, update_stats


def step_record(
    cfg: PretrainConfig,
    fit: GlobalFit | None,
    logged: dict,
    observations: StepObservations,
    gradients: dict,
    lr: dict,
    updates: dict,
) -> dict:
    """Assemble the exact step schema from detached observations and update statistics."""
    suite = preference_suite(
        observations.preference,
        strata=fit.strata if fit is not None else (),
        calibration_bins=cfg.calibration_bins,
        saturation_threshold=cfg.saturation_threshold,
        inc_classification_threshold=cfg.inc_classification_threshold,
        quantiles=cfg.quantiles,
        full=False,
    )
    baselines = (
        preference_baselines(observations.preference, fit)
        if fit is not None
        else {key: None for key in BASELINE_KEYS}
    )
    return {
        "step": logged["step"],
        "phi": logged["phi"],
        "pi": logged["pi"],
        "P": logged["P"],
        "loss": logged["loss"],
        "lr": lr,
        "counts": logged["counts"],
        "components": logged["components"],
        "outputs": {
            "P_inc": suite["inc"],
            "P_succ": suite["succ"],
            "pi_action": policy_suite(
                [observations.policy],
                calibration_bins=cfg.calibration_bins,
                copy_eps=cfg.action_copy_eps,
                full=False,
            ),
        },
        "phi_stats": phi_suite([observations.phi])
        | {"dual_lambda": logged["dual_lambda"]},
        "gradients": gradients,
        "updates": updates,
        "baselines": baselines,
    }


def step(
    cfg: PretrainConfig,
    modules: Modules,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    data: Part,
    batch: list[dict],
    t: int,
    fit: GlobalFit | None,
) -> tuple[dict, StepObservations]:
    """The unchanged backwards/clipping/LR/AdamW order with detached observations."""
    optimizer.zero_grad(set_to_none=True)
    logged = step_gradients(
        batch,
        modules,
        read=data.read,
        m=data.m,
        k_tot=data.k_tot,
        r_tot=data.r_tot,
        w_pi=cfg.w_pi,
        w_P=cfg.w_P,
        micro_batch=cfg.micro_batch,
        cfg=cfg,
    )
    logged["step"] = t
    observations = logged["observations"]
    gradients = gradient_stats(groups, cfg.grad_clip, preference=modules.get("P"))
    lr = update(cfg, optimizer, groups, t, data.total)
    updates = update_stats(cfg, optimizer, groups, lr)
    return (
        step_record(cfg, fit, logged, observations, gradients, lr, updates),
        observations,
    )


def accumulator(cfg: PretrainConfig, fit: GlobalFit | None) -> WindowAccumulator:
    return WindowAccumulator(
        cfg.diagnostic_window,
        strata=fit.strata if fit is not None else (),
        calibration_bins=cfg.calibration_bins,
        saturation_threshold=cfg.saturation_threshold,
        inc_classification_threshold=cfg.inc_classification_threshold,
        quantiles=cfg.quantiles,
        action_copy_eps=cfg.action_copy_eps,
    )
