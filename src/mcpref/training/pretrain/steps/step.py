"""One ordinary optimizer step and its diagnostic record."""

import torch

from mcpref.diagnostics.pretrain.aggregate import WindowAccumulator
from mcpref.diagnostics.pretrain.diagnostics import (
    BASELINE_KEYS,
    GlobalFit,
    StepObservations,
    phi_suite,
    policy_suite,
    preference_baselines,
    preference_suite,
)
from mcpref.infrastructure.profiling.scopes import measured
from mcpref.training.checkpoints import Modules
from mcpref.training.groups import Groups
from mcpref.training.optim import update
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.data.items import Part
from mcpref.training.pretrain.steps.loss import step_gradients
from mcpref.training.pretrain.update import gradient_stats, update_stats


@measured("diagnostics.record")
def step_record(
    cfg: PretrainConfig,
    fit: GlobalFit | None,
    logged: dict,
    observations: StepObservations,
    gradients: dict | None,
    lr: dict,
    updates: dict | None,
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
        w_phi=cfg.w_phi,
        w_pi=cfg.w_pi,
        w_P=cfg.w_P,
        micro_batch=cfg.micro_batch,
        cfg=cfg,
    )
    logged["step"] = t
    observations = logged["observations"]
    gradients = (
        gradient_stats(groups, cfg.grad_clip, preference=modules.get("P"))
        if cfg.gradient_diagnostics
        else None
    )
    lr = update(cfg, optimizer, groups, t, data.total)
    updates = (
        update_stats(cfg, optimizer, groups, lr) if cfg.gradient_diagnostics else None
    )
    optimizer.zero_grad(set_to_none=True)
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
