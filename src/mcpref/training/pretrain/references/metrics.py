"""Preference metrics independent of clip or recurrent observation semantics."""

from mcpref.diagnostics.preference.suites import pool_preferences
from mcpref.diagnostics.pretrain.diagnostics import (
    GlobalFit,
    preference_baselines,
    preference_suite,
)
from mcpref.diagnostics.pretrain.observation import preference_observation
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import TrainingConfig


def preference_data(modules: Modules, out: tuple, labels: dict) -> tuple[dict, object]:
    terms = modules["P_loss"].terms(out, [labels])
    values = {key: value[0].item() for key, value in terms.items()}
    rows = preference_observation((out[0], out[1], [labels]))
    return values, rows


def preference_part(cfg: TrainingConfig, fit: GlobalFit, observations: dict) -> dict:
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
