"""WindowAccumulator oracles of specs/pretrain-diagnostics.md (14)-(16) and loop wiring."""

import inspect
import math

import pytest
import torch

from mcpref.pretrain.aggregate import WindowAccumulator
from mcpref.pretrain.diagnostics import (
    PhiObservations,
    PolicyObservations,
    PreferenceRows,
    StepObservations,
    Stratum,
)

from .test_diagnostics import rows


def accumulator(window=2) -> WindowAccumulator:
    strata = (Stratum("0.0", "comparable", 0.0, 1, 0), Stratum("1.0", "comparable", 1.0, 1, 0), Stratum("inc", "inc", None, 0, 0))  # fmt: skip
    return WindowAccumulator(window, strata=strata, calibration_bins=2, saturation_threshold=0.01,
                             inc_classification_threshold=0.5, quantiles=(0.5,), action_copy_eps=0.1)  # fmt: skip


def policy(
    valid: int,
    confidence: float,
    correct: int,
    q: tuple[float, float],
    copy: float | None,
) -> PolicyObservations:
    return PolicyObservations(torch.tensor(q, dtype=torch.float64), valid, valid, confidence * valid, correct,
                              0.5 * valid, torch.tensor([0, valid], dtype=torch.int64),
                              torch.tensor([0.0, confidence * valid], dtype=torch.float64),
                              torch.tensor([0.0, float(correct)], dtype=torch.float64), math.log(2), copy, 0.25)  # fmt: skip


def phi(d_sum: float) -> PhiObservations:
    return PhiObservations(d_sum, d_sum / 2, 1, 2, 6.0, 2, 1)


def record(
    step: int,
    preference: PreferenceRows | None,
    copy: float | None = 0.4,
    unavailable=False,
) -> tuple[dict, StepObservations]:
    paired = preference is not None
    n = len(preference.inc) if paired else 0
    comparable = int((preference.inc == 0).sum()) if paired else 0
    sum_p = float(preference.p[preference.inc == 0].sum()) if comparable else None
    valid = 1 if step == 1 else 3
    confidence, correct = (0.2, 0) if step == 1 else (0.8, 3)
    q = (1.0, 0.0) if step == 1 else (0.0, 1.0)
    observations = StepObservations(
        preference, policy(valid, confidence, correct, q, copy), phi(float(step))
    )
    null = None
    entry = {
        "step": step,
        "phi": 1.0,
        "pi": 2.0,
        "P": float(step) if paired else None,
        "loss": 3.0,
        "components": {
            "phi": {"push": 1.0, "constraint": 2.0, "dual": 3.0, "norm": 4.0},
            "pi": {"action": 5.0, "shift": 6.0},
            "P": {
                "inc": float(step) if paired else null,
                "succ": 2.0 if paired else null,
            },
        },
        "baselines": dict.fromkeys(
            (
                "zero",
                "inc_only",
                "both",
                "target_entropy",
                "inc_constant",
                "succ_constant",
            ),
            float(step - 1) if paired else null,
        ),
        "counts": {
            "items": 1,
            "clips": 2 if paired else 1,
            "pairs": n,
            "comparable": comparable,
            "incomparable": n - comparable,
            "sum_p": sum_p,
            "action_targets": valid,
            "shift_targets": valid,
            "phi_changed": 1,
            "phi_state_pairs": 2,
        },
        "phi_stats": {"dual_lambda": float(step)},
        "gradients": {
            "global": {
                "norm": 2.0 if step == 1 else 0.0,
                "clip_coefficient": 0.5,
                "active": True,
            },
            "groups": {"g": {"norm": 1.0 if step == 1 else 0.0, "active": True}},
            "P_heads": {
                "pref_weight": {
                    "norm": 0.0 if step == 1 else null,
                    "active": step == 1,
                },
                "inc_weight": {"norm": null, "active": False},
                "inc_bias": {"norm": null, "active": False},
            }
            if paired
            else null,
        },
        "updates": {
            "groups": {
                "g": {
                    "norm": null if unavailable else 2.0,
                    "relative": null if unavailable else 0.5,
                    "unavailable": "zero_decay_factor" if unavailable else null,
                }
            }
        },
    }
    return entry, observations


def test_window_equal_step_losses_pooled_outputs_and_exact_coverage():
    acc = accumulator()
    first = rows([0.0], [0.0], [0.0], [0])
    second = rows([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0, 0, 0])
    acc.add(*record(1, first, copy=None))
    assert acc.drain(final=False) == []
    acc.add(*record(2, second))
    (got,) = acc.drain(final=False)
    assert list(got) == ["version", "first_step", "last_step", "partial", "losses", "coverage", "counts",
                         "preference", "pi", "phi", "gradients", "updates"]  # fmt: skip
    assert (got["version"], got["first_step"], got["last_step"], got["partial"]) == (
        1,
        1,
        2,
        False,
    )
    assert got["losses"]["P"] == 1.5 and got["losses"]["components"]["P"]["succ"] == 2
    assert (
        got["losses"]["baselines"]["zero"] == 0.5
        and got["coverage"]["losses"]["baselines"]["zero"] == 2
    )
    assert got["coverage"]["pi"] == {"uniform": 2, "copy_current": 1, "zero_shift": 2}
    assert (
        got["counts"]["pairs"] == 4
        and got["counts"]["comparable"] == 4
        and got["counts"]["sum_p"] == 1.0
    )
    assert got["preference"]["succ"]["mean"] == pytest.approx(0.75)
    assert got["preference"]["succ"]["mean"] != pytest.approx(0.5)
    strata = {x["key"]: x for x in got["preference"]["strata"]}
    assert strata["0.0"]["count"] == 3 and strata["0.0"][
        "mean_prediction"
    ] == pytest.approx(2 / 3)
    assert strata["1.0"]["count"] == 1 and strata["1.0"]["mean_prediction"] == 1
    assert strata["inc"]["count"] == 0 and strata["inc"]["mean_prediction"] is None
    assert got["pi"]["action"]["top1_support"] == pytest.approx(0.75)
    assert got["pi"]["action"]["mean_confidence"] == pytest.approx(0.65)
    assert got["pi"]["copy_current"] == pytest.approx(0.4) and got["pi"][
        "uniform"
    ] == pytest.approx(math.log(2))
    assert got["pi"]["empirical_constant_oracle"] == pytest.approx(2 * math.log(2))
    assert got["phi"]["dual_lambda"] == 1.5 and got["phi"]["d_alpha_mean"] == 0.75
    assert got["gradients"]["global"] == {
        "mean_norm": 1.0,
        "mean_clip_coefficient": 0.5,
        "active_steps": 2,
    }
    assert got["gradients"]["groups"]["g"] == {"mean_norm": 0.5, "active_steps": 2}
    assert got["gradients"]["P_heads"]["pref_weight"] == {
        "mean_norm": 0.0,
        "active_steps": 1,
    }
    assert acc.drain(final=False) == []


def test_window_null_loss_coverage_and_unavailable_updates():
    acc = accumulator(window=1)
    acc.add(*record(1, None, unavailable=True))
    (got,) = acc.drain()
    assert got["losses"]["P"] is None and got["coverage"]["losses"]["P"] == 0
    assert all(value == 0 for value in got["coverage"]["losses"]["baselines"].values())
    assert got["counts"]["sum_p"] is None and got["preference"]["succ"] is None
    assert got["gradients"]["P_heads"] is None
    update = got["updates"]["groups"]["g"]
    assert update == {"mean_norm": None, "mean_relative": None, "norm_steps": 0, "relative_steps": 0,
                      "unavailable": {"zero_decay_factor": 1}}  # fmt: skip


def test_inactive_gradient_means_exclude_steps():
    acc = accumulator(window=1)
    entry, observations = record(1, None)
    entry["gradients"] = {"global": {"norm": None, "clip_coefficient": None, "active": False},
                          "groups": {"g": {"norm": None, "active": False}}, "P_heads": None}  # fmt: skip
    acc.add(entry, observations)
    (got,) = acc.drain()
    assert got["gradients"]["global"] == {
        "mean_norm": None,
        "mean_clip_coefficient": None,
        "active_steps": 0,
    }
    assert got["gradients"]["groups"]["g"] == {"mean_norm": None, "active_steps": 0}


def test_consecutive_steps_full_partial_empty_and_single_final():
    acc = accumulator()
    acc.add(*record(1, None))
    with pytest.raises(ValueError):
        acc.add(*record(3, None))
    acc.drain(final=True)
    acc = accumulator()
    for step in (1, 2, 3):
        acc.add(*record(step, None))
    got = acc.drain(final=True)
    assert [(x["first_step"], x["last_step"], x["partial"]) for x in got] == [
        (1, 2, False),
        (3, 3, True),
    ]
    with pytest.raises(RuntimeError):
        acc.drain(final=True)
    empty = accumulator()
    assert empty.drain(final=True) == []
    with pytest.raises(RuntimeError):
        empty.drain(final=True)


def test_constructor_has_no_hidden_diagnostic_defaults():
    parameters = inspect.signature(WindowAccumulator).parameters
    assert all(
        p.default is inspect.Parameter.empty
        for name, p in parameters.items()
        if name != "self"
    )


def test_loop_forwards_global_strata_and_every_diagnostic_config_value(tmp_path):
    from .loop_run import Loop

    loop = Loop(tmp_path).write()
    real = WindowAccumulator.__init__
    calls = []

    def capture(self, *args, **kwargs):
        calls.append((args, kwargs))
        return real(self, *args, **kwargs)

    loop.run(patch=lambda mp: mp.setattr(WindowAccumulator, "__init__", capture))
    assert len(calls) == 1
    args, kwargs = calls[0]
    cfg = loop.config
    assert args == (cfg["diagnostic_window"],)
    from mcpref.pretrain.reference import load_global_fit

    assert kwargs == {"strata": tuple(load_global_fit(loop.out).strata),
                      "calibration_bins": cfg["calibration_bins"],
                      "saturation_threshold": cfg["saturation_threshold"],
                      "inc_classification_threshold": cfg["inc_classification_threshold"],
                      "quantiles": tuple(cfg["quantiles"]), "action_copy_eps": cfg["action_copy_eps"]}  # fmt: skip


def test_checkpoint_diagnostics_full_windows_and_final_partial(tmp_path):
    import json

    from .index_oracle import json_bytes
    from .loop_run import Loop

    loop = Loop(tmp_path, diagnostic_window=4, checkpoint_steps=4).write()
    loop.run()
    for j in range(3):
        data = (loop.checkpoint_dir(j) / "diagnostics.json").read_bytes()
        assert data == json_bytes(json.loads(data))
    first = json.loads((loop.checkpoint_dir(0) / "diagnostics.json").read_bytes())
    second = json.loads((loop.checkpoint_dir(1) / "diagnostics.json").read_bytes())
    final = json.loads((loop.checkpoint_dir(2) / "diagnostics.json").read_bytes())
    assert [(x["first_step"], x["last_step"], x["partial"]) for x in first] == [
        (1, 4, False)
    ]
    assert [(x["first_step"], x["last_step"], x["partial"]) for x in second] == [
        (5, 8, False)
    ]
    assert [(x["first_step"], x["last_step"], x["partial"]) for x in final] == [
        (9, 10, True)
    ]
