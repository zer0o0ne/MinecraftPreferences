"""Scalar oracles of specs/pretrain-diagnostics.md (5)-(11), (16) and observation types."""

import dataclasses
import inspect
import math

import pytest
import torch

from mcpref.pretrain.diagnostics import (
    GlobalFit,
    PhiObservations,
    PolicyObservations,
    PreferenceRows,
    Stratum,
    mean_nullable,
    phi_suite,
    policy_suite,
    preference_baselines,
    preference_suite,
)


def logits(values: list[float]) -> torch.Tensor:
    x = [
        math.log(v / (1 - v)) if 0 < v < 1 else (-100 if v == 0 else 100)
        for v in values
    ]
    return torch.tensor(x, dtype=torch.float32)


def rows(
    s: list[float], t: list[float], p: list[float], inc: list[int]
) -> PreferenceRows:
    return PreferenceRows(
        logits(s),
        logits(t),
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(inc, dtype=torch.uint8),
    )


def suite(
    rows_: PreferenceRows | None,
    *,
    full=True,
    quantiles=(0.05, 0.5, 0.95),
    strata=(),
    bins=2,
):
    return preference_suite(rows_, strata=strata, calibration_bins=bins, saturation_threshold=0.01, inc_classification_threshold=0.5, quantiles=quantiles, full=full)  # fmt: skip


def fit(inc_probability=0.25, succ_probability=0.5, comparable=3) -> GlobalFit:
    no_succ = succ_probability is None
    logit = lambda x: None if x is None or x in (0, 1) else math.log(x / (1 - x))
    return GlobalFit(4, 4 - comparable, comparable, None if no_succ else 1.5,
                     inc_probability, succ_probability, logit(inc_probability), logit(succ_probability), ())  # fmt: skip


def bce(x: float, y: float) -> float:
    return max(x, 0) - x * y + math.log1p(math.exp(-abs(x)))


def test_preference_arguments_are_required():
    parameters = inspect.signature(preference_suite).parameters
    assert all(p.default is inspect.Parameter.empty for p in parameters.values())


def test_step_and_full_schemas_and_conditional_succ():
    population = rows(
        [0.001, 0.02, 1.0, 0.5],
        [0.001, 0.02, 0.98, 1.0],
        [0.0, 0.5, 1.0, 0.5],
        [0, 0, 0, 1],
    )
    step = suite(population, full=False)
    assert set(step) == {"inc", "succ"}
    assert all(
        set(step[k]) == {"count", "mean", "std", "brier", "low", "high"}
        for k in ("inc", "succ")
    )
    assert step["inc"]["count"] == 4 and step["succ"]["count"] == 3
    assert step["inc"]["low"] == step["inc"]["high"] == 0.25
    assert step["succ"]["low"] == pytest.approx(1 / 3) and step["succ"][
        "high"
    ] == pytest.approx(1 / 3)
    full = suite(population)
    for key, count in (("inc", 4), ("succ", 3)):
        head = full[key]
        assert set(head) == {
            "count",
            "mean",
            "std",
            "brier",
            "low",
            "high",
            "quantiles",
            "calibration",
            "ece",
        }
        u = (
            torch.sigmoid(population.t if key == "inc" else population.s[:3])
            .double()
            .tolist()
        )
        y = population.inc.double().tolist() if key == "inc" else [0.0, 0.5, 1.0]
        mean = sum(u) / count
        assert head["mean"] == pytest.approx(mean, abs=1e-7)
        assert head["std"] == pytest.approx(
            math.sqrt(sum((x - mean) ** 2 for x in u) / count), abs=1e-7
        )
        assert head["brier"] == pytest.approx(
            sum((x - z) ** 2 for x, z in zip(u, y)) / count, abs=1e-7
        )
        assert len(head["calibration"]) == 2


def test_saturation_thresholds_are_strict():
    boundary = rows([0.5], [0.01], [0.5], [0])
    threshold = float(torch.sigmoid(boundary.t[0]))
    got = preference_suite(boundary, strata=(), calibration_bins=2, saturation_threshold=threshold,
                           inc_classification_threshold=0.5, quantiles=(0.5,), full=False)  # fmt: skip
    assert got["inc"]["low"] == 0
    boundary = rows([0.5], [0.99], [0.5], [0])
    threshold = 1 - float(torch.sigmoid(boundary.t[0]))
    got = preference_suite(boundary, strata=(), calibration_bins=2, saturation_threshold=threshold,
                           inc_classification_threshold=0.5, quantiles=(0.5,), full=False)  # fmt: skip
    assert got["inc"]["high"] == 0


def test_quantile_empty_singleton_even_odd_and_interpolation():
    empty = suite(None, quantiles=(0.25,))
    assert empty["inc"]["quantiles"] == {"0.25": None} and empty["succ"] is None
    for values, q, want in [
        ([0.2], 0.05, 0.2),
        ([0.2, 0.8], 0.5, 0.5),
        ([0.2, 0.4, 0.8], 0.5, 0.4),
        ([0.0, 0.2, 0.4, 1.0], 0.25, 0.15),
    ]:
        got = suite(
            rows([0.5] * len(values), values, [0.5] * len(values), [1] * len(values)),
            quantiles=(q,),
        )
        assert got["inc"]["quantiles"][str(q)] == pytest.approx(want, abs=1e-7)


def test_calibration_boundaries_empty_bins_and_soft_targets():
    population = rows(
        [0.0, 0.5, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.5, 1.0, 0.5], [0, 0, 0, 1]
    )
    got = suite(population, bins=2)
    assert [
        (b["lower"], b["upper"], b["count"]) for b in got["inc"]["calibration"]
    ] == [(0.0, 0.5, 1), (0.5, 1.0, 3)]
    assert got["inc"]["calibration"][0]["mean_prediction"] == 0
    assert got["inc"]["calibration"][1]["mean_prediction"] == pytest.approx(2 / 3)
    assert got["inc"]["calibration"][1]["mean_target"] == pytest.approx(1 / 3)
    assert got["inc"]["ece"] == pytest.approx(0.25 * 0 + 0.75 * abs(2 / 3 - 1 / 3))
    assert got["succ"]["calibration"][1]["mean_target"] == pytest.approx(0.75)
    empty_bins = suite(rows([0.5], [0.5], [0.5], [0]), bins=10)["inc"]["calibration"]
    assert len(empty_bins) == 10 and sum(b["count"] == 0 for b in empty_bins) == 9
    assert all(
        b["mean_prediction"] is b["mean_target"] is None
        for b in empty_bins
        if b["count"] == 0
    )


def test_incomparability_classification_and_one_class_nulls():
    got = suite(rows([0.5] * 4, [0.4, 0.5, 0.6, 0.4], [0.5] * 4, [0, 1, 0, 1]))["inc_classification"]  # fmt: skip
    assert got == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "accuracy": 0.5,
        "tpr": 0.5,
        "tnr": 0.5,
        "balanced_accuracy": 0.5,
    }
    for inc, null_key in (([0, 0], "tpr"), ([1, 1], "tnr")):
        one = suite(rows([0.5] * 2, [0.6, 0.7], [0.5] * 2, inc))["inc_classification"]
        assert one[null_key] is one["balanced_accuracy"] is None
    empty = suite(None)["inc_classification"]
    assert empty["accuracy"] is empty["balanced_accuracy"] is None


def test_strata_formulas_and_empty_catalogue_entries():
    strata = (Stratum("0.0", "comparable", 0.0, 1, 0), Stratum("0.25", "comparable", 0.25, 0, 0),
              Stratum("0.5", "comparable", 0.5, 1, 0), Stratum("1.0", "comparable", 1.0, 1, 0), Stratum("inc", "inc", None, 1, 0))  # fmt: skip
    population = rows(
        [0.0, 0.5, 1.0, 0.5], [0.5, 0.5, 0.5, 0.25], [0.0, 0.5, 1.0, 0.5], [0, 0, 0, 1]
    )
    got = suite(population, strata=strata)["strata"]
    assert [x["key"] for x in got] == ["0.0", "0.25", "0.5", "1.0", "inc"]
    for i, key in enumerate(("0.0", "0.5", "1.0")):
        u = float(torch.sigmoid(population.s[i]))
        target = (0.0, 0.5, 1.0)[i]
        assert got[i if i < 1 else i + 1]["mean_prediction"] == pytest.approx(
            u, abs=1e-7
        )
        assert got[i if i < 1 else i + 1]["bce"] == pytest.approx(
            2 * bce(float(population.s[i]), target), abs=1e-6
        )
        assert got[i if i < 1 else i + 1]["brier"] == pytest.approx(
            (u - target) ** 2, abs=1e-7
        )
    assert got[1]["count"] == 0 and all(
        got[1][k] is None for k in ("mean_prediction", "mean_target", "bce", "brier")
    )
    u = float(torch.sigmoid(population.t[3]))
    assert got[4]["mean_target"] == 1 and got[4]["brier"] == pytest.approx(
        (u - 1) ** 2, abs=1e-7
    )


def test_fixed_global_baselines_and_no_window_refit():
    population = rows(
        [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.7, 0.9], [0.0, 0.5, 1.0, 0.5], [0, 0, 0, 1]
    )
    got = preference_baselines(population, fit())
    u, v, n, nc = 0.25, 0.5, 4, 3
    inc = [-y * math.log(u) - (1 - y) * math.log(1 - u) for y in (0, 0, 0, 1)]
    succ = [-p * math.log(v) - (1 - p) * math.log(1 - v) for p in (0.0, 0.5, 1.0)]
    entropy = [0.0, math.log(2), 0.0]
    assert got["inc_constant"] == pytest.approx(2 * sum(inc) / n)
    assert got["zero"] == pytest.approx(2 * math.log(2) * (1 + nc / n))
    assert got["inc_only"] == pytest.approx(
        got["inc_constant"] + 2 * math.log(2) * nc / n
    )
    assert got["succ_constant"] == pytest.approx(2 * sum(succ) / n)
    assert got["both"] == pytest.approx(got["inc_constant"] + got["succ_constant"])
    assert got["target_entropy"] == pytest.approx(2 * sum(entropy) / n)
    assert all(preference_baselines(None, fit())[k] is None for k in got)


def test_baseline_empty_comparable_and_infinite_nulls():
    inc_only = rows([0.5], [0.5], [0.5], [1])
    got = preference_baselines(inc_only, fit())
    assert (
        got["target_entropy"] == got["succ_constant"] == 0
        and got["both"] == got["inc_constant"]
    )
    no_global_succ = preference_baselines(
        inc_only, fit(succ_probability=None, comparable=0)
    )
    assert no_global_succ["both"] is no_global_succ["succ_constant"] is None
    infinite_inc = preference_baselines(inc_only, fit(inc_probability=0.0))
    assert (
        infinite_inc["inc_constant"]
        is infinite_inc["inc_only"]
        is infinite_inc["both"]
        is None
    )
    comparable = rows([0.5], [0.5], [1.0], [0])
    infinite_succ = preference_baselines(comparable, fit(succ_probability=0.0))
    assert infinite_succ["succ_constant"] is infinite_succ["both"] is None


def policy_observation(valid=2, q=(0.5, 0.5), copy=0.3) -> PolicyObservations:
    return PolicyObservations(torch.tensor(q, dtype=torch.float64), valid, valid, 0.7 * valid, valid // 2,
                              0.6 * valid, torch.tensor([valid // 2, valid - valid // 2]),
                              torch.tensor([0.6, 0.8], dtype=torch.float64), torch.tensor([1.0, 0.0], dtype=torch.float64),
                              math.log(2), copy, 0.7)  # fmt: skip


def test_policy_suite_step_pooled_action_and_empirical_oracle():
    one = policy_observation()
    step = policy_suite([one], calibration_bins=2, copy_eps=0.1, full=False)
    assert step == {
        "count": 2,
        "top1_support": 0.5,
        "mean_confidence": 0.7,
        "action_entropy": 0.6,
    }
    two = policy_observation(valid=1, q=(0.25, 0.75), copy=None)
    got = policy_suite([one, two], calibration_bins=2, copy_eps=0.1, full=True)
    assert got["action"]["count"] == 3 and got["action"][
        "top1_support"
    ] == pytest.approx(1 / 3)
    assert got["action"]["mean_confidence"] == pytest.approx(2.1 / 3)
    assert (
        got["action"]["calibration"][0]["count"] == 1
        and got["action"]["calibration"][1]["count"] == 2
    )
    assert got["uniform"] == pytest.approx(math.log(2)) and got[
        "copy_current"
    ] == pytest.approx(0.3)
    assert got["zero_shift"] == pytest.approx(0.7)
    r = (0.375, 0.625)
    assert got["empirical_constant_oracle"] == pytest.approx(
        -0.75 * math.log(r[0]) - 1.25 * math.log(r[1])
    )


def test_policy_suite_empty_population_has_null_metrics():
    got = policy_suite([], calibration_bins=2, copy_eps=0.1, full=True)
    assert got["action"]["count"] == 0 and got["action"]["top1_support"] is None
    assert got["empirical_constant_oracle"] is None


def test_policy_all_zero_targets_keep_step_baselines_and_null_oracle():
    observation = dataclasses.replace(policy_observation(valid=0, q=(0.0, 0.0), copy=0.0),
                                      uniform=0.0, zero_shift=0.0)  # fmt: skip
    got = policy_suite([observation], calibration_bins=2, copy_eps=0.1, full=True)
    assert got["action"]["count"] == 0 and got["action"]["top1_support"] is None
    assert got["uniform"] == got["copy_current"] == got["zero_shift"] == 0
    assert got["empirical_constant_oracle"] is None


def phi_observation(changed=1) -> PhiObservations:
    return PhiObservations(2.0, 1.5 if changed else 0.0, changed, 2, 9.0, 3, 1)


def test_phi_suite_pooled_norm_hinge_and_empty_changed():
    got = phi_suite([phi_observation(), phi_observation(changed=0)])
    assert got == {"d_alpha_mean": 1.0, "d_alpha_changed_mean": 1.5, "z_norm_mean": 3.0,
                   "hinge_active": 1 / 3, "changed": 1, "state_pairs": 4}  # fmt: skip
    empty = phi_suite([phi_observation(changed=0)])
    assert empty["d_alpha_changed_mean"] is None and empty["changed"] == 0


def test_mean_nullable_includes_zero_and_counts_coverage():
    assert mean_nullable([1.0, None, 0.0, 3.0]) == (4 / 3, 3)
    assert mean_nullable([None, None]) == (None, 0)
