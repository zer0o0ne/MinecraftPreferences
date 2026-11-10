"""Equation (7), Distribution reductions and reference mapping/reasons."""

import math

import pytest

from diagnostics.core.world import bce, config, context, distribution, fit, pair, rows
from mcpref.diagnostics.pretrain.diagnostics import on_pretrain_step, on_reference


def assert_distribution(actual, expected):
    for key, value in expected.items():
        if key == "quantiles":
            assert actual[key] == pytest.approx(value)
        else:
            assert (
                actual[key] == pytest.approx(value)
                if value is not None
                else actual[key] is None
            )


def loss_values(observations):
    result = {"P_inc": [], "P_succ": [], "P": []}
    for s, t, p, inc in zip(
        observations.s, observations.t, observations.p, observations.inc
    ):
        inc_loss = 2 * bce(float(t), int(inc))
        succ_loss = 0.0 if inc else 2 * bce(float(s), float(p))
        result["P_inc"].append(inc_loss)
        result["P_succ"].append(succ_loss)
        result["P"].append(inc_loss + succ_loss)
    return result


def test_pretrain_distributions_and_corrected_progress_naive_oracle():
    observations = rows()
    record = on_pretrain_step(
        observations,
        fit=fit(),
        cfg=config(),
        context=context(
            "checkpoints/000004/losses.json", "pretrain", "joint", steps=(400,)
        ),
    )
    expected = loss_values(observations)
    for name, values in expected.items():
        assert_distribution(record["metrics"][name], distribution(values))
    fitted = fit()
    baseline_inc = 2 * bce(fitted.inc_logit, 0)
    baseline = baseline_inc + 2 * math.log(2)
    floor = (
        baseline_inc
        + 2
        * sum(
            -p * math.log(p) - (1 - p) * math.log(1 - p) if p not in (0, 1) else 0
            for p in (0.0, 0.5, 1.0)
        )
        / 3
    )
    actual = sum(expected["P"]) / 3
    progress = (baseline - actual) / (baseline - floor)
    assert record["metrics"]["pretrain.progress"]["mean"] == pytest.approx(progress)
    assert record["metrics"]["pretrain.progress"]["negative"] == 1


def test_progress_zero_at_zero_s_baseline_and_one_at_soft_target_floor():
    p = (0.25, 0.75)
    s = tuple(math.log(value / (1 - value)) for value in p)
    fitted = fit(inc=0.5, succ=0.5)
    baseline_record = on_pretrain_step(
        rows(s=(0, 0), t=(0, 0), p=p),
        fit=fitted,
        cfg=config(),
        context=context(
            "checkpoints/000004/losses.json", "pretrain", "joint", steps=(100,)
        ),
    )
    ideal_record = on_pretrain_step(
        rows(s=s, t=(0, 0), p=p),
        fit=fitted,
        cfg=config(),
        context=context(
            "checkpoints/000004/losses.json", "pretrain", "joint", steps=(100,)
        ),
    )
    assert baseline_record["metrics"]["pretrain.progress"]["mean"] == pytest.approx(
        0, abs=1e-7
    )
    assert ideal_record["metrics"]["pretrain.progress"]["mean"] == pytest.approx(
        1, abs=1e-7
    )


def test_progress_null_when_baseline_equals_floor_and_empty_rows_are_not_zero_loss():
    degenerate = on_pretrain_step(
        rows(s=(0,), t=(0,), p=(0.5,), inc=(1,)),
        fit=fit(succ=None),
        cfg=config(),
        context=context(
            "checkpoints/000004/losses.json", "pretrain", "joint", steps=(100,)
        ),
    )
    assert degenerate["metrics"]["pretrain.progress"]["mean"] is None
    empty = on_pretrain_step(
        None,
        fit=fit(),
        cfg=config(),
        context=context(
            "checkpoints/000004/losses.json", "pretrain", "joint", steps=(100,)
        ),
    )
    for name in ("P", "P_inc", "P_succ", "pretrain.progress"):
        metric = empty["metrics"][name]
        assert metric["n"] == 0 and metric["mean"] is None and metric["reason"]


@pytest.mark.parametrize(
    "condition, applicable", [("joint", "delta_joint"), ("phi_only", "delta_only_phi")]
)
def test_reference_pairs_and_deltas_only_in_applicable_condition(condition, applicable):
    record = on_reference(
        rows(),
        pairs=[pair()],
        cfg=config(),
        context=context(
            "reference/000001/metrics.json",
            "pretrain-reference",
            condition,
            steps=(500,),
        ),
    )
    other = "delta_only_phi" if applicable == "delta_joint" else "delta_joint"
    assert record["metrics"][applicable]["n"] == 1
    assert record["metrics"][applicable]["mean"] == getattr(pair(), applicable)
    assert record["metrics"][other]["reason"] == "not_applicable_condition"
    assert record["populations"][applicable] == ["e0", "e1"]


@pytest.mark.parametrize("condition", ["P_only", "initial"])
def test_reference_deltas_not_applicable_for_other_conditions(condition):
    record = on_reference(
        rows(),
        pairs=[pair()],
        cfg=config(),
        context=context(
            "reference/000001/metrics.json",
            "pretrain-reference",
            condition,
            steps=(500,),
        ),
    )
    for name in ("delta_joint", "delta_only_phi"):
        assert record["metrics"][name]["mean"] is None
        assert record["metrics"][name]["reason"] == "not_applicable_condition"


@pytest.mark.parametrize("condition", ["joint", "phi_only"])
def test_legacy_reference_applicable_delta_reason(condition):
    record = on_reference(
        rows(),
        pairs=None,
        cfg=config(),
        context=context(
            "reference/000001/metrics.json",
            "pretrain-reference",
            condition,
            steps=(500,),
        ),
    )
    name = "delta_joint" if condition == "joint" else "delta_only_phi"
    assert record["metrics"][name]["mean"] is None
    assert record["metrics"][name]["reason"] == "legacy_reference_pairs_absent"


def test_late_initial_uses_aggregate_fallback_not_fabricated_pair_rows():
    record = on_reference(
        None,
        pairs=None,
        cfg=config(),
        context=context(
            "reference/000001/metrics.json",
            "pretrain-reference",
            "initial",
            steps=(500,),
        ),
    )
    assert record["metrics"]["P"]["n"] == 0
    assert record["metrics"]["P"]["reason"] == "empty_pair_population"
    assert record["joined"] == {}


def test_validation_model_thresholds_use_strict_per_pair_rates():
    boundaries = (math.log(0.2 / 0.8), math.log(0.8 / 0.2))
    observations = rows(
        s=(-3.0, *boundaries, 3.0), t=(0.0,) * 4, p=(0.0, 0.25, 0.75, 1.0)
    )
    record = on_reference(
        observations,
        pairs=[pair()],
        cfg=config(),
        context=context(
            "reference/000001/metrics.json",
            "pretrain-reference",
            "joint",
            split="validation",
            steps=(500,),
            clusters=("e0", "e1", "e2", "e3"),
        ),
    )
    low = record["metrics"]["target.model_lt_0_2"]
    high = record["metrics"]["target.model_gt_0_8"]
    probabilities = __import__("torch").sigmoid(observations.s)
    low_numerator = sum(float(value) < 0.2 for value in probabilities)
    high_numerator = sum(float(value) > 0.8 for value in probabilities)
    assert (low["n"], low["numerator"], low["value"]) == (
        4,
        low_numerator,
        low_numerator / 4,
    )
    assert (high["n"], high["numerator"], high["value"]) == (
        4,
        high_numerator,
        high_numerator / 4,
    )
    assert record["populations"]["target.model_lt_0_2"] == ["e0", "e1", "e2", "e3"]


def test_pretrain_hooks_require_cfg_and_context():
    with pytest.raises(TypeError):
        on_pretrain_step(None, fit=fit())
    with pytest.raises(TypeError):
        on_reference(None, pairs=None, cfg=config())
