"""Hand-calculated mixed targets and pooled action-change denominators."""

import pytest
import torch

from mcpref.diagnostics.pretrain import change
from mcpref.diagnostics.pretrain.diagnostics import policy_suite
from mcpref.diagnostics.pretrain.observation import (
    MicroObservations,
    phi_observation,
    policy_observation,
    step_observation,
)


def rows():
    # A, B, D. The last two rows are unknown predecessor and empty current GT.
    q = torch.tensor(
        [[0, 1, 0], [.5, .5, 0], [.25, .75, 0], [0, 1, 0], [0, 1, 0],
         [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]],
        dtype=torch.float64,
    )[:, None]  # fmt: skip
    previous = torch.tensor([[1, 0, 0]] * 10)[:, None]
    previous[3] = 0  # Known idle, unlike the excluded unknown row.
    logits = torch.zeros_like(q)
    top = [1, 0, 1, 1, 0, 2, 0, 0, 0, 0]
    logits[torch.arange(10), 0, top] = 2
    p = torch.tensor([0.8, 0.5, 0.2, 0.8, 0.8, 0.8, 0.8, 0.2, 0.8, 0.8])[:, None]
    known = torch.tensor([True] * 8 + [False, True])[:, None]
    return logits, q, previous, p, known


def test_fractional_oracle_and_exclusions():
    result = change.counts(*rows())
    expected = torch.tensor(
        [[3, 6], [4.5, 5.25], [2.75, 5.25], [2, 5.25], [1.5, 2.75]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)
    suite = change.suite(result)
    assert suite["count"] == 8
    for name, value in zip(change.METRICS, [1 / 2, 6 / 7, 11 / 21, 8 / 21, 6 / 11]):
        assert suite[name]["value"] == pytest.approx(value)


def micro(inputs):
    logits, q, previous, p, known = inputs
    valid = {"action_valid": q.sum(-1) > 0, "shift_valid": known}
    policy, length = policy_observation(
        (logits, q, previous, valid, torch.zeros(q.shape[0]), (p, known)), 2, 0.1
    )
    phi = phi_observation(
        {name: torch.zeros(1) for name in ("changed", "d_alpha", "z_norm", "hinge")}
        | {"changed": torch.zeros(1, dtype=torch.bool)}
    )
    return MicroObservations(None, policy, phi, length)


def test_unequal_micro_batches_and_steps_pool_sums_before_dividing():
    inputs = rows()
    one = micro([value[:1] for value in inputs])
    rest = micro([value[1:] for value in inputs])
    together = step_observation([one, rest], clips=10).policy
    steps = [
        step_observation([one], clips=1).policy,
        step_observation([rest], clips=9).policy,
    ]
    full = policy_suite(steps, calibration_bins=2, copy_eps=0.1, full=True)
    torch.testing.assert_close(together.change_counts, change.counts(*inputs))
    assert together.change_counts.dtype == torch.float64
    assert full["action"]["change"] == change.suite(together.change_counts)
    assert full["action"]["change"]["repeat_when_predicted"]["value"] == 0.5


@pytest.mark.parametrize("mode", ["repeats", "changes", "head_repeat", "excluded"])
def test_empty_conditional_denominators_are_null(mode):
    logits = torch.tensor([[0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0]]) if mode == "repeats" else torch.tensor([[0.0, 1.0]])
    p = torch.tensor([0.2 if mode == "head_repeat" else 0.8])
    result = change.suite(
        change.counts(
            logits, q, torch.tensor([[1.0, 0.0]]), p, torch.tensor([mode != "excluded"])
        )
    )
    nulls = {
        "repeats": ("head_recall", "action_recall", "joint_recall"),
        "changes": ("false_positive",),
        "head_repeat": ("repeat_when_predicted", "false_positive"),
        "excluded": change.METRICS,
    }
    for key in nulls[mode]:
        assert result[key]["value"] is None and result[key]["denominator"] == 0
    if mode == "head_repeat":
        assert result["head_recall"]["value"] == result["joint_recall"]["value"] == 0


def test_bfloat_targets_normalize_and_argmax_ties_choose_lowest_index():
    q = torch.full((1, 3), 1 / 3, dtype=torch.bfloat16)
    result = change.suite(
        change.counts(
            torch.zeros(1, 3),
            q,
            torch.tensor([[0, 1, 1]]),
            torch.tensor([0.5], requires_grad=True),
            torch.tensor([True]),
        )
    )
    assert result["head_recall"]["denominator"] == pytest.approx(1 / 3)
    assert result["false_positive"]["denominator"] == pytest.approx(2 / 3)
    assert result["action_recall"]["value"] == 1
    assert result["repeat_when_predicted"]["value"] == 0
