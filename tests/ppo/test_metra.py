"""Unit skills, METRA loss and its dual against specs/ppo.md (5)-(6)."""

import math

import pytest
import torch
from torch.testing import assert_close

from mcpref.ppo.config import PpoConfig
from mcpref.ppo.metra import MetraLoss, metra_loss, unit_skill

F64 = torch.float64
ALPHA = torch.tensor([2.0, 0.5, 3.0], dtype=F64)
ACTUAL = torch.tensor([[0.1, 0.2, 0.0], [1.0, -2.0, 0.5]], dtype=F64)
SKILL = torch.tensor([[0.3, -0.4, 0.2], [-0.5, 0.25, 1.0]], dtype=F64)


def unit_oracle(predicted, alpha, eps):
    rows = []
    weights = alpha.detach().tolist()
    for row in predicted.tolist():
        norm = math.sqrt(
            sum((value * weight) ** 2 for value, weight in zip(row, weights))
        )
        denominator = max(norm, eps)
        rows.append([value / denominator for value in row])
    return torch.tensor(rows, dtype=predicted.dtype)


def metra_oracle(actual, skill, alpha, log_lambda, slack):
    log_value = float(log_lambda.detach())
    lam = math.exp(log_value)
    rows = []
    weights = alpha.detach().tolist()
    for delta, zeta in zip(actual.tolist(), skill.tolist()):
        weighted_delta = [value * weight for value, weight in zip(delta, weights)]
        weighted_skill = [value * weight for value, weight in zip(zeta, weights)]
        inner = sum(x * y for x, y in zip(weighted_delta, weighted_skill))
        q = min(slack, 1.0 - sum(value**2 for value in weighted_delta))
        rows.append(-inner - lam * q + log_value * q)
    return torch.tensor(rows, dtype=actual.dtype)


def config(**changes):
    base = {
        "horizon": 1,
        "clip": 0.2,
        "skill_eps": 0.5,
        "dual_init": 2.5,
        "dual_slack": 0.1,
    }
    return PpoConfig(**(base | changes))


def test_unit_skill_oracle_covers_unit_sub_epsilon_zero_and_exact_epsilon():
    predicted = torch.tensor(
        [
            [0.3, 0.4, 0.0],
            [0.1, 0.2, 0.1],
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=F64,
        requires_grad=True,
    )
    alpha = ALPHA.clone().requires_grad_()
    got = unit_skill(predicted, alpha, 0.5)
    want = unit_oracle(predicted, alpha, 0.5)
    assert got.shape == (4, 3) and got.dtype == F64
    assert not got.requires_grad and got.grad_fn is None
    assert_close(got, want)
    assert_close(got[1], predicted[1].detach() / 0.5)
    weighted_norm = torch.linalg.vector_norm(got * alpha, dim=-1)
    assert_close(weighted_norm[[0, 3]], torch.ones(2, dtype=F64))
    assert weighted_norm[1] < 1.0 and weighted_norm[2] == 0.0
    assert predicted.grad is None and alpha.grad is None


def test_metra_weighted_inner_product_and_both_constraint_branches():
    log_lambda = torch.tensor(math.log(2.0), dtype=F64)
    got = metra_loss(ACTUAL, SKILL, ALPHA, log_lambda, 0.1)
    # Row weighted dots are 0.1 and 2.375; q values are slack=0.1 and 1-d²=-6.25.
    want = torch.tensor(
        [
            -0.1 - 2.0 * 0.1 + math.log(2.0) * 0.1,
            -2.375 - 2.0 * (-6.25) + math.log(2.0) * (-6.25),
        ],
        dtype=F64,
    )
    assert got.shape == (2,) and got.dtype == F64
    assert_close(got, want)
    assert_close(got, metra_oracle(ACTUAL, SKILL, ALPHA, log_lambda, 0.1))


def test_metra_gradients_reach_actual_and_dual_but_not_skill():
    actual = ACTUAL.clone().requires_grad_()
    skill = SKILL.clone().requires_grad_()
    log_lambda = torch.tensor(math.log(2.0), dtype=F64, requires_grad=True)
    metra_loss(actual, skill, ALPHA, log_lambda, 0.1).sum().backward()
    want_actual = -ALPHA.square() * SKILL
    # Only row 1 has 1-d² < slack; lambda is detached in the constraint term.
    want_actual[1] += 2.0 * 2.0 * ALPHA.square() * ACTUAL[1]
    assert_close(actual.grad, want_actual)
    assert_close(log_lambda.grad, torch.tensor(0.1 - 6.25, dtype=F64))
    assert skill.grad is None


def test_exact_clamp_tie_has_zero_constraint_gradient_and_dual_gradient_slack():
    actual = torch.tensor([[0.5, 0.5]], dtype=F64, requires_grad=True)
    skill = torch.tensor([[1.0, -1.0]], dtype=F64)
    alpha = torch.ones(2, dtype=F64)
    log_lambda = torch.tensor(0.0, dtype=F64, requires_grad=True)
    got = metra_loss(actual, skill, alpha, log_lambda, 0.5)
    assert_close(got, torch.tensor([-0.5], dtype=F64))
    got.backward()
    assert_close(actual.grad, torch.tensor([[-1.0, 1.0]], dtype=F64))
    assert_close(log_lambda.grad, torch.tensor(0.5, dtype=F64))


def test_module_initialises_exactly_one_scalar_float32_dual():
    loss = MetraLoss(config())
    parameters = list(loss.parameters())
    assert len(parameters) == 1 and parameters[0] is loss.log_lambda
    assert loss.log_lambda.shape == () and loss.log_lambda.dtype == torch.float32
    assert loss.log_lambda.requires_grad
    assert_close(loss.log_lambda, torch.tensor(math.log(2.5)))
    assert set(loss.state_dict()) == {"log_lambda"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_module_forward_matches_equations_and_detaches_predicted_skill(dtype):
    loss = MetraLoss(config())
    predicted = torch.tensor(
        [[0.3, 0.4, 0.0], [0.1, 0.2, 0.1]], dtype=dtype, requires_grad=True
    )
    actual = ACTUAL.to(dtype).requires_grad_()
    alpha = ALPHA.to(dtype)
    got = loss(actual, predicted, alpha)
    want = metra_oracle(
        actual,
        unit_oracle(predicted, alpha, 0.5),
        alpha,
        loss.log_lambda.to(dtype),
        0.1,
    )
    tolerance = {"rtol": 1e-6, "atol": 1e-6} if dtype == torch.float32 else {}
    assert got.shape == (2,) and got.dtype == dtype
    assert_close(got, want, **tolerance)
    got.sum().backward()
    assert predicted.grad is None
    assert actual.grad is not None and actual.grad.abs().sum() > 0
    assert loss.log_lambda.grad is not None
    assert_close(loss.log_lambda.grad, torch.tensor(-6.15), **tolerance)


def test_metra_dual_group_and_state_dict_round_trip():
    source = MetraLoss(config())
    groups = source.groups()
    assert set(groups) == {"metra_dual"}
    assert len(groups["metra_dual"]) == 1
    assert groups["metra_dual"][0] is source.log_lambda
    with torch.no_grad():
        source.log_lambda.fill_(math.log(7.0))
    target = MetraLoss(config(dual_init=9.0))
    target.load_state_dict(source.state_dict())
    assert_close(target.log_lambda, torch.tensor(math.log(7.0)))
    assert set(target.state_dict()) == {"log_lambda"}
