"""Equation (2): exact differentiable and reward-only encoding boundaries for H=4."""

import torch

from training.rl.learning.loss_fakes import (
    assert_gradient_close,
    expected,
    gradients,
    history,
    load_step,
    modules,
)
from training.rl.learning.test_loss import WEIGHTS, assert_result, call, fresh, pending


def observe_encoding(monkeypatch):
    records = {}

    def encoded(ref, phi, perception, sampling, states, action_count):
        saved = []

        def pack(tensor):
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            value = history(ref.index, phi)
        records[ref.index] = (
            torch.is_grad_enabled(),
            value.Y.requires_grad,
            value.Y.grad_fn is not None,
            len(saved),
        )
        return value

    monkeypatch.setattr(
        "mcpref.training.rl.learning.encoding.encode_refs",
        lambda refs, *args: [encoded(ref, *args) for ref in refs],
    )
    monkeypatch.setattr("mcpref.training.rl.learning.loss.load_step", load_step)
    return records


def assert_differentiable(records, indices):
    for index in indices:
        enabled, requires_grad, has_grad_fn, saved = records[index]
        assert enabled and requires_grad and has_grad_fn and saved > 0


def assert_reward_only(records, indices):
    for index in indices:
        enabled, requires_grad, has_grad_fn, saved = records[index]
        assert not enabled and not requires_grad and not has_grad_fn and saved == 0


def test_fresh_h4_reward_only_grad_modes_saved_tensors_and_float64_oracle(
    monkeypatch,
):
    records = observe_encoding(monkeypatch)
    actual = modules()
    result = call(
        [fresh(0)], actual, micro_batch=1, m=1, n_a=1, n_delta=1, r_c=0, horizon=4
    )
    assert set(records) == {0, 1, 2, 3, 4}
    assert_differentiable(records, [0, 1])
    assert_reward_only(records, [2, 3, 4])
    reference = modules()
    total, means = expected([fresh(0)], reference, 1, 1, 1, 0, WEIGHTS, horizon=4)
    total.backward()
    assert_result(result, total, means)
    assert_gradient_close(gradients(actual), gradients(reference))


def test_pending_h4_reward_only_grad_modes_saved_tensors_and_float64_oracle(
    monkeypatch,
):
    records = observe_encoding(monkeypatch)
    actual = modules()
    result = call(
        [pending(2)], actual, micro_batch=1, m=1, n_a=1, n_delta=0, r_c=0, horizon=4
    )
    assert set(records) == {2, 3, 4, 5, 6}
    assert_differentiable(records, [2])
    assert_reward_only(records, [3, 4, 5, 6])
    reference = modules()
    total, means = expected([pending(2)], reference, 1, 1, 0, 0, WEIGHTS, horizon=4)
    total.backward()
    assert_result(result, total, means)
    assert_gradient_close(gradients(actual), gradients(reference))
    assert actual["pi"].shift_calls == 0
    assert actual["metra"].log_lambda.grad is None
