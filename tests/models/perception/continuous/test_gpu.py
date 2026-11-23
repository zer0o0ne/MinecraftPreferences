"""Run on Spark: pytest this file -m gpu -n 0; real Qwen, FLA and checkpoint replay."""

import copy
import importlib
import inspect

import pytest
import torch
from models.perception.continuous.test_functional import (
    assert_gradients,
    gradients,
    inputs,
    position,
    reference,
    text_model,
    unroll,
)
from models.perception.continuous.test_qwen_replay import assert_sliding

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.state import ContinuousState, detach_state

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(180)]
BF16 = {"rtol": 1.6e-2, "atol": 1.6e-2}


def assert_fla():
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    fla = importlib.import_module("fla.ops.gated_delta_rule")
    chosen = inspect.getclosurevars(module.torch_chunk_gated_delta_rule).nonlocals[
        "implementation"
    ]
    assert chosen is fla.chunk_gated_delta_rule


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_sliding_replay_crosses_optimizer_updates(dtype):
    assert_fla()
    assert_sliding("cuda", dtype, autocast=dtype == torch.float32)
    torch.cuda.synchronize()


def test_cuda_fla_values_and_checkpointed_k_window_gradients():
    assert_fla()
    model = text_model(torch.bfloat16, "cuda").train()
    oracle = copy.deepcopy(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    xs = [x.requires_grad_() for x in inputs("cuda", torch.bfloat16)]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    actual, _ = unroll(model, xs)
    earlier = torch.autograd.grad(
        actual[:, -xs[-1].shape[1] :].float().sum(), xs[0], retain_graph=True
    )[0]
    assert torch.isfinite(earlier).all() and earlier.abs().sum() > 0
    with torch.no_grad():
        expected = reference(oracle, ys)
    torch.testing.assert_close(actual, expected, **BF16)
    expected, _ = unroll(oracle, ys)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_gradients(
        gradients(model, xs, actual), gradients(oracle, ys, expected), rtol=0, atol=0
    )


def test_fla_all_derivatives_including_initial_state_match_independent_recurrence():
    """Kernel error is checked before BF16 network gradients accumulate roundoff."""
    assert_fla()
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    fla = importlib.import_module("fla.ops.gated_delta_rule")
    generator = torch.Generator(device="cuda").manual_seed(44)
    values = [
        torch.randn(
            1, 11, 2, 8, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(3)
    ]
    values += [-torch.rand(1, 11, 2, generator=generator, device="cuda")]
    values += [
        torch.rand(1, 11, 2, generator=generator, device="cuda", dtype=torch.bfloat16)
    ]
    values += [torch.randn(1, 2, 8, 8, generator=generator, device="cuda")]
    results = []
    for kernel in (
        fla.chunk_gated_delta_rule,
        module.torch_chunk_gated_delta_rule.__wrapped__,
    ):
        q, k, v, g, beta, state = [value.clone().requires_grad_() for value in values]
        out, final = kernel(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        grads = torch.autograd.grad(
            out.float().sum() + final.sum(), (q, k, v, g, beta, state)
        )
        assert grads[-1].norm() > 0
        results.append((out, final, *grads))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, **BF16)


@pytest.mark.parametrize("window", [1, 2])
def test_cuda_fla_tbptt_update_keeps_values_and_cuts_previous_gradient(window):
    assert_fla()
    model = text_model(torch.bfloat16, "cuda").train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    chunks, state = inputs("cuda", torch.bfloat16), ()
    losses = []
    for i, x in enumerate(chunks):
        output, state = text_step(model, x, position(x), state)
        losses.append(output.float().square().mean())
        if (i + 1) % window == 0 or i == len(chunks) - 1:
            sum(losses).backward()
            assert all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            state = detach_state(ContinuousState(state, {})).layers
            assert all(
                not s.conv.requires_grad and not s.recurrent.requires_grad
                for s in state
            )
    torch.cuda.synchronize()
