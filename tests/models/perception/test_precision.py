"""FP8 scope, exact rounding/gradient oracles, and real Qwen compatibility."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from torchao.float8.float8_linear import Float8Linear

from mcpref.models.perception.precision import enable_fp8_mlp
from models.perception.world import backbone


def test_only_text_mlp_changes_and_parameters_rng_state_are_preserved():
    model = backbone(torch.bfloat16).eval()
    parameters = dict(model.named_parameters())
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    enable_fp8_mlp(model)
    expected = {
        f"language_model.layers.{i}.mlp.{name}"
        for i in range(2)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    assert {
        name
        for name, module in model.named_modules()
        if isinstance(module, Float8Linear)
    } == expected
    assert torch.equal(rng, torch.get_rng_state())
    assert all(not module.training for module in model.modules())
    for name, value in model.named_parameters():
        assert value is parameters[name] and value.dtype == torch.bfloat16
    assert model.state_dict().keys() == state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot-autograd"])
def test_fp8_rounding_keeps_weight_gradient_high_precision(compiled):
    model = backbone(torch.bfloat16)
    enable_fp8_mlp(model)
    linear = model.language_model.layers[0].mlp.up_proj
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, :2] = 1
    if compiled:
        linear.compile(backend="aot_eager", fullgraph=True, dynamic=True)
    x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
    x[..., :2] = torch.tensor([1.0625, 1.125], dtype=torch.bfloat16)
    x.requires_grad_()
    y = linear(x)
    # Scale 256: 272 is halfway between E4M3 values 256 and 288, rounds to 256.
    assert y[0, 0, 0] == 2.125
    g_x, g_w = torch.autograd.grad(y.sum(), (x, linear.weight), retain_graph=True)
    assert torch.equal(g_x[..., :2], torch.ones_like(g_x[..., :2]))
    assert torch.equal(g_x[..., 2:], torch.zeros_like(g_x[..., 2:]))
    # Original BF16 x, not the quantized 1.0, enters the weight-gradient GEMM.
    torch.testing.assert_close(
        g_w, x.detach().reshape(1, 32).expand(64, 32), rtol=0, atol=0
    )
    again = torch.autograd.grad(y.sum(), (x, linear.weight))
    for a, b in zip(again, (g_x, g_w)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [3, 17, 29])
def test_mlp_outputs_and_input_gradients_track_bf16(seed):
    model = backbone(torch.bfloat16, seed=seed)
    reference = copy.deepcopy(model.language_model.layers[0].mlp)
    enable_fp8_mlp(model)
    candidate = model.language_model.layers[0].mlp
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 17, 32, generator=generator, dtype=torch.bfloat16)
    xs = [x.clone().requires_grad_(), x.clone().requires_grad_()]
    outputs = [module(inp) for module, inp in zip((reference, candidate), xs)]
    grads = [
        torch.autograd.grad(y.float().square().sum(), inp)[0]
        for y, inp in zip(outputs, xs)
    ]
    for a, b in (outputs, grads):
        # Smoke-test error budget, not a guarantee of long-run convergence.
        relative = (a.float() - b.float()).norm() / a.float().norm()
        assert relative < 0.10
        assert torch.isfinite(b).all()


def test_bad_projection_shape_fails_before_training():
    model = backbone(torch.bfloat16)
    model.language_model.layers[0].mlp.up_proj = torch.nn.Linear(32, 63)
    with pytest.raises(ValueError, match="divisible by 16"):
        enable_fp8_mlp(model)


@pytest.mark.gpu
@pytest.mark.timeout(180)
def test_native_cuda_compiled_mlp_checkpoint_diagnostics_and_weight_reload():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = backbone(torch.bfloat16).to("cuda")
    enable_fp8_mlp(model)
    mlp = model.language_model.layers[0].mlp
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.01, fused=True)
    for states in (17, 31, 1):
        x = torch.randn(
            1, states, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        optimizer.zero_grad(set_to_none=True)
        with (
            torch.compiler.set_stance("force_eager") if states == 31 else nullcontext()
        ):
            y = checkpoint(mlp, x, use_reentrant=False)
            loss = y.float().square().mean()
            if states == 31:
                gradients = torch.autograd.grad(
                    loss, tuple(mlp.parameters()), retain_graph=True
                )
                assert all(torch.isfinite(g).all() for g in gradients)
            loss.backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        optimizer.step()
    state = copy.deepcopy(mlp.state_dict())
    restored = backbone(torch.bfloat16).to("cuda")
    enable_fp8_mlp(restored)
    restored.language_model.layers[0].mlp.load_state_dict(state)
    torch.testing.assert_close(
        restored.language_model.layers[0].mlp(x), mlp(x), rtol=0, atol=0
    )
