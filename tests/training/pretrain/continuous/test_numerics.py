"""Attention VJPs repeat exactly; numerical runtime settings survive nested errors."""

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mcpref.training.pretrain.continuous.numerics import reproducible


def settings():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
        torch.backends.cudnn.benchmark,
    )


def test_nested_cuda_settings_restore_after_failure(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = settings()
    with pytest.raises(RuntimeError, match="interrupted"), reproducible("cuda"):
        assert settings() == (True, False, False, False)
        with reproducible("cuda"):
            assert settings() == (True, False, False, False)
        assert settings() == (True, False, False, False)
        raise RuntimeError("interrupted")
    assert settings() == before


def test_cpu_does_not_change_cuda_environment_or_flags(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    before = settings()
    with reproducible("cpu"):
        assert settings() == before
    with (
        pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"),
        reproducible("cuda"),
    ):
        pytest.fail("Invalid CUDA settings were accepted")
    assert settings() == before


@pytest.mark.gpu
@pytest.mark.timeout(60)
def test_repeated_cuda_attention_vjps_are_exact_and_match_fp32_math():
    with reproducible("cuda"):
        generator = torch.Generator(device="cuda").manual_seed(4242)
        values = tuple(
            torch.randn(1, 301, heads, 256, generator=generator, device="cuda")
            .to(torch.bfloat16)
            .transpose(1, 2)
            .requires_grad_()
            for heads in (8, 2, 2)
        )
        cotangent = torch.randn(
            values[0].shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        options = {
            "dropout_p": 0.0,
            "is_causal": True,
            "scale": 256**-0.5,
            "enable_gqa": True,
        }
        output = F.scaled_dot_product_attention(*values, **options)
        expected = torch.autograd.grad(output, values, cotangent, retain_graph=True)
        for _ in range(10):
            actual = torch.autograd.grad(output, values, cotangent, retain_graph=True)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
        oracle = tuple(x.detach().float().requires_grad_() for x in values)
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(*oracle, **options)
        target = torch.autograd.grad(reference, oracle, cotangent.float())
        for a, e in zip(expected, target):
            assert (a.float() - e).norm() / e.norm() < 0.01
