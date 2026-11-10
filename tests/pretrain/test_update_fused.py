"""PT-3: complete CUDA AdamW steps and legacy/fused checkpoint resume."""

import dataclasses
import io

import pytest
import torch
from torch.testing import assert_close

from mcpref.pretrain.checks import Groups
from mcpref.pretrain.update import adamw, update

from .test_loss import CFG


def case(device: str) -> tuple[object, Groups, list[torch.nn.Parameter]]:
    cfg = dataclasses.replace(
        CFG,
        device=device,
        grad_clip=0.05,
        warmup_steps=1,
        groups={
            "decay": {"lr": 0.01, "weight_decay": 0.1},
            "free": {"lr": 0.02, "weight_decay": 0.0},
        },
    )
    parameters = [
        torch.nn.Parameter(torch.linspace(-0.5, 0.75, 6, device=device).reshape(2, 3)),
        torch.nn.Parameter(torch.tensor([0.25, -0.5], device=device)),
        torch.nn.Parameter(torch.tensor(0.75, device=device)),
        torch.nn.Parameter(torch.tensor([1.0, -1.0], device=device)),
    ]
    groups = Groups()
    groups.update({"decay": parameters[:2], "free": parameters[2:]})
    groups.parameter_names = {id(p): f"m.{i}" for i, p in enumerate(parameters)}
    return cfg, groups, parameters


def reference(cfg, groups: Groups) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [{"params": ps} | cfg.groups[q] for q, ps in groups.items()],
        betas=cfg.betas,
        eps=cfg.adam_eps,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    )


def gradients(parameters: list[torch.nn.Parameter], second: bool) -> None:
    values = [
        torch.tensor([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]).reshape(2, 3),
        None if second else torch.tensor([-0.4, 0.2]),
        torch.zeros(()) if second else torch.tensor(0.5),
        None,
    ]
    for parameter, value in zip(parameters, values):
        parameter.grad = (
            None if value is None else value.to(parameter.device) + second * 0.05
        )


def rng_state() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def restore_rng(state: tuple[torch.Tensor, torch.Tensor]) -> None:
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def assert_update_equal(
    expected_parameters: list[torch.nn.Parameter],
    expected: torch.optim.AdamW,
    actual_parameters: list[torch.nn.Parameter],
    actual: torch.optim.AdamW,
) -> None:
    for wanted, got in zip(expected_parameters, actual_parameters):
        assert_close(got, wanted, rtol=2e-6, atol=2e-7)
        assert (got.grad is None) == (wanted.grad is None)
        if wanted.grad is not None:
            assert_close(got.grad, wanted.grad, rtol=2e-6, atol=2e-7)
    wanted_states = expected.state_dict()["state"]
    got_states = actual.state_dict()["state"]
    assert wanted_states.keys() == got_states.keys()
    for index, wanted in wanted_states.items():
        got = got_states[index]
        assert wanted.keys() == got.keys()
        assert got["step"].cpu().equal(wanted["step"].cpu())
        assert_close(
            got["exp_avg"].cpu(), wanted["exp_avg"].cpu(), rtol=2e-6, atol=2e-7
        )
        assert_close(
            got["exp_avg_sq"].cpu(), wanted["exp_avg_sq"].cpu(), rtol=2e-6, atol=2e-7
        )


def checkpoint(parameters, optimizer, rng) -> dict:
    stream = io.BytesIO()
    torch.save(
        {
            "parameters": [p.detach().cpu() for p in parameters],
            "optimizer": optimizer.state_dict(),
            "rng": rng,
        },
        stream,
    )
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=True)


def resumed(cfg, saved: dict) -> tuple[Groups, list, torch.optim.AdamW]:
    _, groups, parameters = case("cuda")
    with torch.no_grad():
        for parameter, value in zip(parameters, saved["parameters"]):
            parameter.copy_(value)
    optimizer = adamw(cfg, groups)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    assert all(group["fused"] for group in optimizer.param_groups)
    for parameter in parameters:
        if parameter in optimizer.state:
            assert optimizer.state[parameter]["step"].device == parameter.device
    return groups, parameters, optimizer


def test_cpu_factory_retains_nonfused_backend():
    cfg, groups, _ = case("cpu")
    optimizer = adamw(cfg, groups)
    assert all(
        group["foreach"] is False and group["fused"] is False
        for group in optimizer.param_groups
    )


@pytest.mark.gpu
def test_cuda_complete_step_and_checkpoint_resume_match_default_adamw():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cfg, groups, parameters = case("cuda")
    _, reference_groups, reference_parameters = case("cuda")
    expected = reference(cfg, reference_groups)
    actual = adamw(cfg, groups)
    assert all(group["fused"] for group in actual.param_groups)

    torch.manual_seed(19)
    initial_rng = rng_state()
    gradients(reference_parameters, second=False)
    expected_lr = update(cfg, expected, reference_groups, 2, 10)
    expected_rng = rng_state()
    restore_rng(initial_rng)
    gradients(parameters, second=False)
    actual_lr = update(cfg, actual, groups, 2, 10)
    actual_rng = rng_state()
    assert actual_lr == expected_lr
    assert all(a.equal(b) for a, b in zip(actual_rng, expected_rng))
    assert_update_equal(reference_parameters, expected, parameters, actual)
    idle = parameters[-1]
    assert idle.grad is None and idle not in actual.state

    legacy = checkpoint(reference_parameters, expected, expected_rng)
    fused = checkpoint(parameters, actual, actual_rng)
    gradients(reference_parameters, second=True)
    restore_rng(expected_rng)
    final_lr = update(cfg, expected, reference_groups, 3, 10)
    final_rng = rng_state()
    for saved in (legacy, fused):
        flags = [(g["foreach"], g["fused"]) for g in saved["optimizer"]["param_groups"]]
        restored_groups, restored_parameters, restored = resumed(cfg, saved)
        assert flags == [
            (g["foreach"], g["fused"]) for g in saved["optimizer"]["param_groups"]
        ]
        gradients(restored_parameters, second=True)
        restored_lr = update(cfg, restored, restored_groups, 3, 10)
        assert restored_lr == final_lr
        assert_update_equal(
            reference_parameters, expected, restored_parameters, restored
        )
        assert all(a.equal(b) for a, b in zip(rng_state(), final_rng))
        assert restored_parameters[1].grad is None
        assert_close(restored_parameters[-1], idle, rtol=0, atol=0)
