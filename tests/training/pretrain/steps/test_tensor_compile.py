"""PT-9b: ordinary CUDA tensor paths preserve one optimizer step and resume."""

import copy
import dataclasses
import io

import pytest
import torch
from models.perception.world import config as perception_config
from models.perception.world import line
from models.policy.policy_oracle import cfg_of as policy_config
from models.policy.policy_world import action_masks, random_clips
from models.preference.preference_oracle import cfg_of as preference_config

from mcpref.models.perception.cursor import load_sprite
from mcpref.models.perception.loss import PhiLoss, latent_metric
from mcpref.models.perception.meta import clip_states
from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from mcpref.models.preference.loss import PreferenceLoss
from mcpref.models.preference.network import Preference
from mcpref.training.pretrain.steps.micro_loss import micro_loss

FACTORS = (0.5, 0.03, 1.0, 1.0)
PAIRS = [(0, 1, {"p": 0.75, "inc": 0})]


class Latents(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.sprite = cfg, load_sprite(cfg)
        generator = torch.Generator().manual_seed(71)
        z = torch.randn(2, 5, len(cfg.alpha), generator=generator)
        alpha = torch.tensor(cfg.alpha)
        self.Z = torch.nn.Parameter(0.1 * z / alpha)

    def prepare(self, clips):
        return [clip_states(clip, self.cfg, self.sprite) for clip in clips]

    def forward(self, clips, states):
        return self.Z


def case(tmp_path):
    perception = perception_config(tmp_path, alpha=[1.0, 2.0, 4.0, 8.0])
    policy = dataclasses.replace(
        policy_config(),
        history=1,
        width=16,
        heads=4,
        hidden=32,
        layers=2,
        shift_hidden=32,
    )
    preference = dataclasses.replace(
        preference_config(),
        history=1,
        width=16,
        heads=4,
        hidden=32,
        layers=("S", "X"),
    )
    masks = action_masks(tmp_path / "masks")
    clips = random_clips(81, 2, 5)
    for clip in clips:
        for old in clip.lines:
            values = line(hotbar=old["hotbar"])
            old.clear()
            old.update(values)
    modules = {
        "phi": Latents(perception),
        "phi_loss": PhiLoss(perception),
        "pi": Policy(policy, len(perception.alpha), masks.names),
        "pi_loss": PolicyLoss(policy, masks, latent_metric(perception)),
        "P": Preference(preference, len(perception.alpha)),
        "P_loss": PreferenceLoss(),
    }
    return clips, {name: module.cuda() for name, module in modules.items()}


def optimizer_of(modules):
    parameters = [p for module in modules.values() for p in module.parameters()]
    return torch.optim.AdamW(
        parameters,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=True,
    )


def parameters_of(modules):
    return {
        f"{name}.{key}": parameter
        for name, module in modules.items()
        for key, parameter in module.named_parameters()
    }


def run(modules, optimizer, clips, compiled):
    optimizer.zero_grad(set_to_none=True)
    loss, result = micro_loss(modules, clips, PAIRS, FACTORS, compile_tensors=compiled)
    loss.backward()
    gradients = {
        key: None if parameter.grad is None else parameter.grad.detach().clone()
        for key, parameter in parameters_of(modules).items()
    }
    optimizer.step()
    return loss.detach(), result, gradients


def assert_close(got, want):
    if isinstance(got, torch.Tensor):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
    elif dataclasses.is_dataclass(got):
        got_values = {f.name: getattr(got, f.name) for f in dataclasses.fields(got)}
        want_values = {f.name: getattr(want, f.name) for f in dataclasses.fields(want)}
        assert_close(got_values, want_values)
    elif isinstance(got, dict):
        assert got.keys() == want.keys()
        for key in got:
            assert_close(got[key], want[key])
    elif isinstance(got, (tuple, list)):
        assert len(got) == len(want)
        for x, y in zip(got, want):
            assert_close(x, y)
    elif got is None or want is None:
        assert got is want
    else:
        assert got == want


def assert_step_close(got, want):
    for got_value, want_value in zip(got, want):
        assert_close(got_value, want_value)


def snapshot(modules, optimizer):
    stream = io.BytesIO()
    torch.save(
        {
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        stream,
    )
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=True)


@pytest.mark.gpu
@pytest.mark.timeout(120)
def test_compiled_step_checkpoint_and_resume(tmp_path):
    clips, eager = case(tmp_path)
    initial = copy.deepcopy(eager)
    compiled = copy.deepcopy(eager)
    keys = {name: list(module.state_dict()) for name, module in eager.items()}
    eager_optimizer, compiled_optimizer = optimizer_of(eager), optimizer_of(compiled)
    torch.manual_seed(91)
    torch.cuda.manual_seed_all(91)
    initial_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

    expected = run(eager, eager_optimizer, clips, compiled=False)
    expected_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    saved = snapshot(eager, eager_optimizer)
    torch.set_rng_state(initial_rng[0])
    torch.cuda.set_rng_state(initial_rng[1])
    actual = run(compiled, compiled_optimizer, clips, compiled=True)
    assert torch.equal(torch.get_rng_state(), expected_rng[0])
    assert torch.cuda.get_rng_state().equal(expected_rng[1])
    assert_step_close(actual, expected)
    assert {
        name: list(module.state_dict()) for name, module in compiled.items()
    } == keys
    assert_close(parameters_of(compiled), parameters_of(eager))
    assert_close(compiled_optimizer.state_dict(), eager_optimizer.state_dict())

    resumed = copy.deepcopy(initial)
    resumed_optimizer = optimizer_of(resumed)
    for name, module in resumed.items():
        module.load_state_dict(saved["modules"][name], strict=True)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    expected_second = run(eager, eager_optimizer, clips, compiled=False)
    expected_second_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    actual_second = run(resumed, resumed_optimizer, clips, compiled=True)
    assert_step_close(actual_second, expected_second)
    assert_close(parameters_of(resumed), parameters_of(eager))
    assert_close(resumed_optimizer.state_dict(), eager_optimizer.state_dict())
    assert torch.equal(torch.get_rng_state(), expected_second_rng[0])
    assert torch.cuda.get_rng_state().equal(expected_second_rng[1])
    assert {name: list(module.state_dict()) for name, module in resumed.items()} == keys
