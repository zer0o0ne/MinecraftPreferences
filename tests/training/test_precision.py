"""Selected storage precision reaches loading, gradients and optimizer resume."""

import io
from dataclasses import replace

import pytest
import torch
from models.perception import oracle
from models.perception.world import (
    backbone,
    config,
    random_clip,
    runs_backbone,
    tokenizer,
)
from models.policy.policy_oracle import cfg_of as policy_config
from models.preference.preference_oracle import cfg_of as preference_config

from mcpref.models.factory import build_modules
from mcpref.models.perception.continuous import observation_step
from mcpref.training.groups import check_groups
from mcpref.training.optim import adamw, update
from training.pretrain.classical.steps.test_loss import CFG


@runs_backbone
@pytest.mark.parametrize(
    "precision,dtype", [("fp32", torch.float32), ("bf16", torch.bfloat16)]
)
@pytest.mark.parametrize("fp8", [False, True])
def test_storage_precision_preserves_source_and_small_updates(
    tmp_path, masks, precision, dtype, fp8
):
    source = backbone(torch.float32)
    with torch.no_grad():
        source.language_model.layers[0].linear_attn.A_log[0] = 0.1234567
    path = tmp_path / "source"
    source.save_pretrained(path)
    tokenizer().save_pretrained(path)
    modules = build_modules(
        config(tmp_path, model=str(path), fp8_mlp=fp8),
        masks,
        policy_config(history=3),
        preference_config(history=3),
        True,
        "cpu",
        1.0,
        precision,
    )
    for name, parameter in modules["phi"].backbone.named_parameters():
        torch.testing.assert_close(
            parameter, dict(source.named_parameters())[name].to(dtype), rtol=0, atol=0
        )
    for module in modules.values():
        assert all(p.dtype == dtype for p in module.parameters())
    declared = {
        name: params
        for module in modules.values()
        for name, params in module.groups().items()
    }
    cfg = replace(
        CFG,
        precision=precision,
        device="cpu",
        grad_clip=1.0,
        warmup_steps=1,
        groups={name: {"lr": 1e-5, "weight_decay": 0.0} for name in declared},
    )
    groups = check_groups(cfg, modules)
    optimizer = adamw(cfg, groups)
    clip = random_clip(13, [0], [False])
    frame, line = oracle.frames(clip)[0], oracle.state_line(clip, 0)
    dual = modules["phi_loss"].log_lambda
    before = dual.detach().clone()

    def step(number):
        z, _ = observation_step(modules["phi"], frame, line)
        assert z.dtype == dtype
        (z.square().sum() + dual.square()).backward()
        for params in groups.values():
            for p in params:
                assert p.grad is None or (
                    p.grad.dtype == dtype and torch.isfinite(p.grad).all()
                )
        update(cfg, optimizer, groups, number, 3)
        optimizer.zero_grad(set_to_none=True)

    step(1)
    assert bool(dual != before) == (precision == "fp32")
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == dtype
    saved = io.BytesIO()
    torch.save(
        {
            "modules": {k: m.state_dict() for k, m in modules.items()},
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        },
        saved,
    )
    step(2)
    expected = {
        k: {n: p.clone() for n, p in m.state_dict().items()} for k, m in modules.items()
    }
    saved.seek(0)
    restored = torch.load(saved, weights_only=True)
    for name, module in modules.items():
        module.load_state_dict(restored["modules"][name])
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["rng"])
    step(2)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)
