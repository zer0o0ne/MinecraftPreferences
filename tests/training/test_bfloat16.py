"""Fresh BF16 pretrain/RL steps, diagnostics and checkpoint continuation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diagnostics.core.world import config as diagnostic_config
from diagnostics.core.world import context
from models.perception.world import (
    backbone,
    config,
    random_clip,
    runs_backbone,
    tokenizer,
)
from models.policy.policy_oracle import cfg_of as policy_config
from models.preference.preference_oracle import cfg_of as preference_config
from torchao.float8.float8_linear import Float8Linear

from mcpref.diagnostics.producers.policy import on_ppo_observation, on_shift_observation
from mcpref.models.factory import build_modules
from mcpref.models.perception.stream import PhiStream
from mcpref.runtime.mcts.tree import LatentHistory
from mcpref.runtime.sampling.models import decode_stream, encode_stream
from mcpref.training.groups import check_groups
from mcpref.training.optim import adamw, update
from mcpref.training.pretrain.checkpoints import restore, write_checkpoint
from mcpref.training.pretrain.steps.loss import gradient_steps, step_gradients
from mcpref.training.rl import cycle
from mcpref.training.rl.learning import loss as rl_loss
from training.pretrain.steps.test_loss import CFG
from training.rl.learning.loss_fakes import ppo_config

torch.set_num_threads(1)


@pytest.fixture(params=["bf16", "fp8-mlp", "fp8-aot"])
def setup(tmp_path, masks, request):
    directory = tmp_path / "model"
    original = backbone(torch.float32)
    original.save_pretrained(directory)
    tokenizer().save_pretrained(directory)
    use_fp8 = request.param != "bf16"
    perception = config(tmp_path, model=str(directory), fp8_mlp=use_fp8)
    policy, preference = policy_config(history=3), preference_config(history=3)
    modules = build_modules(perception, masks, policy, preference, True, "cpu", 1.0)
    converted = [
        m for m in modules["phi"].backbone.modules() if isinstance(m, Float8Linear)
    ]
    assert len(converted) == (6 if use_fp8 else 0)
    if request.param == "fp8-aot":
        for layer in modules["phi"].backbone.language_model.layers:
            layer.mlp.compile(backend="aot_eager", fullgraph=True, dynamic=True)
    for name, value in original.named_buffers():
        got = dict(modules["phi"].backbone.named_buffers())[name]
        torch.testing.assert_close(got, value, rtol=0, atol=0)
    clips = [random_clip(i, [0, 1, 2], [False] * 3) for i in range(2)]
    for clip in clips:
        clip.info.update(
            {
                "relpath": "fixture/episode",
                "actions": [["forward"], ["jump"]],
                "n": [1, 1],
                "classes": [None] * 3,
                "pitch": [0.0] * 3,
            }
        )
    return modules, perception, policy, preference, clips


def optimizer(modules):
    declared = {
        name: parameters
        for module in modules.values()
        for name, parameters in module.groups().items()
    }
    cfg = replace(
        CFG,
        device="cpu",
        grad_clip=1.0,
        warmup_steps=1,
        groups={name: {"lr": 0.01, "weight_decay": 0.01} for name in declared},
    )
    groups = check_groups(cfg, modules)
    return cfg, groups, adamw(cfg, groups)


def assert_training_storage(modules, opt):
    parameters = [p for module in modules.values() for p in module.parameters()]
    assert parameters and all(p.dtype == torch.bfloat16 for p in parameters)
    gradients = [p.grad for p in parameters if p.grad is not None]
    assert gradients and all(g.dtype == torch.bfloat16 for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)
    for module in modules.values():
        assert any(p.grad is not None for p in module.parameters()) or not list(
            module.parameters()
        )
    for state in opt.state.values():
        assert state["step"].dtype == torch.float32
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].dtype == torch.bfloat16
            assert torch.isfinite(state[key]).all()


@runs_backbone
def test_pretrain_step_checkpoint_and_stream(setup, tmp_path):
    modules, _, _, _, clips = setup
    cfg, groups, opt = optimizer(modules)
    items = [{"a": "0", "b": "1", "labels": {"p": 0.75, "inc": 0}}]

    def step():
        opt.zero_grad(set_to_none=True)
        return step_gradients(
            items, modules, lambda p: clips[int(p)], 1, 2, 1, 0.3, 1.0, 1.0, 1, cfg=cfg
        )

    step()
    opt.zero_grad(set_to_none=True)
    gradient_steps(
        items,
        modules,
        lambda p: clips[int(p)],
        1,
        2,
        1,
        0.3,
        1.0,
        1.0,
        1,
        cfg,
        (groups, tmp_path / "gradient-work", None),
    )
    update(cfg, opt, groups, 1, 3)
    assert_training_storage(modules, opt)
    path = tmp_path / "checkpoints"
    write_checkpoint(
        path, 0, {"step": 1}, modules, opt, torch.device("cpu"), [], [], []
    )
    rng = torch.get_rng_state()
    step()
    update(cfg, opt, groups, 2, 3)
    expected = {
        name: {key: value.clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }
    saved = restore(path, 1, modules)
    opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(rng)
    step()
    update(cfg, opt, groups, 2, 3)
    assert_training_storage(modules, opt)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)
    phi = modules["phi"].eval()
    frames, _ = phi.prepare(clips[:1])[0]
    stream = PhiStream(phi, 3)
    with torch.no_grad():
        z = stream.step(frames[0], clips[0].lines[0])
        assert z.dtype == torch.bfloat16 and torch.isfinite(z).all()
        restored = decode_stream(phi, 3, encode_stream(stream), torch.device("cpu"))
        expected = stream.step(frames[1], clips[0].lines[1])
        actual = restored.step(frames[1], clips[0].lines[1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@runs_backbone
def test_rl_builder_and_all_roles(setup, masks, monkeypatch):
    modules, perception, policy, preference, clips = setup
    pretrain = SimpleNamespace(
        perception="p",
        action_mask="a",
        policy="pi",
        preference="P",
        latent_init_scale=1.0,
    )
    monkeypatch.setattr(cycle, "load_pretrain_config", lambda _: pretrain)
    for name, value in [
        ("load_perception_config", perception),
        ("load_mask_config", masks),
        ("load_masks", masks),
        ("load_policy_config", policy),
        ("load_preference_config", preference),
    ]:
        monkeypatch.setattr(cycle, name, lambda _, value=value: value)
    monkeypatch.setattr(cycle, "build_modules", lambda *args: modules)
    sampling = SimpleNamespace(pretrain="pretrain", steps=4, out_dir="data/sampling")
    mods = cycle.build_rl_modules(SimpleNamespace(device="cpu"), sampling, ppo_config())
    cfg, groups, opt = optimizer(mods)

    def encode(refs, phi, *args):
        z = phi([clips[ref.index % 2] for ref in refs])
        c = z.new_zeros((len(refs), 3, len(masks.names) + 1))
        c[..., -1] = 1
        return [LatentHistory(y.unsqueeze(0), a.unsqueeze(0)) for y, a in zip(z, c)]

    monkeypatch.setattr("mcpref.training.rl.learning.encoding.encode_refs", encode)
    monkeypatch.setattr(
        rl_loss,
        "load_step",
        lambda *args: SimpleNamespace(
            action=0,
            probabilities=torch.full((len(masks.names),), 1 / len(masks.names)),
        ),
    )
    items = [
        {"kind": "fresh", "index": 0},
        {"kind": "pending", "index": 1},
        {
            "kind": "preference",
            "a": {"episode": "a", "index": 0},
            "b": {"episode": "b", "index": 1},
            "labels": {"p": 0.75, "inc": 0},
        },
    ]
    observations = {
        key: [] for key in ("ppo", "shift", "ppo_clusters", "shift_clusters", "logits")
    }
    result = rl_loss.step_gradients(
        items,
        mods,
        sampling=sampling,
        perception=perception,
        states=3,
        alpha=torch.tensor(perception.alpha, dtype=torch.bfloat16),
        ppo=ppo_config(),
        discount=0.8,
        min_p=0.1,
        m=3,
        n_a=2,
        n_delta=1,
        r_c=1,
        w_ppo=1.0,
        w_delta=1.0,
        w_metra=1.0,
        w_P=1.0,
        micro_batch=4,
        observations=observations,
    )
    assert all(value is not None for value in result.values())
    update(cfg, opt, groups, 1, 3)
    assert_training_storage(mods, opt)
    for key, producer in [("ppo", on_ppo_observation), ("shift", on_shift_observation)]:
        for observation in observations[key]:
            record = producer(
                observation, cfg=diagnostic_config(), context=context(clusters=("a",))
            )
            assert record["metrics"]
