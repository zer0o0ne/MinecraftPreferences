"""Functional Qwen values, untruncated gradients and checkpoint replay oracles."""

import copy

import pytest
import torch
from models.perception.world import backbone_config, runs_backbone
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.state import ContinuousState, detach_state

pytestmark = runs_backbone
F32 = {"rtol": 3e-4, "atol": 3e-5}


def text_model(dtype=torch.float32, device="cpu"):
    config = backbone_config().text_config
    config.num_hidden_layers = 4
    config.layer_types = ["linear_attention", "full_attention"] * 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18)
        return Qwen3_5TextModel(config).to(device=device, dtype=dtype)


def inputs(device="cpu", dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(71)
    return [
        torch.randn(1, length, 32, generator=generator, device=device, dtype=dtype)
        for length in (5, 7, 4)
    ]


def position(x):
    return torch.arange(x.shape[1], device=x.device).view(1, 1, -1).expand(3, 1, -1)


def unroll(model, xs):
    state, outputs = (), []
    for x in xs:
        out, state = text_step(model, x, position(x), state)
        outputs.append(out)
    return torch.cat(outputs, dim=1), state


def reference(model, xs):
    x = torch.cat(xs, dim=1)
    labels = torch.cat(
        [torch.full((v.shape[1],), i, device=x.device) for i, v in enumerate(xs)]
    )
    valid = (labels[:, None] == labels[None, :]).tril()
    mask = torch.zeros_like(valid, dtype=x.dtype).masked_fill(~valid, -torch.inf)
    return model(
        inputs_embeds=x,
        position_ids=torch.cat([position(v) for v in xs], dim=-1),
        attention_mask={"full_attention": mask[None, None], "linear_attention": None},
        use_cache=False,
    ).last_hidden_state


def gradients(model, xs, output):
    probe = (
        torch.arange(output.numel(), device=output.device)
        .reshape_as(output)
        .remainder(13)
    )
    return torch.autograd.grad(
        (output * probe).sum(), (*xs, *model.parameters()), allow_unused=True
    )


def assert_gradients(actual, expected, **tolerance):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, **tolerance)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_values_and_all_gradients_match_full_masked_sequence(checkpointed):
    model = text_model().train()
    model.config._attn_implementation = "eager"
    oracle = copy.deepcopy(model)
    if checkpointed:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    xs = [x.requires_grad_() for x in inputs()]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    actual, state = unroll(model, xs)
    expected = reference(oracle, ys)
    torch.testing.assert_close(actual, expected, **F32)
    assert_gradients(
        gradients(model, xs, actual), gradients(oracle, ys, expected), **F32
    )
    assert len(state) == 2
    assert state[0].conv.shape == (1, 48, 4)
    assert state[0].recurrent.shape == (1, 2, 8, 8)


def test_checkpoint_replay_keeps_dropout_rng_and_gradients():
    model = text_model().train()
    model.layers[1].self_attn.attention_dropout = 0.3
    oracle = copy.deepcopy(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    xs = [x.requires_grad_() for x in inputs()]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    rng = torch.get_rng_state()
    actual, _ = unroll(model, xs)
    grads = gradients(model, xs, actual)
    torch.set_rng_state(rng)
    expected, _ = unroll(oracle, ys)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_gradients(grads, gradients(oracle, ys, expected), rtol=0, atol=0)


def test_detach_cuts_gradient_but_preserves_values_and_previous_storage():
    model = text_model().train()
    xs = [x.requires_grad_() for x in inputs()]
    _, memory = text_step(model, xs[0], position(xs[0]), ())
    saved = [(s.conv.clone(), s.recurrent.clone()) for s in memory]
    detached = detach_state(ContinuousState(memory, {"xpos": 2.0}), device="cpu")
    output, next_memory = text_step(model, xs[1], position(xs[1]), detached.layers)
    prior, current = torch.autograd.grad(output.sum(), xs[:2], allow_unused=True)
    assert prior is None and current is not None and current.abs().sum() > 0
    for state, (conv, recurrent) in zip(memory, saved):
        torch.testing.assert_close(state.conv, conv, rtol=0, atol=0)
        torch.testing.assert_close(state.recurrent, recurrent, rtol=0, atol=0)
    with torch.no_grad():
        expected, _ = text_step(model, xs[1], position(xs[1]), memory)
        reset, _ = text_step(model, xs[1], position(xs[1]), ())
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not torch.allclose(output, reset)
    for state in next_memory:
        assert (
            state.conv.untyped_storage().nbytes()
            == state.conv.numel() * state.conv.element_size()
        )
