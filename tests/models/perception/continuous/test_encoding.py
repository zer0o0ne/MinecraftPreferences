"""Prepared inputs and split autodiff match the original observation execution."""

import copy

import pytest
import torch
from models.perception.continuous.test_gpu import assert_fla
from models.perception.continuous.test_observation import world
from models.perception.world import runs_backbone

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.observation import decode, encode, prepare
from mcpref.models.perception.continuous.state import ContinuousState
from mcpref.models.perception.meta import meta_text
from mcpref.models.perception.tokens import image_inputs, positions, segment


def original(phi, frame, line, state):
    """Original direct composition, including HF's uncached vision metadata."""
    origin = line if state is None else state.origin
    pixels, grid = image_inputs(phi.processor, [frame])
    ids = segment(
        phi.tokenizer, phi.sp, grid[0], meta_text(phi.cfg, line, origin), phi.cfg.marker
    )
    device = phi.head[0].weight.device
    tokens = torch.tensor([ids], device=device)
    pos = positions(phi.backbone, ids, grid, phi.sp).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        embeds = phi.backbone.get_input_embeddings()(tokens)
        features = phi.backbone.get_image_features(
            pixels.to(device), grid.to(device), return_dict=True
        )
        images = torch.cat(features.pooler_output).to(embeds)
        mask, _ = phi.backbone.get_placeholder_mask(
            tokens, inputs_embeds=embeds, image_features=images
        )
        hidden, states = text_step(
            phi.backbone.language_model,
            embeds.masked_scatter(mask, images),
            pos,
            () if state is None else state.layers,
        )
        return phi.head(hidden[0, -1]), ContinuousState(states, origin)


def execute(phi, frames, lines, split):
    state, outputs, frontends = None, [], []
    for frame, line in zip(frames[:2], lines[:2]):
        if split is None:
            z, state = original(phi, frame, line, state)
        else:
            prepared = prepare(phi, frame, line, lines[0])
            embeddings = encode(phi, prepared)
            leaf = embeddings.detach().requires_grad_() if split else embeddings
            z, state = decode(phi, leaf, prepared, state)
            if split:
                frontends.append((embeddings, leaf))
        outputs.append(z)
    output = torch.stack(outputs)
    (output.square().sum() + output[0].sin().sum()).backward()
    for embeddings, leaf in frontends:
        embeddings.backward(leaf.grad)
    return output, state


def assert_original(tmp_path, checkpointed, dtype, split, device):
    phi, frames, lines = world(tmp_path)
    phi = phi.to(device=device, dtype=dtype).train()
    phi.backbone.language_model.layers[1].self_attn.attention_dropout = 0.3
    if not checkpointed:
        phi.backbone.gradient_checkpointing_disable()
    reference = copy.deepcopy(phi)
    prepared = prepare(phi, frames[0], lines[0], lines[0])
    assert prepared.grid.device.type == "cpu"
    assert prepared.vision["image_cu_seqlens"].device.type == "cpu"
    assert prepared.tokens.device.type == device
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    actual, memory = execute(phi, frames, lines, split)
    after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    expected, expected_memory = execute(reference, frames, lines, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert memory.origin == expected_memory.origin == lines[0]
    assert torch.equal(after, torch.get_rng_state())
    if cuda_after is not None:
        assert torch.equal(cuda_after, torch.cuda.get_rng_state())
    for a, b in zip(memory.layers, expected_memory.layers):
        torch.testing.assert_close(a.conv, b.conv, rtol=0, atol=0)
        torch.testing.assert_close(a.recurrent, b.recurrent, rtol=0, atol=0)
    actual_gradients, expected_gradients = [], []
    for (name, a), (_, b) in zip(phi.named_parameters(), reference.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            if device != "cuda" or not split:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
            actual_gradients.append(a.grad)
            expected_gradients.append(b.grad)
    if device == "cuda" and split:
        from training.pretrain.continuous.test_graphs_qwen import (
            assert_gradient_direction,
        )

        assert_gradient_direction(actual_gradients, expected_gradients, 0.01)


@runs_backbone
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("split", [False, True])
def test_values_gradients_rng_and_deferred_encoder_match_original(
    tmp_path, checkpointed, dtype, split
):
    assert_original(tmp_path, checkpointed, dtype, split, "cpu")


@pytest.mark.gpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("split", [False, True])
def test_cuda_cpu_metadata_and_split_encoder_match_original(
    tmp_path, checkpointed, dtype, split
):
    assert_fla()
    with torch.random.fork_rng(devices=["cuda"]):
        torch.manual_seed(173)
        assert_original(tmp_path, checkpointed, dtype, split, "cuda")
    torch.cuda.synchronize()


@runs_backbone
def test_prepared_metadata_reusable_after_parameter_update(tmp_path):
    phi, frames, lines = world(tmp_path)
    phi.eval()
    rng = torch.get_rng_state()
    prepared = prepare(phi, frames[0], lines[0], lines[0])
    assert torch.equal(rng, torch.get_rng_state())
    assert prepared.grid.device.type == "cpu"
    assert prepared.vision["image_cu_seqlens"].device.type == "cpu"
    with torch.no_grad():
        before, _ = decode(phi, encode(phi, prepared), prepared)
        phi.backbone.visual.pos_embed.weight[:, 0].add_(0.25)
        actual, _ = decode(phi, encode(phi, prepared), prepared)
        expected, _ = original(phi, frames[0], lines[0], None)
    assert not torch.equal(before, actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@runs_backbone
def test_shape_validation_matches_original_without_scalar_reads(tmp_path, monkeypatch):
    phi, frames, lines = world(tmp_path)
    prepared = prepare(phi, frames[0], lines[0], lines[0])
    get_features = phi.backbone.get_image_features

    def truncated(*args, **kwargs):
        features = get_features(*args, **kwargs)
        features.pooler_output = (features.pooler_output[0][:-1],)
        return features

    monkeypatch.setattr(phi.backbone, "get_image_features", truncated)
    message = "Image features and image tokens do not match, tokens: 2, features: 1"
    with pytest.raises(ValueError, match=message):
        original(phi, frames[0], lines[0], None)
    with pytest.raises(ValueError, match=message):
        encode(phi, prepared)
