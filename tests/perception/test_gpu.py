"""GPU tests of specs/perception.md (marker `gpu`, run by the user on the Spark with the real
checkpoint and FLA): (13) the stream step's latency on synthetic states; 220 image tokens
per 640 x 360 frame; the delta rule `transformers` chose is FLA's `chunk_gated_delta_rule`;
FLA's kernel against the torch function of M on the same random bfloat16 inputs. One CPU
test checks the introspection the GPU test relies on where FLA is absent."""

import importlib
import inspect
import statistics
import time

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers.image_utils import PILImageResampling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)

from mcpref import REPO
from mcpref.clips.collect import load_config as load_data_config
from mcpref.perception.config import load_config
from mcpref.perception.phi import build_phi
from mcpref.perception.stream import PhiStream
from mcpref.perception.tokens import image_inputs, segment, specials
from mcpref.pretrain.clip import Clip

from .world import fla_chosen

M = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
BF16 = {"rtol": 1.6e-2, "atol": 1.6e-2}  # bfloat16's precision, also absolute


def chosen(function) -> object:
    """The implementation `use_kernel_func_from_hub_with_fallback` bound into `function`."""
    return inspect.getclosurevars(function).nonlocals["implementation"]


@pytest.mark.skipif(fla_chosen(), reason="FLA present")
def test_torch_path_chosen_without_fla():
    """Domain: without FLA the torch function (the undecorated original) is chosen."""
    f = M.torch_chunk_gated_delta_rule
    assert chosen(f) is f.__wrapped__


@pytest.mark.gpu
def test_fla_chosen():
    """The chunked delta rule Qwen3_5GatedDeltaNet calls is FLA's on the Spark."""
    fla = importlib.import_module("fla.ops.gated_delta_rule")
    assert chosen(M.torch_chunk_gated_delta_rule) is fla.chunk_gated_delta_rule


@pytest.mark.parametrize("initial", [False, True], ids=["zero state", "initial state"])
@pytest.mark.gpu
def test_fla_against_the_torch_function(initial):
    """chunk_gated_delta_rule of FLA vs M's torch function, as M calls them (g in float32
    as M computes it), length 100 (not a multiple of 64): outputs and final states."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    gen = torch.Generator(device="cuda").manual_seed(0)
    B, T, Hh, K = 1, 100, 4, 128

    def rand(*shape) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device="cuda").bfloat16()

    q, k, v = rand(B, T, Hh, K), rand(B, T, Hh, K), rand(B, T, Hh, K)
    beta = torch.sigmoid(rand(B, T, Hh).float()).bfloat16()
    g = (-torch.nn.functional.softplus(rand(B, T, Hh).float())).bfloat16().float()
    h0 = rand(B, Hh, K, K).float() if initial else None
    kwargs = {"g": g, "beta": beta, "initial_state": h0, "output_final_state": True,
              "use_qk_l2norm_in_kernel": True}  # fmt: skip
    out_f, state_f = chunk_gated_delta_rule(q, k, v, **kwargs)
    out_t, state_t = M.torch_chunk_gated_delta_rule.__wrapped__(q, k, v, **kwargs)
    torch.testing.assert_close(out_f.float(), out_t.float(), **BF16)
    torch.testing.assert_close(state_f.float(), state_t.float(), **BF16)


def synthetic_line() -> dict:
    """(13): hotbar 0, yaw 0, pitch 0, position (0, 64, 0), 15 item types of quantity 1."""
    inventory = [{"type": f"item_{k}", "quantity": 1} for k in range(15)]
    return {"inventory": inventory, "hotbar": 0, "yaw": 0.0, "pitch": 0.0, "xpos": 0.0,
            "ypos": 64.0, "zpos": 0.0}  # fmt: skip


@pytest.mark.gpu
@pytest.mark.timeout(3600)  # the first run downloads the weights
def test_stream_latency(capsys):
    """(13): median of 3L stream steps after a warm-up stream of L, each timed with
    perf_counter around the step and cuda.synchronize(); fails above 300 ms (target
    150 ms); prints it and one training forward and backward of a clip of L states."""
    cfg = load_config(REPO / "configs/perception.yaml")
    L = load_data_config(REPO / "configs/perception-data.yaml").states
    phi = build_phi(cfg).to("cuda").eval()
    rng = np.random.default_rng(0)
    stream, line = PhiStream(phi, L), synthetic_line()
    times = []
    for k in range(4 * L):
        frame = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        stream.step(frame, line)
        torch.cuda.synchronize()
        if k >= L:
            times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    phi.train()
    frames = rng.integers(0, 256, (L, 640 * 360 * 3 // 2), dtype=np.uint8)
    info = {"states": list(range(L)), "gui": [False] * L, "n": [1] * (L - 1)}
    clip = Clip(frames=frames, width=640, height=360, info=info, lines=[line] * L)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    phi([clip]).sum().backward()
    torch.cuda.synchronize()
    train = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\nstream step: median {1000 * median:.1f} ms over {3 * L} steps (target "
              f"150 ms); training forward and backward of {L} states: {train:.2f} s")  # fmt: skip
    assert median <= 0.300


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_image_tokens_per_frame():
    """A 640 x 360 frame: grid [1, 22, 40], 220 image tokens in its segment and 220 image
    features of the real backbone."""
    cfg = load_config(REPO / "configs/perception.yaml")
    phi = build_phi(cfg).to("cuda").eval()
    vision = phi.backbone.config.vision_config
    proc = Qwen2VLImageProcessor(
        do_resize=True, size={"shortest_edge": cfg.min_pixels, "longest_edge": cfg.max_pixels},
        resample=PILImageResampling.BICUBIC, do_rescale=True,
        rescale_factor=cfg.rescale_factor, do_normalize=True,
        image_mean=list(cfg.image_mean), image_std=list(cfg.image_std),
        do_convert_rgb=True, patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size,
    )  # fmt: skip
    frame = np.random.default_rng(0).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    pv, grid = image_inputs(proc, [frame])
    assert grid.tolist() == [[1, 22, 40]]
    sp = specials(phi.backbone.config)
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    assert segment(tok, sp, grid[0], "x", cfg.marker).count(sp.image) == 220
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = phi.backbone.get_image_features(pv.cuda(), grid.cuda()).pooler_output
    assert [f.shape[0] for f in features] == [220]
