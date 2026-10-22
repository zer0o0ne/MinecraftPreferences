"""(3)-(5) of specs/perception.md: `specials` from the backbone's config; `image_inputs`
(4) on a 640 x 360 frame (grid [1, 22, 40], 880 rows) and on a frame whose channel axis
would be ambiguous; `segment` (3) on hand metas; `positions` (5) by hand and against an
in-test transcription of the rule of [V1] (oracle.rope)."""

import dataclasses

import numpy as np
import pytest
import torch
from transformers import Qwen3_5Config

from mcpref.perception.tokens import (
    Specials,
    image_inputs,
    positions,
    segment,
    specials,
)

from . import oracle
from .world import B_ID, BASE, E_ID, MARKER_ID, P_ID, backbone, tokenizer

SP = Specials(start=B_ID, image=P_ID, end=E_ID, merge=2)
STARTING = BASE | {"min_pixels": 65536, "max_pixels": 16777216,
                   "image_mean": [0.5] * 3, "image_std": [0.5] * 3}  # fmt: skip


def test_specials_from_the_config():
    """b, p, e and m_s of B's config, each from its own field."""
    config = Qwen3_5Config(vision_start_token_id=11, image_token_id=12,
                           vision_end_token_id=13, video_token_id=14,
                           vision_config={"spatial_merge_size": 3})  # fmt: skip
    assert specials(config) == Specials(start=11, image=12, end=13, merge=3)
    assert [f.name for f in dataclasses.fields(Specials)] == ["start", "image", "end", "merge"]  # fmt: skip
    with pytest.raises(dataclasses.FrozenInstanceError):
        SP.start = 0


def test_specials_of_the_tiny_backbone():
    assert specials(backbone().config) == SP


def random_frames(*shapes) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, s, dtype=np.uint8) for s in shapes]


def test_image_inputs_640x360():
    """[V1]: 640 x 360 -> 640 x 352, grid [1, 22, 40], pixel_values (880, 1536)."""
    proc = oracle.processor(STARTING)
    (f,) = random_frames((360, 640, 3))
    pv, grid = image_inputs(proc, [f])
    assert grid.tolist() == [[1, 22, 40]] and pv.shape == (880, 1536)
    torch.testing.assert_close(pv, oracle.pixels([f], STARTING)[0], rtol=0, atol=0)
    assert pv.device.type == "cpu" and grid.device.type == "cpu"


def test_image_inputs_channels_last():
    """A (3, 64, 3) frame is 3 rows of 64 RGB pixels: min_pixels 65536 resizes it to
    64 x 1184 (grid [1, 4, 74]), not to a 64-row, 3-column image ([1, 74, 4])."""
    proc = oracle.processor(STARTING)
    (f,) = random_frames((3, 64, 3))
    pv, grid = image_inputs(proc, [f])
    assert grid.tolist() == [[1, 4, 74]] and pv.shape == (296, 1536)
    torch.testing.assert_close(pv, oracle.pixels([f], STARTING)[0], rtol=0, atol=0)


def test_image_inputs_frames_in_order():
    """The rows of all frames in order, one grid row per frame; the config's processor
    arguments (resize, mean, std) visible in the values."""
    proc = oracle.processor(BASE)
    frames = random_frames((30, 70, 3), (64, 32, 3), (30, 70, 3))
    pv, grid = image_inputs(proc, frames)
    assert grid.tolist() == [[1, 2, 4], [1, 4, 2], [1, 2, 4]]
    parts = [image_inputs(proc, [f])[0] for f in frames]
    torch.testing.assert_close(pv, torch.cat(parts), rtol=0, atol=0)
    torch.testing.assert_close(pv, oracle.pixels(frames, BASE)[0], rtol=0, atol=0)


def test_segment_by_hand():
    """sigma_s = [b] + [p]^{t h w / 4} + [e] + tau(mu_s) + tau(omega): the special marker
    one id (split_special_tokens=False), no BOS."""
    tok = tokenizer()
    got = segment(tok, SP, torch.tensor([1, 2, 4]), "Hi\n", "<|im_end|>")
    assert got == [B_ID, P_ID, P_ID, E_ID, ord("H"), ord("i"), ord("\n"), MARKER_ID]


@pytest.mark.parametrize(
    "grid, n", [([1, 22, 40], 220), ([2, 4, 6], 12), ([1, 2, 2], 1)]
)
def test_segment_image_tokens(grid, n):
    """N_s = t h w / m_s^2 of its grid row."""
    got = segment(tokenizer(), SP, torch.tensor(grid), "", "<|im_end|>")
    assert got == [B_ID, *[P_ID] * n, E_ID, MARKER_ID]


def test_segment_merge_of_the_specials():
    sp = Specials(start=B_ID, image=P_ID, end=E_ID, merge=3)
    got = segment(tokenizer(), sp, torch.tensor([1, 6, 9]), "", "<|im_end|>")
    assert got.count(P_ID) == 6


def test_segment_multi_token_marker_and_utf8():
    """A marker of several tokens ends the segment (its last token the readout); the meta
    as UTF-8 bytes of the byte-level tokenizer."""
    got = segment(tokenizer(), SP, torch.tensor([1, 2, 2]), "é\t1", "##")
    assert got == [B_ID, P_ID, E_ID, *"é\t1".encode(), ord("#"), ord("#")]


def hand_ids() -> list[int]:
    """Two segments: grid [1, 2, 4] (H' x W' = 1 x 2) then [1, 4, 2] (2 x 1)."""
    t = ord("t")
    return [B_ID, P_ID, P_ID, E_ID, t, MARKER_ID, B_ID, P_ID, P_ID, E_ID, MARKER_ID]


def test_positions_by_hand():
    """b (0); image 1 x 2 at q = 1: (1,1,1), (1,1,2), q = 3; e 3, t 4, marker 5, b 6;
    image 2 x 1 at q = 7: (7,7,7), (7,8,7), q = 9; e 9, marker 10."""
    grid = torch.tensor([[1, 2, 4], [1, 4, 2]])
    got = positions(backbone(), hand_ids(), grid, SP)
    t = [0, 1, 1, 3, 4, 5, 6, 7, 7, 9, 10]
    h = [0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    w = [0, 1, 2, 3, 4, 5, 6, 7, 7, 9, 10]
    assert got.shape == (3, 1, 11)
    assert got[:, 0].tolist() == [t, h, w]


def test_positions_against_the_rule():
    """Three segments (grids [1, 22, 40], [1, 4, 74], [1, 2, 4]) with hand metas."""
    grid = torch.tensor([[1, 22, 40], [1, 4, 74], [1, 2, 4]])
    tok = tokenizer()
    ids = []
    for row, text in zip(grid, ["Inv: none\nH1", "", "x\ny"]):
        ids += segment(tok, SP, row, text, "<|im_end|>")
    got = positions(backbone(), ids, grid, SP)
    assert got.dtype == torch.int64
    assert torch.equal(got, oracle.rope(ids, grid.tolist()))
