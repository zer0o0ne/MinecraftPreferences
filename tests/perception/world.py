"""Tiny worlds of the tests of specs/perception.md (Tests): a valid config of the tiny world
written as YAML, a cursor sprite, a tiny byte-level tokenizer with b, p, e and the special
marker, a tiny random `Qwen3_5Model` (two text layers, linear then full attention; a tiny
vision tower with patch 16, merge 2, temporal patch 2), clips of yuv420p frames with neutral
chroma; the skip condition of Domain (FLA chosen by `transformers`)."""

import hashlib
import importlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5Model

from mcpref.perception.config import PerceptionConfig, load_config
from mcpref.pretrain.clip import Clip

W, H = 70, 30  # clip frames; resized by (4) to 64 x 32: grid [1, 2, 4], N_s = 2
F64 = torch.float64
B_ID, P_ID, E_ID, VIDEO_ID, MARKER_ID = 256, 257, 258, 259, 260
SPECIALS = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>",
            "<|im_end|>"]  # fmt: skip
D_H = 32  # text_config.hidden_size of the tiny backbone


def sprite_rgba() -> np.ndarray:
    """A 20 x 24 RGBA sprite (larger than k_c = 16), alpha 0 and 255 included."""
    rgba = np.random.default_rng(11).integers(0, 256, (20, 24, 4), dtype=np.uint8)
    rgba[0, 0, 3], rgba[1, 1, 3], rgba[15, 15, 3] = 0, 255, 255
    return rgba


def png(array: np.ndarray, mode: str) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array, mode).save(out, "PNG")
    return out.getvalue()


SPRITE = png(sprite_rgba(), "RGBA")
SPRITE_SHA = hashlib.sha256(SPRITE).hexdigest()
S_RGB = sprite_rgba()[:16, :16, :3]  # (1): the top-left k_c x k_c, colour
S_ALPHA = sprite_rgba()[:16, :16, 3:] / 255.0  # alpha / 255, float64, trailing axis 1

BASE = {  # the values of a valid config of the tiny world; `cursor` set by `write_config`
    "model": "tiny/model",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "seed": 7,
    "alpha": [1.0, 2, 0.5, 3],
    "head_layers": 1,
    "head_width": 8,
    "c": 1.5,
    "beta": 0.7,
    "dual_init": 30.0,
    "dual_slack": 1.0e-3,
    "tau": 1.0,
    "min_pixels": 1024,
    "max_pixels": 1048576,
    "rescale_factor": 1 / 255,
    "image_mean": [0.4, 0.5, 0.6],
    "image_std": [0.2, 0.3, 0.25],
    "decimals": 1,
    "meta": "{inventory}H{hotbar} Y{yaw} P{pitch} X{x} {y} {z}\n",
    "inventory_line": "Inv: {items}\n",
    "item_line": "{quantity} {type}",
    "no_items": "none",
    "marker": "<|im_end|>",
    "cursor": None,
    "cursor_url": "http://127.0.0.1:9/cursor.png",
    "cursor_sha256": SPRITE_SHA,
    "cursor_crop": 16,
    "mouse_height": 720,
    "http_timeout": 5,
    "http_retries": 2,
    "retry_delay": 0.01,
}


def values(tmp_path: Path, **changes) -> dict:
    """BASE with `changes`; `cursor` the absolute path tmp/sprite/cursor.png unless given."""
    return BASE | {"cursor": str(tmp_path / "sprite" / "cursor.png")} | changes


def write_config(tmp_path: Path, sprite: bytes | None = SPRITE, **changes) -> Path:
    """The YAML of `values(tmp_path, **changes)`; `sprite` written at a string `cursor`
    unless None."""
    doc = values(tmp_path, **changes)
    if sprite is not None and isinstance(doc["cursor"], str):
        Path(doc["cursor"]).parent.mkdir(parents=True, exist_ok=True)
        Path(doc["cursor"]).write_bytes(sprite)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "perception.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def config(tmp_path: Path, **changes) -> PerceptionConfig:
    return load_config(write_config(tmp_path, **changes))


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> character map of byte-level BPE."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    bs += range(ord("®"), ord("ÿ") + 1)
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tokenizer() -> PreTrainedTokenizerFast:
    """Byte-level, no merges: byte k of a text's UTF-8 is id k; the specials 256..261.
    Its defaults differ from (3)'s arguments: a BOS (261) added and special tokens split,
    unless the call passes add_special_tokens=False and split_special_tokens=False."""
    vocab = {c: b for b, c in bytes_to_unicode().items()}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([*SPECIALS, "<|bos|>"])
    tok.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A", special_tokens=[("<|bos|>", 261)]
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, split_special_tokens=True)
    assert fast.convert_tokens_to_ids([*SPECIALS, "<|bos|>"]) == list(range(256, 262))
    assert fast("a<|im_end|>")["input_ids"] == [261, 97, *b"<|im_end|>"]
    return fast


def backbone_config() -> Qwen3_5Config:
    rope = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
            "mrope_section": [2, 1, 1], "mrope_interleaved": True}  # fmt: skip
    text = {"vocab_size": 272, "hidden_size": D_H, "intermediate_size": 64,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
            "initializer_range": 0.1, "rope_parameters": rope}  # fmt: skip
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
              "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
              "out_hidden_size": D_H, "num_position_embeddings": 16,
              "initializer_range": 0.1}  # fmt: skip
    return Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=P_ID,
                         video_token_id=VIDEO_ID, vision_start_token_id=B_ID,
                         vision_end_token_id=E_ID, attn_implementation="sdpa")  # fmt: skip


def backbone(dtype: torch.dtype = F64, seed: int = 0) -> Qwen3_5Model:
    """The tiny backbone, initialised under its own seed (global generators untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Qwen3_5Model(backbone_config()).to(dtype)


def fla_chosen() -> bool:
    """Domain: `import fla.ops.gated_delta_rule` succeeds and gives `chunk_gated_delta_rule`."""
    try:
        module = importlib.import_module("fla.ops.gated_delta_rule")
    except Exception:  # noqa: BLE001  (any failure: the torch path, as transformers does)
        return False
    return getattr(module, "chunk_gated_delta_rule", None) is not None


runs_backbone = pytest.mark.skipif(
    fla_chosen(), reason="FLA computes the delta rule: the backbone fails on the CPU"
)


def yuv(y: np.ndarray) -> np.ndarray:
    """A yuv420p frame row: luma `y` (H x W), chroma 128 (neutral)."""
    chroma = np.full(y.size // 2, 128, dtype=np.uint8)
    return np.concatenate([np.asarray(y, np.uint8).ravel(), chroma])


def rgb(y: np.ndarray) -> np.ndarray:
    """(5) of specs/vpt-gui-frames.md at neutral chroma: R = G = B = clip(floor(255 (Y - 16)
    / 219 + 1/2), 0, 255), in exact integer arithmetic."""
    n = (np.asarray(y, np.int64) - 16) * 255
    v = np.clip((2 * n + 219) // 438, 0, 255).astype(np.uint8)
    return np.repeat(v[..., None], 3, axis=-1)


def item(type_: str, quantity: int) -> dict:
    return {"type": type_, "quantity": quantity}


def line(**fields) -> dict:
    """A line with the fields of (2), the mouse and the GUI flag of the data."""
    base = {"inventory": [], "hotbar": 0, "yaw": 0.0, "pitch": 0.0, "xpos": 0.0,
            "ypos": 64.0, "zpos": 0.0, "mouse": {"x": 0.0, "y": 0.0},
            "isGuiOpen": False}  # fmt: skip
    return base | fields


def clip(lumas: list, lines: list, states: list, gui: list, n: list) -> Clip:
    """A Clip of Notation: frames.yuv rows from `lumas`, `info` with states, gui, n."""
    frames = np.stack([yuv(y) for y in lumas])
    info = {"id": "c", "states": states, "gui": gui, "n": n}
    return Clip(frames=frames, width=W, height=H, info=info, lines=lines)


def random_line(rng: np.random.Generator) -> dict:
    types = ["log", "dirt", "stone", "a_b"]
    inventory = [item(str(rng.choice(types)), int(rng.integers(1, 64)))
                 for _ in range(int(rng.integers(0, 4)))]  # fmt: skip
    mouse = {"x": float(rng.uniform(0, 1700)), "y": float(rng.uniform(0, 740))}
    return line(inventory=inventory, hotbar=int(rng.integers(0, 9)),
                yaw=float(rng.uniform(-900, 900)), pitch=float(rng.uniform(-90, 90)),
                xpos=float(rng.uniform(-500, 500)), ypos=float(rng.uniform(0, 128)),
                zpos=float(rng.uniform(-500, 500)), mouse=mouse,
                isGuiOpen=bool(rng.integers(0, 2)))  # fmt: skip


def random_clip(seed: int, states: list[int], gui: list[bool]) -> Clip:
    """A clip with random lumas and lines; a line between states unused unless a state."""
    rng = np.random.default_rng(seed)
    L = len(states)
    lumas = [rng.integers(16, 236, (H, W)) for _ in range(L)]
    lines = [random_line(rng) for _ in range(states[-1] - states[0] + 1)]
    n = [int(k) for k in rng.integers(1, 6, L - 1)]
    return clip(lumas, lines, states, gui, n)
