"""(4), (6)-(8), (12) of specs/perception.md: phi, the backbone B with the MLP head over
the interleaved states of a clip."""

from itertools import accumulate, pairwise

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Cache,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    Qwen2VLImageProcessor,
    Qwen3_5Model,
)
from transformers.image_utils import PILImageResampling
from transformers.trainer_pt_utils import get_parameter_names

from mcpref.perception.config import PerceptionConfig
from mcpref.perception.cursor import load_sprite
from mcpref.perception.meta import clip_states
from mcpref.perception.tokens import image_inputs, positions, segment, specials
from mcpref.pretrain.clip import Clip

NO_DECAY = [
    r"bias",
    r"layernorm",
    r"rmsnorm",
    r"(?:^|\.)norm(?:$|\.)",
    r"_norm(?:$|\.)",
]
"""(12): P_T, the parameter-name patterns without weight decay of [T]."""
PARTS = {
    "phi_vision": "backbone.visual.",
    "phi_llm": "backbone.language_model.",
    "phi_head": "head.",
}
"""(12): each pair of groups and the name prefix of its parameters."""


class Head(torch.nn.Sequential):
    """(7): z = head(h_r), h_r cast to the dtype of the head's parameters, outside
    autocast."""

    def forward(self, h: Tensor) -> Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            return super().forward(h.to(self[0].weight.dtype))


def make_head(cfg: PerceptionConfig, d_h: int) -> Head:
    """(7): `head_layers` Linear -> SiLU blocks of width `head_width`, then Linear to D;
    drawn on the CPU from `seed`, no global generator's state used or changed."""
    widths = [d_h] + [cfg.head_width] * cfg.head_layers
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg.seed)
        layers = []
        for n_in, n_out in pairwise(widths):
            layers += [torch.nn.Linear(n_in, n_out), torch.nn.SiLU()]
        return Head(*layers, torch.nn.Linear(widths[-1], len(cfg.alpha)))


def image_processor(
    cfg: PerceptionConfig, vision: PretrainedConfig
) -> Qwen2VLImageProcessor:
    """(4): the torchvision image processor, every argument explicit; P_v, t_p, m_s
    from B's vision config."""
    return Qwen2VLImageProcessor(
        do_resize=True,
        size={"shortest_edge": cfg.min_pixels, "longest_edge": cfg.max_pixels},
        resample=PILImageResampling.BICUBIC,
        do_rescale=True,
        rescale_factor=cfg.rescale_factor,
        do_normalize=True,
        image_mean=cfg.image_mean,
        image_std=cfg.image_std,
        do_convert_rgb=True,
        patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size,
        merge_size=vision.spatial_merge_size,
    )


class Phi(torch.nn.Module):
    """(8): clips -> Z of shape (n, L, D), each clip in a forward of its own."""

    def __init__(
        self,
        backbone: Qwen3_5Model,
        tokenizer: PreTrainedTokenizerBase,
        cfg: PerceptionConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = make_head(cfg, backbone.config.text_config.hidden_size)
        self.tokenizer, self.cfg = tokenizer, cfg
        self.sp = specials(backbone.config)
        self.processor = image_processor(cfg, backbone.config.vision_config)
        self.sprite = load_sprite(cfg)
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def hidden(
        self,
        ids: list[int],
        pixels: Tensor,
        grid: Tensor,
        pos: Tensor,
        cache: Cache | None,
    ) -> Tensor:
        """(6), (9): B's last hidden states of `ids` under autocast, the inputs moved to
        phi's device; the cache used when given."""
        device = self.head[0].weight.device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = self.backbone(
                input_ids=torch.tensor([ids], dtype=torch.int64, device=device),
                pixel_values=pixels.to(device),
                image_grid_thw=grid.to(device),
                position_ids=pos.to(device),
                use_cache=cache is not None,
                past_key_values=cache,
            )
        return out.last_hidden_state[0]

    def clip_z(self, clip: Clip) -> Tensor:
        """(3)-(7) of one clip: z_s = head(h_{r_s}), shape (L, D)."""
        frames, metas = clip_states(clip, self.cfg, self.sprite)
        pixels, grid = image_inputs(self.processor, frames)
        segments = [
            segment(self.tokenizer, self.sp, g, mu, self.cfg.marker)
            for g, mu in zip(grid, metas)
        ]
        ids = [t for sigma in segments for t in sigma]
        r = [n - 1 for n in accumulate(len(sigma) for sigma in segments)]
        h = self.hidden(
            ids, pixels, grid, positions(self.backbone, ids, grid, self.sp), None
        )
        return self.head(h[r])

    def forward(self, clips: list[Clip]) -> Tensor:
        """(8): Z, in the dtype of the head."""
        return torch.stack([self.clip_z(clip) for clip in clips])

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): the decayed and other parameters of B.visual, B.language_model and the
        head, each list in named_parameters() order."""
        decayed = set(get_parameter_names(self, [torch.nn.LayerNorm], NO_DECAY))
        named = list(self.named_parameters())
        out = {}
        for group, prefix in PARTS.items():
            part = [(n, p) for n, p in named if n.startswith(prefix)]
            out[group] = [p for n, p in part if n in decayed]
            out[f"{group}_no_decay"] = [p for n, p in part if n not in decayed]
        return out


def build_phi(cfg: PerceptionConfig) -> Phi:
    """Interface: B and tau_tok of `model` at `revision`; phi on the CPU."""
    backbone = Qwen3_5Model.from_pretrained(
        cfg.model,
        revision=cfg.revision,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    return Phi(backbone, tokenizer, cfg)
