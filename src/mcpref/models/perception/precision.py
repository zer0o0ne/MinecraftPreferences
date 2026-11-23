"""Conservative FP8 text-MLP computation; see specs/perception.md, Precision."""

from dataclasses import replace

import torch
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from torchao.float8.config import Float8GemmConfig
from transformers import Qwen3_5Model


def _cpu_autocast_input(module, args):
    """torchao 0.18 Float8Linear only applies CUDA autocast itself."""
    if torch.is_autocast_enabled("cpu"):
        return (args[0].to(torch.get_autocast_dtype("cpu")),)
    return args


def enable_fp8_mlp(backbone: Qwen3_5Model) -> None:
    """Preserve parameter storage; CPU uses FP8 emulation."""
    device = next(backbone.parameters()).device
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"FP8 MLP requires CPU emulation or CUDA, got {device}")
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 9):
        raise ValueError("FP8 MLP requires CUDA compute capability >= 8.9")
    config = replace(
        Float8LinearConfig.from_recipe_name("rowwise_with_gw_hp"),
        gemm_config_output=Float8GemmConfig(use_fast_accum=False),
        gemm_config_grad_input=Float8GemmConfig(use_fast_accum=False),
        gemm_config_grad_weight=Float8GemmConfig(use_fast_accum=False),
        enable_fsdp_float8_all_gather=False,
        pad_inner_dim=False,
        emulate=device.type == "cpu",
        round_scales_to_power_of_2=True,
    )
    for layer in backbone.language_model.layers:
        mlp = layer.mlp
        for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if projection.in_features % 16 or projection.out_features % 16:
                raise ValueError(
                    "FP8 MLP projection dimensions must be divisible by 16"
                )
        convert_to_float8_training(mlp, config=config)
        if device.type == "cpu":
            for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                projection.register_forward_pre_hook(_cpu_autocast_input)
        mlp.train(mlp.training)
        if device.type == "cuda":
            mlp.compile(
                backend="inductor",
                fullgraph=True,
                dynamic=True,
                options={"compile_threads": 1},
            )
