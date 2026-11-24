"""Reproducible CUDA backward for retained recurrent VJPs and checkpoint resume."""

import os
from contextlib import contextmanager

import torch


@contextmanager
def reproducible(device):
    """Enter before CUDA model construction; preserve the caller's Torch flags."""
    if torch.device(device).type != "cuda":
        yield
        return
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise ValueError(
            "Continuous CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        # All consumed tensors are initialized; filling every FLA workspace is costly.
        torch.utils.deterministic.fill_uninitialized_memory = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
