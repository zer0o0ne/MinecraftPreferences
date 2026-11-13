"""Capture a bounded window of complete optimizer iterations, including their I/O."""

import os
import sys
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from mcpref.config import (
    REPO,
    boolean,
    config_mapping,
    integer,
    positive_int,
    read_yaml,
)
from mcpref.infrastructure.profiling.report import publish
from mcpref.infrastructure.profiling.scopes import ACTIVE, module_scopes, phase


@dataclass(frozen=True)
class ProfileConfig:
    skip_steps: int
    steps: int
    top_ops: int


def settings() -> ProfileConfig | None:
    path = os.environ.get("MCPREF_PROFILE", "configs/training/profile.yaml")
    if not path:
        return None
    path = REPO / path
    if "MCPREF_PROFILE" not in os.environ and not path.exists():
        return None
    raw = read_yaml(path)
    enabled = (
        boolean("enabled", raw.pop("enabled", True)) if isinstance(raw, dict) else True
    )
    raw = config_mapping(raw, path, ProfileConfig)
    cfg = ProfileConfig(
        integer("skip_steps", raw["skip_steps"], 0, float("inf")),
        positive_int("steps", raw["steps"]),
        positive_int("top_ops", raw["top_ops"]),
    )
    return cfg if enabled else None


class TrainingProfile(AbstractContextManager):
    def __init__(self, cfg, root: Path, modules: dict, stage: str):
        self.settings = settings()
        self.device = torch.device(cfg.device)
        self.modules, self.stage = modules, stage
        self.directory = root / "profiles" / stage / uuid4().hex
        self.seen = self.captured = 0
        self.metadata = {
            "stage": stage,
            "device": str(self.device),
            "torch": torch.__version__,
            "batch": cfg.batch,
            "micro_batch": cfg.micro_batch,
            "gradient_diagnostics": cfg.gradient_diagnostics,
        }

    def __enter__(self):
        if self.settings is not None:
            print(
                f"profile: {self.stage}: skip {self.settings.skip_steps} executed steps, "
                f"capture {self.settings.steps}; output {self.directory}",
                file=sys.stderr,
            )
        return self

    def __exit__(self, *exc):
        if self.settings is not None and self.captured == 0:
            print(f"profile: {self.stage}: no complete steps captured", file=sys.stderr)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def step(self, step: int, items: int):
        self.seen += 1
        cfg = self.settings
        if cfg is None or not cfg.skip_steps < self.seen <= cfg.skip_steps + cfg.steps:
            yield
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        cuda = self.device.type == "cuda"
        if cuda:
            if (
                torch.profiler.ProfilerActivity.CUDA
                not in torch.profiler.supported_activities()
            ):
                raise RuntimeError("CUDA profiling is unavailable; check PyTorch/CUPTI")
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.synchronize()
        memory = None
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        batches = []
        token = ACTIVE.set(batches)
        try:
            with (
                torch.profiler.profile(
                    activities=activities,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as trace,
                module_scopes(self.modules),
            ):
                start = perf_counter()
                with phase("step"):
                    yield
                    with phase("device_wait"):
                        self.synchronize()
                wall = perf_counter() - start
                if cuda:
                    memory = {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            self.device
                        ),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                            self.device
                        ),
                    }
        finally:
            ACTIVE.reset(token)
        publish(
            trace,
            self.directory / f"step-{step:06d}",
            self.metadata
            | {
                "step": step,
                "executed_step": self.seen,
                "items": items,
                "wall_s": wall,
                "items_per_s": items / wall,
                "cuda_activity": cuda,
                "memory": memory,
                "qwen_batches": batches,
            },
            cuda,
            cfg.top_ops,
        )
        self.captured += 1
