"""Select and read the initial runtime weights; specs/env-sampling.md (1)."""

from pathlib import Path

import torch

from mcpref.config import REPO
from mcpref.infrastructure.runs.files import finished_run
from mcpref.runtime.sampling.config import SamplingConfig


def checkpoint_path(cfg: SamplingConfig) -> Path:
    directory = REPO / cfg.checkpoint_dir
    if any(part.endswith(".part") for part in directory.resolve().parts):
        raise ValueError(f"checkpoint_dir must name a published directory: {directory}")
    paths = [directory / name for name in ("checkpoint.pt", "model.pt")]
    present = [path for path in paths if path.is_file()]
    if len(present) > 1:
        raise ValueError(
            f"ambiguous checkpoint_dir: both checkpoint.pt and model.pt in {directory}"
        )
    if not present:
        raise FileNotFoundError(f"no checkpoint.pt or model.pt in {directory}")
    path = present[0]
    if path.name == "model.pt":
        finished_run(directory, cfg.checkpoint_dir, "complete")
    return path


def load_weights(path: Path) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return saved["modules"] if path.name == "checkpoint.pt" else saved
