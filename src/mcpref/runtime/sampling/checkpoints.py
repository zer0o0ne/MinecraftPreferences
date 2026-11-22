"""Select and read the initial runtime weights; specs/env-sampling.md (1)."""

import json
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


def _check_mode(actual: str, expected: str) -> None:
    if actual != expected or actual not in ("classical", "continuous"):
        raise RuntimeError(
            f"Pretraining mode mismatch: artifact={actual}, sampling={expected}; "
            "continuous artifacts require the continuous perception stream adapter"
        )


def load_weights(path: Path, mode: str = "classical") -> dict:
    metadata = path.parent / "metadata.json"
    info = json.loads(metadata.read_text()) if metadata.exists() else None
    if info is not None:
        _check_mode(info.get("pretrain_mode", "classical"), mode)
    allowed = []
    if mode == "continuous":
        import numpy as np
        from numpy.core.multiarray import _reconstruct

        from mcpref.models.perception.continuous.state import ContinuousState, GDNState
        from mcpref.training.pretrain.continuous.state import Stream

        allowed = [
            ContinuousState,
            GDNState,
            Stream,
            _reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint8)),
        ]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    _check_mode(saved.get("pretrain_mode", "classical"), mode)
    if mode == "continuous":
        for values in (info or {}, saved):
            for key, expected in (("attention", "observation"), ("memory", "fragment")):
                if key in values and values[key] != expected:
                    raise RuntimeError(f"Incompatible continuous perception {key}")
    return saved["modules"] if path.name == "checkpoint.pt" else saved
