"""Calls of `fit` of specs/gui-stack.md on the tiny set of fit_oracle.py, with stderr
captured; the config and the oracle's view of it."""

import contextlib
import dataclasses
import io
from pathlib import Path

import pytest
import torch
from progress_bars import tqdm_defaults

from gui.stacking import fit_oracle as fo
from gui.stacking import stack_oracle as so
from mcpref.gui.stacking.config import GuiStackConfig
from mcpref.gui.stacking.fit import fit

SIZES = [5, 2]  # B = 2: M = 3 + 1 = 4
# T = 8; a checkpoint after every step
FIT = {"hidden": 4, "l2": 0.02, "lr": 0.05, "betas": (0.8, 0.95), "adam_eps": 1e-3,
       "epochs": 2, "batch": 2, "checkpoint_steps": 1, "seed": 9}  # fmt: skip


BASE = {
    "masks_dir": "data/gui_masks", "hidden": 64, "l2": 1e-4, "lr": 0.01,
    "betas": (0.9, 0.999), "adam_eps": 1e-8, "epochs": 50, "batch": 32,
    "checkpoint_steps": 500, "seed": 0, "device": "cpu", "out_dir": "data/gui_stack",
}  # fmt: skip


def config(_repo: Path, **fields):
    """The explicit starting config with `fields` replaced; no repository read."""
    return dataclasses.replace(GuiStackConfig(**BASE), **fields)


def fit_in(repo: Path, out: Path, read, cfg: dict) -> tuple:
    """fit on SIZES (C = 3, |Y| = 4, sum_S w = 3) with checkpoints in out/checkpoints:
    ((theta_T, L(theta_0), L(theta_T)), stderr), tqdm's defaults."""
    err = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        tqdm_defaults(mp)
        with contextlib.redirect_stderr(err):
            got = fit(config(repo, **cfg), read, SIZES, 3, 4, 3.0, out / "checkpoints")
    return got, err.getvalue()


def checkpoint(out: Path, j: int) -> dict:
    """Checkpoint j loaded by `torch.load(path)` with its defaults (weights_only=True)."""
    return torch.load(out / "checkpoints" / f"{j:06d}" / "checkpoint.pt")


def checkpoints(out: Path) -> list[str]:
    return sorted(p.name for p in (out / "checkpoints").iterdir())


def oracle(z, y, w, cfg: dict) -> tuple:
    """theta_0 and (theta_t, m_t, v_t), t = 1..T, of (2), (4) in float64."""
    theta0 = so.init(3, cfg["hidden"], 4, cfg["seed"])
    return theta0, fo.adam_path(theta0, z, y, w, SIZES, cfg)
