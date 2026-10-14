"""Synthetic `{masks_dir}/features.npz` in the layout of specs/gui-masks.md (Produces), configs
of specs/gui-stack.md, and runs of its `main` with stderr captured."""

import contextlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from progress_bars import tqdm_defaults

from mcpref.stacking.run import main

from .stack_oracle import weights

VARIANTS, CHOICES = "abcdef", ("agreement", "macro_recall")
PRED_KEYS = [f"test_pred_{v}_{c}" for v in VARIANTS for c in CHOICES]
PARTS = ("train", "validation", "test")
NAN = float("nan")
CONFIG = {  # every field but the paths
    "hidden": 5, "l2": 0.01, "lr": 0.05, "betas": [0.8, 0.95], "adam_eps": 1.0e-3,
    "steps": 4, "seed": 7, "device": "cpu",
}  # fmt: skip


@dataclass(frozen=True)
class Part:
    ids: list
    y: list
    g: list
    x: list  # frames x C


@dataclass(frozen=True)
class World:
    columns: list
    labels: list
    train: Part
    validation: Part
    test: Part
    # PRED_KEYS -> labels over the test frames
    preds: dict = field(default_factory=dict)


def preds_for(test: Part, labels: list, seed: int) -> dict:
    """A distinct random labelling of the test frames per key of PRED_KEYS."""
    rng = np.random.default_rng(seed)
    return {k: [str(v) for v in rng.choice(labels, len(test.y))] for k in PRED_KEYS}


def tiny_case(seed: int) -> tuple:
    """C = 3, H = 4, |Y| = 4: theta (float64), z of 7 frames, label indices y with label 1
    absent, w_f = 1 / n_{y_f} (sum 3)."""
    rng = np.random.default_rng(seed)
    theta = [rng.normal(size=s) for s in ((4, 3), (4,), (4, 4), (4,))]
    y = [0, 2, 0, 3, 0, 2, 3]
    return theta, rng.normal(size=(7, 3)), y, weights(y)


def write_features(path: Path, world: World, drop=()) -> None:
    """numpy.savez, no object arrays: str arrays, `<part>_g` bool, `<part>_x` float64."""
    c = len(world.columns)
    arrays = {"columns": np.array(world.columns, dtype=str)}
    arrays["labels"] = np.array(world.labels, dtype=str)
    for name in PARTS:
        part = getattr(world, name)
        arrays[f"{name}_ids"] = np.array(part.ids, dtype=str)
        arrays[f"{name}_y"] = np.array(part.y, dtype=str)
        arrays[f"{name}_g"] = np.array(part.g, dtype=bool)
        x = np.array(part.x, dtype=np.float64)
        arrays[f"{name}_x"] = x.reshape(len(part.ids), c)
    arrays.update({k: np.array(v, dtype=str) for k, v in world.preds.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v for k, v in arrays.items() if k not in drop})


@dataclass
class Run:
    code: int
    err: str
    out: Path

    def report(self) -> dict:
        return json.loads((self.out / "report.json").read_text())

    def predictions(self) -> list:
        lines = (self.out / "predictions.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def model(self) -> dict:
        """model.pt loaded by `torch.load(path)` with its defaults (weights_only=True)."""
        return torch.load(self.out / "model.pt")


def write_config(root: Path, repo: Path, config: dict) -> Path:
    """Config with masks_dir = root/masks, out_dir = root/out, both relative to the
    repository root."""
    paths = {k: os.path.relpath(root / v, repo) for k, v in (("masks_dir", "masks"),
                                                            ("out_dir", "out"))}  # fmt: skip
    path = root / "gui-stack.yaml"
    path.write_text(yaml.safe_dump({**paths, **config}, sort_keys=False))
    return path


def run_main(
    root: Path, repo: Path, world: World | None, config=CONFIG, drop=()
) -> Run:
    """Writes the features of `world` (none if None) and a config under root, runs main from
    another directory with tqdm's defaults (no TQDM_* variable)."""
    if world is not None:
        write_features(root / "masks/features.npz", world, drop)
    path = write_config(root, repo, config)
    (root / "cwd").mkdir()
    err = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        tqdm_defaults(mp)
        mp.chdir(root / "cwd")
        with contextlib.redirect_stderr(err):
            code = main([str(path)])
    return Run(code, err.getvalue(), root / "out")


LABELS = ["alpha", "beta", "gamma", "other"]
K = LABELS[:-1]
TE = Part(["te0", "te1", "te2", "te3"], ["gamma", "alpha", "beta", "other"],
          [True, False, True, False],
          [[0.5, 0.4, 0.4, 0.9], [0.15, 0.1, 0.55, 0.1], [0.85, 0.65, 0.35, -3.0],
           [0.6, 0.5, 0.2, 0.1]])  # fmt: skip
# NaN in (b), (c) columns of train, c/beta NaN in all of it; f/gamma__m 0.1 over S (s = 0);
# gamma not in S
NAN_WORLD = World(
    columns=["a/alpha", "b/beta", "c/beta", "f/gamma__m"],
    labels=LABELS,
    train=Part(["tr0", "tr1", "tr2", "tr3", "tr4"],
               ["alpha", "beta", "alpha", "other", "alpha"],
               [True, False, True, True, False],
               [[0.2, NAN, NAN, 0.1], [0.9, 0.5, NAN, 0.1], [0.3, 0.7, NAN, 0.1],
                [0.6, NAN, NAN, 0.1], [0.1, 0.3, NAN, 0.1]]),
    validation=Part(["va0", "va1"], ["beta", "alpha"], [False, True],
                    [[0.8, 0.6, 0.3, 0.1], [0.25, 0.2, 0.5, 0.1]]),
    test=TE,
    preds=preds_for(TE, LABELS, 0),
)  # fmt: skip
TIE_TEST = Part(["t0", "t1", "t2", "t3", "t4"], ["alpha", "beta", "gamma", "other", "beta"],
                [True, True, False, False, True], [[3.0], [-1.0], [0.1], [7.5], [0.0]])  # fmt: skip
TIE_WORLD = World(  # one column, constant over S: z = 0 for every frame
    columns=["b/beta"],
    labels=LABELS,
    train=Part(["tr0", "tr1", "tr2"], ["beta", "gamma", "beta"], [True, False, True],
               [[0.1], [NAN], [0.1]]),
    validation=Part(["va0"], ["gamma"], [False], [[0.1]]),
    test=TIE_TEST,
    preds=preds_for(TIE_TEST, LABELS, 1),
)  # fmt: skip
# H = 1, seed 3 (test_tie_precondition): b1 < 0, so h = 0 and l = b2 for every frame; only b2
# has a gradient (lambda = 0). With betas 0, eps = 1e-30 and lr = 2^30, the one Adam step is
# -2^30 sign(dL/db2_k), exact in float32, and rounds the initial |b2_k| < 1 away;
# dL/db2_k = p_k - 1/2 < 0 for beta, gamma (the two labels of S), p_k > 0 for alpha, other:
# l = (-2^30, 2^30, 2^30, -2^30), p = (0, 1/2, 1/2, 0).
TIE_CONFIG = {**CONFIG, "hidden": 1, "l2": 0.0, "lr": 2.0**30, "betas": [0.0, 0.0],
              "adam_eps": 1.0e-30, "steps": 1, "seed": 3}  # fmt: skip
