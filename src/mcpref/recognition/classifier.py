"""specs/gui-classifier.md: the screen classifier gamma, variant (d) of specs/gui-masks.md in
the gathered form (1), loaded by (L) from the outputs of block gui-masks."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from mcpref import REPO, config_mapping, repo_path
from mcpref.gui.label_files import sha256
from mcpref.gui.screens import ScreenClass, load_screens
from mcpref.recognition.metrics import CRITERIA
from mcpref.recognition.recognise import predict
from mcpref.recognition.stored import (
    CATALOGUE,
    THRESHOLDS,
    complete_run,
    read_thresholds,
    shard_bar,
)
from mcpref.shards.store import shard_path

VARIANT = "d"
Gathered = tuple[np.ndarray, np.ndarray]
"""Notation: (I, T_I) of one pair, T_I as int16."""


@dataclass(frozen=True)
class GuiClassifierConfig:
    """Notation: configs/gui-classifier.yaml; masks_dir as written (messages of (L)),
    resolved against the repository root where used."""

    masks_dir: str
    criterion: str


def load_config(path: Path) -> GuiClassifierConfig:
    """Notation: the YAML file loaded into GuiClassifierConfig; violated constraints raise."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, GuiClassifierConfig)
    repo_path("masks_dir", raw["masks_dir"])
    criterion, names = raw["criterion"], list(CRITERIA)
    if criterion not in names:
        raise ValueError(f"criterion {criterion!r} is not one of {names}")
    return GuiClassifierConfig(**raw)


@dataclass(frozen=True)
class Model:
    """(L): I and T_I of every pair concatenated in class and member order, each pair's
    start offset and |M|, the positions of the pairs of P_k per class with a pair."""

    flat: np.ndarray
    t_i: np.ndarray
    starts: np.ndarray
    sizes: list[int]
    pairs: dict[str, slice]


def counts(x: np.ndarray, model: Model, tau: int) -> list[int]:
    """(1): c of every pair, by one gather of the masks' pixels of x."""
    gathered = x.reshape(-1, 3)[model.flat].astype(np.int16)
    within = np.abs(gathered - model.t_i).max(axis=-1) <= tau
    return np.add.reduceat(within, model.starts, dtype=np.int64).tolist()


def class_scores(c: list[int], model: Model) -> dict[str, Fraction]:
    """(1): S_k = max over the pairs of P_k of c / |M|; classes without a pair absent."""
    return {k: max(map(Fraction, c[r], model.sizes[r])) for k, r in model.pairs.items()}


def classify(
    frames: list[np.ndarray],
    model: Model,
    order: list[str],
    tau: int,
    theta: Fraction,
    mu: Fraction,
) -> list[str]:
    """(1), (2): gui-masks (5) on the scores of each frame, independently of its batch."""
    return [
        predict(class_scores(counts(x, model, tau), model), order, theta, mu)
        for x in frames
    ]


def check_run(masks_dir: str, screens: str) -> str:
    """(L) steps 1, 2: run.json complete and built from the caller's catalogue; h_K."""
    run = complete_run(REPO / masks_dir, masks_dir)
    h_k = sha256(REPO / screens)
    if run["sha256"]["screens"] != h_k:
        raise RuntimeError(
            f"{masks_dir} was built from another catalogue than {screens}"
        )
    return h_k


def shard_pairs(
    path: Path, masks_dir: str, k: str, cls: ScreenClass, size: tuple[int, int], h_t
) -> list[Gathered]:
    """(L) step 3 for class k: one numpy.load, only the d/... arrays; per pair in member
    order the checks, then (I, T_I) kept and the pair's bytes of (3) fed to h_t."""
    width, height = size
    kept = []
    with np.load(path, allow_pickle=False) as z:
        for m in (m for m in cls.members if f"{VARIANT}/{m}/T" in z.files):
            t = z[f"{VARIANT}/{m}/T"]
            if t.shape != (height, width, 3):
                h, w = t.shape[:2]
                raise RuntimeError(
                    f"{masks_dir} holds {w}x{h} templates, not {width}x{height}"
                )
            i = np.flatnonzero(z[f"{VARIANT}/{m}/M"])
            if i.size == 0:
                raise RuntimeError(f"{masks_dir}: the mask of d/{m} of {k} is empty")
            t_i = t.reshape(-1, 3)[i]
            h_t.update(
                f"{k}/{m}\n".encode() + i.astype("<i8").tobytes() + t_i.tobytes()
            )
            kept.append((i, t_i.astype(np.int16)))
    return kept


def concatenated(per_class: Mapping[str, list[Gathered]]) -> Model:
    """(L): the pairs of every class concatenated in order, with their start offsets."""
    pairs = [p for ps in per_class.values() for p in ps]
    sizes = np.array([i.size for i, _ in pairs], dtype=np.int64)
    n = np.cumsum([0] + [len(ps) for ps in per_class.values()]).tolist()
    return Model(
        flat=np.concatenate([np.empty(0, np.int64), *(i for i, _ in pairs)]),
        t_i=np.concatenate([np.empty((0, 3), np.int16), *(t for _, t in pairs)]),
        starts=np.cumsum(sizes) - sizes,
        sizes=sizes.tolist(),
        pairs={k: slice(a, b) for k, a, b in zip(per_class, n, n[1:]) if b > a},
    )


def read_model(
    masks_dir: str, classes: Mapping[str, ScreenClass], size: tuple[int, int]
) -> tuple[Model, str]:
    """(L) step 3: catalogue shards s = 0..|K|-1 one at a time (bar `classifier`); the
    model and h_T of (3)."""
    directory, h_t, per_class = REPO / masks_dir / CATALOGUE, hashlib.sha256(), {}
    with shard_bar("classifier", len(classes), 0, len(classes), 0, "shard") as update:
        for s, (k, cls) in enumerate(classes.items()):
            path = shard_path(directory, s) / "pairs.npz"
            per_class[k] = shard_pairs(path, masks_dir, k, cls, size, h_t)
            update()
    return concatenated(per_class), h_t.hexdigest()


def identifier(tau: int, theta: Fraction, mu: Fraction, h_t: str, h_k: str) -> str:
    """(3): gamma_id."""
    return json.dumps(
        {
            "variant": VARIANT,
            "tau": tau,
            "theta": str(theta),
            "mu": str(mu),
            "templates": h_t,
            "catalogue": h_k,
        }
    )


def load_classifier(
    cfg: GuiClassifierConfig, screens: str, width: int, height: int
) -> tuple[Callable[[list[np.ndarray]], list[str]], str]:
    """(L): gamma of (1), (2) and gamma_id of (3); screens as written in the caller's
    config, resolved against the repository root."""
    h_k = check_run(cfg.masks_dir, screens)
    classes = load_screens(REPO / screens)
    model, h_t = read_model(cfg.masks_dir, classes, (width, height))
    chosen = read_thresholds(REPO / cfg.masks_dir / THRESHOLDS)
    tau, theta, mu = chosen[VARIANT][cfg.criterion]
    gamma = partial(
        classify, model=model, order=list(classes), tau=tau, theta=theta, mu=mu
    )
    return gamma, identifier(tau, theta, mu, h_t, h_k)
