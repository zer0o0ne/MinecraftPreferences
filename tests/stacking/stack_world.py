"""Synthetic gui-masks outputs (worlds of stack_files.py), configs of specs/gui-stack.md, and
the float64 reference of a run on a world."""

import numpy as np

from . import fit_oracle as fo
from . import stack_oracle as so
from .stack_files import EMPTY, Shard, World

VARIANTS, CHOICES = "abcdef", ("agreement", "macro_recall")
NAN = float("nan")
LABELS = ["alpha", "beta", "gamma", "other"]
K = LABELS[:-1]
# S: shards of 2, 3, 1, 1 frames, B = 2: M = 1 + 2 + 1 + 1 = 5, T = 10; checkpoints at 3, 6, 9
CONFIG = {  # every field but the paths
    "hidden": 5, "l2": 0.01, "lr": 0.05, "betas": [0.8, 0.95], "adam_eps": 1.0e-3,
    "epochs": 2, "batch": 2, "checkpoint_steps": 3, "seed": 7, "device": "cpu",
}  # fmt: skip


def with_pred(ids: list, y: list, g: list, x: list, seed: int) -> Shard:
    """A test shard with a random `pred` (frames x 6 x 2 labels of Y)."""
    pred = np.random.default_rng(seed).choice(LABELS, (len(ids), 6, 2)).tolist()
    return Shard(ids, y, g, x, pred)


# NaN in (b), (c) columns of train, c/beta NaN in every train shard; f/gamma__m 0.1 over S
# (s = 0); gamma not in S; an empty shard in every part
NAN_WORLD = World(["a/alpha", "b/beta", "c/beta", "f/gamma__m"], LABELS, {
    "train": [Shard(["tr0", "tr1"], ["alpha", "beta"], [True, False],
                    [[0.2, NAN, NAN, 0.1], [0.9, 0.5, NAN, 0.1]]),
              EMPTY,
              Shard(["tr2", "tr3", "tr4"], ["alpha", "other", "alpha"], [True, True, False],
                    [[0.3, 0.7, NAN, 0.1], [0.6, NAN, NAN, 0.1], [0.1, 0.3, NAN, 0.1]])],
    "validation": [Shard(["va0"], ["beta"], [False], [[0.8, 0.6, 0.3, 0.1]]),
                   Shard(["va1"], ["alpha"], [True], [[0.25, 0.2, 0.5, 0.1]]),
                   EMPTY],
    "test": [with_pred(["te0", "te1"], ["gamma", "alpha"], [True, False],
                       [[0.5, 0.4, 0.4, 0.9], [0.15, 0.1, 0.55, 0.1]], 0),
             EMPTY,
             with_pred(["te2", "te3"], ["beta", "other"], [True, False],
                       [[0.85, 0.65, 0.35, -3.0], [0.6, 0.5, 0.2, 0.1]], 1)],
})  # fmt: skip
# One column, constant over S (z = 0 for every frame); S = one validation shard (Q = 1)
TIE_WORLD = World(["b/beta"], LABELS, {
    "train": [EMPTY, EMPTY, EMPTY],
    "validation": [Shard(["va0", "va1", "va2", "va3"], ["beta", "gamma", "beta", "gamma"],
                         [True, False, True, False], [[0.1]] * 4), EMPTY, EMPTY],
    "test": [with_pred(["t0", "t1"], ["alpha", "beta"], [True, True], [[3.0], [-1.0]], 2),
             EMPTY,
             with_pred(["t2", "t3", "t4"], ["gamma", "other", "beta"],
                       [False, False, True], [[0.1], [7.5], [0.0]], 3)],
})  # fmt: skip
# H = 1, seed 3 (test_tie_precondition): b1 < 0, so h = 0 and l = b2 for every frame; only b2
# has a gradient (lambda = 0). M = 1, T = 1: (3') is (3). With betas 0, eps = 1e-30 and
# lr = 2^30, the one Adam step is -2^30 sign(dL/db2_k), exact in float32, and rounds the
# initial |b2_k| < 1 away; dL/db2_k = p_k - 1/2 < 0 for beta, gamma (the two labels of S),
# p_k > 0 for alpha, other: l = (-2^30, 2^30, 2^30, -2^30), p = (0, 1/2, 1/2, 0).
TIE_CONFIG = {**CONFIG, "hidden": 1, "l2": 0.0, "lr": 2.0**30, "betas": [0.0, 0.0],
              "adam_eps": 1.0e-30, "epochs": 1, "batch": 4, "seed": 3}  # fmt: skip


def s_shards(world: World) -> list[Shard]:
    """Shards of S: the non-empty train shards in order, then the non-empty validation ones."""
    return [s for p in ("train", "validation") for s in world.parts[p] if s.ids]


def matrix(shards: list[Shard], c: int) -> np.ndarray:
    return np.array([r for s in shards for r in s.x], dtype=np.float64).reshape(-1, c)


def reference(world: World, cfg: dict) -> dict:
    """A run on `world`: mu, s, z, label indices y, w of S; the shard sizes; theta_0 ..
    theta_T of (2), (4); z of each test shard."""
    shards, c = s_shards(world), len(world.columns)
    y = [world.labels.index(v) for s in shards for v in s.y]
    mu, s = so.moments(matrix(shards, c))
    z, w = so.standardised(matrix(shards, c), mu, s), so.weights(y)
    sizes = [len(sh.ids) for sh in shards]
    theta0 = so.init(c, cfg["hidden"], len(world.labels), cfg["seed"])
    path = [theta0] + [t for t, _, _ in fo.adam_path(theta0, z, y, w, sizes, cfg)]
    z_te = [so.standardised(matrix([t], c), mu, s) for t in world.parts["test"]]
    return {"mu": mu, "s": s, "z": z, "y": y, "w": w, "sizes": sizes, "path": path,
            "z_te": z_te}  # fmt: skip
