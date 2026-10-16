"""The outputs of gui-masks that gui-stack reads, written as specs/gui-masks.md (Produces)
says, without src/: `run.json` with `complete`, `features/header.json`,
`features/<part>/<name(s)>/features.npz`; runs of `main` of specs/gui-stack.md with stderr
captured and every key read from a features shard recorded."""

import contextlib
import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from numpy.lib.npyio import NpzFile
from progress_bars import tqdm_defaults

from mcpref.stacking.run import main

PARTS = ("train", "validation", "test")
TAUS = 2  # |tau_grid| of the validation arrays c, m


@dataclass(frozen=True)
class Shard:
    """The frames of one features shard; `pred` (test only): frames x 6 x 2 labels."""

    ids: list
    y: list
    g: list
    x: list  # frames x C
    pred: list = field(default_factory=list)


EMPTY = Shard([], [], [], [])


@dataclass(frozen=True)
class World:
    columns: list
    labels: list
    parts: dict  # part -> list of Shard, as many for every part (one per index shard)


def name(k: int) -> str:
    """specs/shards.md: name(k)."""
    return f"{k:06d}"


def arrays(part: str, shard: Shard, c: int, k: int) -> dict:
    """The keys of a features shard of `part`: str, bool, float64, int64 arrays."""
    n = len(shard.ids)
    out = {"ids": np.array(shard.ids, dtype=str), "y": np.array(shard.y, dtype=str)}
    out["g"] = np.array(shard.g, dtype=bool)
    out["x"] = np.array(shard.x, dtype=np.float64).reshape(n, c)
    if part == "validation":
        out["c"] = out["m"] = np.ones((n, 6, k, TAUS), dtype=np.int64)
    if part == "test":
        out["pred"] = np.array(shard.pred, dtype=str).reshape(n, 6, 2)
    return out


def write_masks(masks: Path, world: World) -> None:
    """header.json, the complete features shards of every part, then run.json with
    "complete": true."""
    features = masks / "features"
    features.mkdir(parents=True)
    header = {"columns": world.columns, "labels": world.labels}
    (features / "header.json").write_text(json.dumps(header))
    c, k = len(world.columns), len(world.labels) - 1
    for part, shards in world.parts.items():
        for s, shard in enumerate(shards):
            (features / part / name(s)).mkdir(parents=True)
            path = features / part / name(s) / "features.npz"
            np.savez(path, **arrays(part, shard, c, k))
    run = {"config": {"out_dir": "masks"}, "sha256": {"config": "0" * 64}}
    (masks / "run.json").write_text(json.dumps({**run, "complete": True}))


@dataclass
class Reads:
    """Every key read from a features shard, as (part, shard, key) in order; a read after
    which `fail(log)` holds raises OSError (an unreadable file)."""

    fail: Callable[[list], bool] | None = None
    log: list = field(default_factory=list)

    def install(self, mp: pytest.MonkeyPatch) -> None:
        get = NpzFile.__getitem__

        def getitem(npz, key):
            path = Path(npz.zip.filename)
            if path.name == "features.npz":
                self.log.append((path.parents[1].name, int(path.parent.name), key))
                if self.fail and self.fail(self.log):
                    raise OSError(f"{path}: unreadable")
            return get(npz, key)

        mp.setattr(NpzFile, "__getitem__", getitem)

    def count(self, part: str, s: int, key: str) -> int:
        return self.log.count((part, s, key))


@dataclass
class Run:
    code: int
    err: str
    out: Path
    reads: Reads

    def report(self) -> dict:
        return json.loads((self.out / "report.json").read_text())

    def prediction_files(self) -> list[Path]:
        shards = sorted((self.out / "predictions").iterdir())
        return [d / "predictions.jsonl" for d in shards]

    def predictions(self) -> list[list[dict]]:
        """The records of each prediction shard, in shard order."""
        files = self.prediction_files()
        return [[json.loads(r) for r in f.read_text().splitlines()] for f in files]

    def model(self) -> dict:
        """model.pt loaded by `torch.load(path)` with its defaults (weights_only=True)."""
        return torch.load(self.out / "model.pt")


def write_config(root: Path, repo: Path, config: dict) -> Path:
    """root/gui-stack.yaml: masks_dir = root/masks, out_dir = root/out, both relative to the
    repository root."""
    paths = {k: os.path.relpath(root / v, repo) for k, v in (("masks_dir", "masks"),
                                                            ("out_dir", "out"))}  # fmt: skip
    path = root / "gui-stack.yaml"
    path.write_text(yaml.safe_dump({**paths, **config}, sort_keys=False))
    return path


def run(root: Path, repo: Path, config: dict, fail=None) -> Run:
    """Runs main on root/gui-stack.yaml (written from `config`) from another directory, with
    tqdm's defaults (no TQDM_* variable)."""
    path = write_config(root, repo, config)
    (root / "cwd").mkdir(exist_ok=True)
    err, reads = io.StringIO(), Reads(fail)
    with pytest.MonkeyPatch.context() as mp:
        tqdm_defaults(mp)
        reads.install(mp)
        mp.chdir(root / "cwd")
        with contextlib.redirect_stderr(err):
            code = main([str(path)])
    return Run(code, err.getvalue(), root / "out", reads)


def run_main(root: Path, repo: Path, world: World, config: dict, fail=None) -> Run:
    """Writes the outputs of gui-masks for `world` under root/masks, then `run`."""
    write_masks(root / "masks", world)
    return run(root, repo, config, fail)
