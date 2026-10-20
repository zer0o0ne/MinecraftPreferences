"""Fake outputs of block gui-masks as its Produces says (specs/gui-masks.md: run.json, the
catalogue template shards, thresholds.json) with a tiny catalogue in the YAML forms of
specs/gui-screens.md, for the readers of specs/gui-classifier.md (its tests and those of
`main` of specs/perception-data.md); gamma_id of gui-classifier (3), transcribed; its
`load_classifier` on them."""

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

from mcpref.recognition.classifier import GuiClassifierConfig, load_classifier

CRITERIA = ("agreement", "macro_recall")  # gui-masks (6), the order of CRITERIA
VARIANTS = "abcdef"  # the variants of thresholds.json
TEXTURES = {"url": "http://127.0.0.1:9/c.jar", "sha1": "0" * 40, "out_dir": "no-tex"}
TEXTURES.update(http_timeout=1.0, http_retries=0, retry_delay=0.0)
OTHERS = {"tau": 0, "theta": "1", "mu": "1"}  # thresholds of the variants but (d)

Pair = tuple[np.ndarray, np.ndarray]  # (T uint8 H x W x 3, M bool H x W) of variant (d)


def rel(path: Path, repo: Path) -> str:
    """A path relative to the repository root, as configs write it."""
    return os.path.relpath(path, repo)


def elsewhere(tmp: Path, repo: Path) -> Path:
    """A new directory from which the paths relative to the repository root of files
    under `tmp` lead nowhere."""
    cwd = tmp.joinpath(*["cwd"] * (rel(tmp, repo).count("..") + 1))
    cwd.mkdir(parents=True)
    return cwd


def catalogue_text(classes: dict, note: str = "") -> str:
    """A catalogue with the classes and members of `classes` (class -> member -> ...), in
    that order; `note` (an ignored `note` string) changes its bytes, not its classes."""
    entry = {"blits": [], "dynamic": [], "labels": []}
    data = {
        k: {
            "members": {m: entry for m in members},
            "description": f"{k}.",
            "note": note,
        }
        | {"window": [0, 0, 4, 4], "elements": []}
        for k, members in classes.items()
    }
    return yaml.safe_dump(
        {"classes": data, "textures": TEXTURES}, sort_keys=False, allow_unicode=True
    )


def decoys(member: str, pair: Pair | None, width: int, height: int) -> dict:
    """The (a), (e), (f) arrays of a member: inverted (d) templates under full masks, so a
    classifier reading them labels frames otherwise."""
    t = np.full((height, width, 3), 7, np.uint8) if pair is None else 255 - pair[0]
    full = np.ones((height, width), bool)
    f = t.astype(np.int64).sum(axis=2) * 1000
    arrays = {f"{v}/{member}/T": t for v in "ae"} | {f"f/{member}/T": f}
    return arrays | {f"{v}/{member}/M": full for v in "aef"}


def shard_arrays(members: dict, width: int, height: int) -> dict:
    """Keys of a catalogue shard, members in reverse catalogue order (pairs are taken in
    member order, not file order)."""
    arrays = {}
    for m, pair in reversed(members.items()):
        if pair is not None:
            arrays |= {f"d/{m}/T": pair[0], f"d/{m}/M": pair[1]}
        arrays |= decoys(m, pair, width, height)
    return arrays


@dataclass
class Masks:
    """A gui-masks `out_dir` root/masks and its catalogue root/screens.yaml. `classes`:
    class -> member -> its (d) pair, or None for a member without (d) keys, in catalogue
    order; `thresholds`: criterion -> (tau, theta, mu) of variant (d), theta and mu as the
    strings written."""

    root: Path
    repo: Path
    classes: dict
    thresholds: dict
    width: int
    height: int
    note: str = ""

    @property
    def out(self) -> Path:
        return self.root / "masks"

    @property
    def catalogue(self) -> Path:
        return self.root / "screens.yaml"

    @property
    def masks_dir(self) -> str:
        return rel(self.out, self.repo)

    @property
    def screens(self) -> str:
        return rel(self.catalogue, self.repo)

    def write(self) -> "Masks":
        """Every file, run.json last with "complete": true."""
        self.out.mkdir(parents=True)
        self.catalogue.write_text(catalogue_text(self.classes, self.note))
        for s, members in enumerate(self.classes.values()):
            path = self.out / "templates/catalogue" / f"{s:06d}" / "pairs.npz"
            path.parent.mkdir(parents=True)
            np.savez(path, **shard_arrays(members, self.width, self.height))
        self.write_thresholds()
        self.write_run(complete=True)
        return self

    def write_thresholds(self) -> None:
        d = {
            c: dict(zip(("tau", "theta", "mu"), t)) for c, t in self.thresholds.items()
        }
        doc = {v: d if v == "d" else dict.fromkeys(CRITERIA, OTHERS) for v in VARIANTS}
        (self.out / "thresholds.json").write_text(json.dumps(doc, indent=2) + "\n")

    def write_run(self, complete: bool | None) -> None:
        """run.json of gui-masks (R); `complete` None: as written at the start of its run."""
        run = {"config": {"tau_grid": [0]}, "sha256": {"config": "0" * 64}}
        run["sha256"]["screens"] = hashlib.sha256(
            self.catalogue.read_bytes()
        ).hexdigest()
        run |= {"frames": {"shards": 1, "ok": 1, "failed": 0}, "labels": "1" * 64}
        if complete is not None:
            run["complete"] = complete
        (self.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    def pairs(self) -> dict:
        """class -> its (d) pairs in member order, classes without a pair left out."""
        out = {
            k: [p for p in ms.values() if p is not None]
            for k, ms in self.classes.items()
        }
        return {k: ps for k, ps in out.items() if ps}


def h_templates(classes: dict, width: int) -> str:
    """(3): h_T over the classes in order and their (d) pairs in member order, with
    I = {W r + q : M(r, q)} ascending and T_I = T at I."""
    h = hashlib.sha256()
    for k, members in classes.items():
        for m, pair in members.items():
            if pair is None:
                continue
            t, mask = pair
            rows, cols = mask.shape
            i = [width * r + q for r in range(rows) for q in range(cols) if mask[r, q]]
            h.update(f"{k}/{m}\n".encode())
            h.update(b"".join(struct.pack("<q", j) for j in i))
            pixels = (t[j // width, j % width] for j in i)
            h.update(bytes(int(v) for p in pixels for v in p))
    return h.hexdigest()


def gamma_id(masks: Masks, criterion: str) -> str:
    """(3): gamma_id of the classifier of `masks` under `criterion`."""
    tau, theta, mu = masks.thresholds[criterion]
    doc = {"variant": "d", "tau": tau, "theta": str(Fraction(theta))}
    doc |= {
        "mu": str(Fraction(mu)),
        "templates": h_templates(masks.classes, masks.width),
    }
    doc["catalogue"] = hashlib.sha256(masks.catalogue.read_bytes()).hexdigest()
    return json.dumps(doc)


def write_classifier_config(path: Path, masks: Masks, criterion: str) -> Path:
    """configs/gui-classifier.yaml of `masks`."""
    doc = {"masks_dir": masks.masks_dir, "criterion": criterion}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def both(tau: int, theta: str, mu: str) -> dict:
    """The same thresholds of (d) for every criterion."""
    return dict.fromkeys(CRITERIA, (tau, theta, mu))


def load(masks: Masks, criterion: str = "agreement", **written):
    """`load_classifier` on `masks` at its size; `written` replaces masks_dir, screens,
    width or height as the caller writes them."""
    args = {"masks_dir": masks.masks_dir, "screens": masks.screens} | written
    size = written.get("width", masks.width), written.get("height", masks.height)
    cfg = GuiClassifierConfig(masks_dir=args["masks_dir"], criterion=criterion)
    return load_classifier(cfg, args["screens"], *size)
