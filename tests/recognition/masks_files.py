"""The files of a world of masks_world.py as `main` of specs/gui-masks.md consumes them: plan
shards, frames and run.json (specs/vpt-gui-frames.md, Produces), label shards and run.json
(specs/gui-labels.md, Produces), catalogue and textures (specs/gui-screens.md), config."""

import itertools
import json
import os
from pathlib import Path

import masks_world as mw
import numpy as np
import yaml
from PIL import Image

TEXTURES = {"url": "http://127.0.0.1:9/c.jar", "sha1": "0" * 40}
TEXTURES.update(http_timeout=1.0, http_retries=0, retry_delay=0.0)


BOUNDS = (3, 3, 4, 20, 37)  # plan shard k = positions [BOUNDS[k - 1], BOUNDS[k])


def plan_shards(fs: list) -> list:
    """The frames fs (position order) cut into plan shards: the first with the first failed
    record, an empty one, one holding only the ghost's failed record, then three whose
    episodes span shards."""
    cuts = (0, *BOUNDS, max(len(fs), BOUNDS[-1]))
    return [fs[a:b] for a, b in itertools.pairwise(cuts)]


def name(k: int) -> str:
    """name(k) of specs/shards.md."""
    return f"{k:06d}"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def counts(shards: list) -> dict:
    """`frames` of {frames_dir}/run.json: statuses that occurred only."""
    fs = [f for s in shards for f in s]
    n = {s: sum(f.status == s for f in fs) for s in ("ok", "failed")}
    return {s: c for s, c in n.items() if c}


def fingerprint(shards: list) -> dict:
    """`frames` of specs/gui-masks.md (R) and of {labels_dir}/run.json: shards, ok, failed."""
    return {"shards": len(shards), "ok": 0, "failed": 0, **counts(shards)}


def plan_record(f, i: int) -> dict:
    line = {"isGuiOpen": True, "isGuiInventory": f.flag}
    rec = {"id": f.id, "subset": "7xx", "relpath": f.relpath, "version": "7.0"}
    return {**rec, "frame": i, "prev_line": line, "line": line}


def write_frames(root: Path, shards: list) -> None:
    """plan/<name(k)>/shard.json, frames/<id>.yuv of the `ok` frames (U = V = 128), run.json
    with its counts."""
    (root / "frames").mkdir(parents=True)
    chroma = bytes([128]) * (mw.H * mw.W // 2)
    for k, fs in enumerate(shards):
        relpaths = dict.fromkeys(f.relpath for f in fs)
        episodes = [
            {"subset": "7xx", "relpath": r, "reason": None, "options": {}}
            for r in relpaths
        ]
        frames = [plan_record(f, 100 + i) for i, f in enumerate(fs)]
        rng = np.random.Generator(np.random.PCG64(k)).bit_generator.state
        data = {"episodes": episodes, "frames": frames, "rng": rng}
        write_json(root / "plan" / name(k) / "shard.json", data)
        for f in (f for f in fs if f.status == "ok"):
            data = f.y.astype(np.uint8).tobytes() + chroma
            (root / "frames" / f"{f.id}.yuv").write_bytes(data)
    config = {"width": mw.W, "height": mw.H, "seed": 0}
    run = {"config": config, "episodes": {"null": 1}, "frames": counts(shards)}
    write_json(root / "run.json", run)


def write_labels(root: Path, shards: list) -> None:
    """labels/<name(k)>/labels.json (records of the `ok` frames, `end`) and run.json."""
    end = 0
    for k, fs in enumerate(shards):
        recs = [
            {"id": f.id, "seed": end + i, "answer": f.label, "label": f.label}
            | {"gui_inventory": f.flag}
            for i, f in enumerate(fs)
            if f.status == "ok"
        ]
        end += len(fs)
        write_json(
            root / "labels" / name(k) / "labels.json", {"labels": recs, "end": end}
        )
    sha = {"config": "0" * 64, "vlm": "1" * 64, "screens": "2" * 64}
    run = {"config": {"seed": 0}, "sha256": sha, "frames": fingerprint(shards)}
    write_json(root / "run.json", {**run, "prompt": ["-", "<image>", "-"]})


def write_textures(root: Path, images: dict) -> None:
    for name, image in images.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.astype(np.uint8)).save(root / name)


def write_screens(path: Path, classes: dict, out_dir: str) -> Path:
    """A catalogue in the YAML forms of specs/gui-screens.md; textures under `out_dir`."""
    data = {"classes": classes, "textures": {**TEXTURES, "out_dir": out_dir}}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def write(
    root: Path, repo: Path, fs: list, config: dict, catalogue=mw.CATALOGUE
) -> Path:
    """Writes every input under root, the frames fs cut by plan_shards, config paths
    relative to the repository root; returns the config path."""
    rel = {
        p: os.path.relpath(root / p, repo)
        for p in ("frames", "labels", "screens.yaml", "tex", "out")
    }
    write_textures(root / "tex", mw.textures())
    write_screens(root / "screens.yaml", catalogue, rel["tex"])
    shards = plan_shards(fs)
    write_frames(root / "frames", shards)
    write_labels(root / "labels", shards)
    cfg = {"frames_dir": rel["frames"], "labels_dir": rel["labels"]}
    cfg.update(screens=rel["screens.yaml"], **config, out_dir=rel["out"])
    (root / "gui-masks.yaml").write_text(yaml.safe_dump(cfg))
    return root / "gui-masks.yaml"
