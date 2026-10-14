"""The files of a world of masks_world.py as `main` of specs/gui-masks.md consumes them:
frames (specs/vpt-gui-frames.md), labels (specs/gui-labels.md), catalogue and textures
(specs/gui-screens.md), config."""

import json
import os
from pathlib import Path

import masks_world as mw
import numpy as np
import yaml
from PIL import Image

TEXTURES = {"url": "http://127.0.0.1:9/c.jar", "sha1": "0" * 40}
TEXTURES.update(http_timeout=1.0, http_retries=0, retry_delay=0.0)


def write_frames(root: Path, fs: list) -> None:
    """frames.jsonl, run.json, frames/<id>.yuv (U = V = 128) of specs/vpt-gui-frames.md."""
    (root / "frames").mkdir(parents=True)
    recs, chroma = [], bytes([128]) * (mw.H * mw.W // 2)
    for i, f in enumerate(fs):
        line = {"isGuiOpen": True, "isGuiInventory": f.flag}
        recs.append(
            {"id": f.id, "subset": "7xx", "relpath": f.relpath, "version": "7.0"}
        )
        recs[-1].update(frame=i, prev_line=line, line=line, status=f.status)
        if f.y is not None:
            data = f.y.astype(np.uint8).tobytes() + chroma
            (root / "frames" / f"{f.id}.yuv").write_bytes(data)
    (root / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    run = {"config": {"width": mw.W, "height": mw.H}, "index_etags": {}, "episodes": {}}
    (root / "run.json").write_text(json.dumps({**run, "frames": {}}))


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
    """Writes every input under root, config paths relative to the repository root;
    returns the config path."""
    rel = {
        p: os.path.relpath(root / p, repo)
        for p in ("frames", "labels", "screens.yaml", "tex", "out")
    }
    write_textures(root / "tex", mw.textures())
    write_screens(root / "screens.yaml", catalogue, rel["tex"])
    write_frames(root / "frames", fs)
    (root / "labels").mkdir()
    ok = [f for f in fs if f.status == "ok" and f.label is not None]
    lines = [{"id": f.id, "seed": 0, "answer": f.label, "label": f.label} for f in ok]
    (root / "labels/labels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in lines)
    )
    cfg = {"frames_dir": rel["frames"], "labels_dir": rel["labels"]}
    cfg.update(screens=rel["screens.yaml"], **config, out_dir=rel["out"])
    (root / "gui-masks.yaml").write_text(yaml.safe_dump(cfg))
    return root / "gui-masks.yaml"
