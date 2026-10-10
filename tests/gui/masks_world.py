"""Synthetic inputs of `main` of specs/gui-masks.md (catalogue, textures, frames, labels,
config) and the report expected by masks_oracle.py."""

import itertools
import json
import os
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import masks_oracle as o
import numpy as np
import yaml
from PIL import Image

W, H = 16, 8
CLASSES = ["alpha", "beta", "gamma", "delta"]
LABELS = [*CLASSES, "other"]
CRITERIA = ("agreement", "macro_recall")
N_EP = 18
EPISODES = [f"data/7.0/ep{7 * e % N_EP:02d}.mp4" for e in range(N_EP)]  # not sorted
CONFIG = {"split": ["0.5", "0.25", "0.25"], "seed": 230, "min_episodes": 2}
CONFIG.update(spread_quantile=0.75, spread_max=20, patch=4, patches_per_class=3, eps=1)
CONFIG.update(tau_grid=[12, 4, 40], theta_grid=["0.9", "0.5", "0.75"])
CONFIG.update(mu_grid=["0", "0.25", "0.1"], chunk_rows=3)


def blit(texture: str, u: int, v: int, w: int, h: int, x: int, y: int) -> dict:
    return {"texture": texture, "u": u, "v": v, "w": w, "h": h, "x": x, "y": y}


def named(rects, prefix: str) -> list:
    return [{"name": f"{prefix}{i}", "rect": r} for i, r in enumerate(rects)]


def member(blits: list, dynamic=(), labels=()) -> dict:
    return {
        "blits": blits,
        "dynamic": named(dynamic, "d"),
        "labels": named(labels, "l"),
    }


def whole(texture: str, w: int, x: int) -> list:
    """One blit of the whole texture (w x H) at (x, 0)."""
    return [blit(texture, 0, 0, w, H, x, 0)]


def screen(members: dict, window: list, x: int, y: int) -> dict:
    go = {"index": 37, "name": "go", "kind": "button", "hit": [x, y, 2, 2]}
    go["click"] = [x + 1.0, y + 1.0]
    return {"members": members, "description": "-", "window": window, "elements": [go]}


ONE = member(whole("gui/alpha1.png", 8, 0), dynamic=[[0, 0, 2, 1]])
TWO = member(whole("gui/alpha2.png", 8, 0))
DELTA = member(whole("gui/delta.png", W, 0), labels=[[0, 0, 4, 2]])
CATALOGUE = {
    "alpha": screen({"one": ONE, "two": TWO}, [0, 0, 8, 8], 6, 6),
    "beta": screen({"only": member(whole("gui/beta.png", 8, 8))}, [8, 0, 8, 8], 14, 0),
    "gamma": screen({"only": member(whole("gui/clear.png", W, 0))}, [0, 0, W, H], 0, 6),
    "delta": screen({"only": DELTA}, [0, 0, W, H], 14, 6),
}
_rng = np.random.default_rng(2026)
PATTERNS = {k: _rng.integers(16, 236, (H, 8 if k == "alpha" else W)) for k in CLASSES}
FAR_FROM_125 = [
    *range(16, 80),
    *range(172, 236),
]  # beta's inverse 251 - Y differs a lot
PATTERNS["beta"] = _rng.choice(FAR_FROM_125, (H, 8))
MISLABEL = {"alpha": "beta", "beta": "delta", "gamma": "other", "delta": "other"}
MISLABEL["other"] = "alpha"


def grey(y: np.ndarray) -> np.ndarray:
    """vpt-gui-frames (5) with U = V = 128: R = G = B = floor(255/219 (Y - 16) + 0.5), clipped."""
    v = np.floor(255 / 219 * (y.astype(np.float64) - 16) + 0.5)
    return np.clip(v, 0, 255).astype(np.int64)


def rgba(y: np.ndarray, alpha: np.ndarray, shift=(0, 0, 0)) -> np.ndarray:
    rgb = np.clip(grey(y)[..., None] + np.array(shift), 0, 255)
    return np.concatenate([rgb, alpha[..., None]], axis=2)


def textures() -> dict:
    """The classes' own colours: alpha1 with two holes, alpha2 brighter on the left, beta less
    red, delta transparent where its frames vary, clear fully transparent."""
    opaque = np.full((H, 8), 255)
    holes = opaque.copy()
    holes[7, 0] = holes[3, 3] = 0
    brighter = PATTERNS["alpha"] + np.where(np.arange(8) < 4, 26, 0)
    corner = np.full((H, W), 255)
    corner[:4, 12:] = 0
    return {
        "gui/alpha1.png": rgba(PATTERNS["alpha"], holes),
        "gui/alpha2.png": rgba(brighter, opaque),
        "gui/beta.png": rgba(PATTERNS["beta"], opaque, (-3, 0, 0)),
        "gui/delta.png": rgba(PATTERNS["delta"], corner),
        "gui/clear.png": np.zeros((H, W, 4), dtype=np.int64),
    }


@dataclass
class Frame:
    id: str
    relpath: str
    status: str
    flag: bool  # line.isGuiInventory
    label: str | None  # VLM label; None: no record in labels.jsonl
    y: np.ndarray | None  # Y plane; None for a failed frame


def y_plane(cls: str, partial: bool, rng) -> np.ndarray:
    """alpha: its half (partial: a quarter covered); beta: its half, the pattern or its inverse
    251 - Y (high spread); gamma, delta: the whole frame, delta random in its transparent
    corner; other: random (partial: alpha's top half). Noise +-5 in Y."""
    y = rng.integers(16, 236, (H, W))
    noise = rng.integers(-5, 6, (H, W))
    if cls == "alpha":
        cover = y[:4, :4].copy()
        y[:, :8] = PATTERNS["alpha"] + noise[:, :8]
        if partial:
            y[:4, :4] = cover
    elif cls == "other" and partial:
        y[:4, :8] = PATTERNS["alpha"][:4] + noise[:4, :8]
    elif cls == "beta":
        y[:, 8:] = PATTERNS["beta"] if rng.integers(2) else 251 - PATTERNS["beta"]
        y[:, 8:] += noise[:, 8:]
    elif cls in ("gamma", "delta"):
        corner = y[:4, 12:].copy()
        y = PATTERNS[cls] + noise
        if cls == "delta":
            y[:4, 12:] = corner
    return np.clip(y, 16, 235)


def frames(relabel: bool = False) -> list:
    """Episodes interleaved, 3 frames each; episode 5 shows gamma; mislabels, invalid,
    no_answer, a missing label; failed records first (of the last episode, of a ghost).
    relabel: the first episode unlabelled, episode 4 invalid."""
    rng, out = np.random.default_rng(7), []
    special = {(2, 1): "invalid", (9, 0): "no_answer", (12, 2): None}
    for s, e in itertools.product(range(3), range(N_EP)):
        cls = "gamma" if e == 5 else ["alpha", "beta", "delta", "other"][(e + s) % 4]
        n = 3 * e + s
        label = special.get((e, s), MISLABEL[cls] if n % 10 == 7 else cls)
        if relabel and e in (0, 4):
            label = None if e == 0 else "invalid"
        stem, flag = Path(EPISODES[e]).stem, (cls == "alpha") != (n % 5 == 0)
        y = y_plane(cls, n % 3 == 2, rng)
        out.append(Frame(f"7.0_{stem}_{s + 10}", EPISODES[e], "ok", flag, label, y))
    out.insert(0, Frame("7.0_last_1", EPISODES[-1], "failed", False, None, None))
    out.insert(
        3, Frame("7.0_ghost_1", "data/7.0/ghost.mp4", "failed", True, None, None)
    )
    return out


def write_frames(root: Path, fs: list) -> None:
    """frames.jsonl, run.json, frames/<id>.yuv (U = V = 128) of specs/vpt-gui-frames.md."""
    (root / "frames").mkdir(parents=True)
    recs, chroma = [], bytes([128]) * (H * W // 2)
    for i, f in enumerate(fs):
        line = {"isGuiOpen": True, "isGuiInventory": f.flag}
        recs.append(
            {"id": f.id, "subset": "7xx", "relpath": f.relpath, "version": "7.0"}
        )
        recs[-1].update(frame=i, prev_line=line, line=line, status=f.status)
        if f.y is not None:
            (root / "frames" / f"{f.id}.yuv").write_bytes(
                f.y.astype(np.uint8).tobytes() + chroma
            )
    (root / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    run = {
        "config": {"width": W, "height": H},
        "index_etags": {},
        "episodes": {},
        "frames": {},
    }
    (root / "run.json").write_text(json.dumps(run))


def write(root: Path, repo: Path, fs: list, config: dict) -> Path:
    """Writes every input under root, config paths relative to the repository root;
    returns the config path."""
    rel = {
        p: os.path.relpath(root / p, repo)
        for p in ("frames", "labels", "screens.yaml", "tex", "out")
    }
    for name, image in textures().items():
        (root / "tex" / name).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.astype(np.uint8)).save(root / "tex" / name)
    tex = {"url": "http://127.0.0.1:9/c.jar", "sha1": "0" * 40, "out_dir": rel["tex"]}
    tex.update(http_timeout=1.0, http_retries=0, retry_delay=0.0)
    (root / "screens.yaml").write_text(
        yaml.safe_dump({"classes": CATALOGUE, "textures": tex})
    )
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


def parts(fs: list, config: dict) -> dict:
    """Split: one draw per episode of the `ok` records, in order of first appearance."""
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    probs = [float(Fraction(p)) for p in config["split"]]
    episodes = dict.fromkeys(f.relpath for f in fs if f.status == "ok")
    return {e: int(rng.choice(3, p=probs)) for e in episodes}


def build(tr: list, x: dict, config: dict) -> tuple:
    """(1)-(3): Q, variant -> class -> pairs, and the images: path stem -> (RGB, mask)."""
    pairs, images, tex = {"a": {}, "b": {}, "c": {}}, {}, textures()
    for k in CLASSES:
        for m, (canvas, mask) in o.outline_pairs(CATALOGUE[k], tex, W, H).items():
            pairs["a"].setdefault(k, []).append((canvas[..., :3], mask))
            images[f"a/{k}__{m}"] = (
                np.where(canvas[..., 3:] == 255, canvas[..., :3], 0),
                mask,
            )
    episodes = {k: {f.relpath for f in tr if f.label == k} for k in CLASSES}
    qualified = [k for k in CLASSES if len(episodes[k]) >= config["min_episodes"]]
    q, s_max = config["spread_quantile"], config["spread_max"]
    spread = {
        k: o.spread_pair([x[f.id] for f in tr if f.label == k], q, s_max)
        for k in qualified
    }
    spread = {k: v for k, v in spread.items() if v[2].any()}
    pairs["b"] = {k: [(t, m)] for k, (t, _, m) in spread.items()}
    if len(spread) >= 2:
        args = [{k: v[i] for k, v in spread.items()} for i in (0, 1)]
        masks = o.patch_masks(
            *args, config["patch"], config["patches_per_class"], config["eps"]
        )
        pairs["c"] = {k: [(spread[k][0], masks[k])] for k in masks}
    for v in "bc":
        images.update({f"{v}/{k}": p[0] for k, p in pairs[v].items()})
    return qualified, pairs, images


def evaluation(te: list, yhat: list) -> dict:
    """(7) on F^te and on its parts by g_f."""

    def on(rows: list) -> dict:
        return o.metrics(*([r[i] for r in rows] for i in range(3)), LABELS)

    rows = [(f.id, f.label, p, f.flag) for f, p in zip(te, yhat)]
    by_flag = {
        str(g).lower(): on([r for r in rows if r[3] == g]) for g in (True, False)
    }
    return {**on(rows), "by_flag": by_flag}


def variant(pairs: dict, x: dict, va: list, te: list, grid: tuple) -> tuple:
    """(4)-(7) of one variant: criterion -> {thresholds, test}, and the test predictions."""
    s = {
        (f.id, t): {k: o.score(x[f.id], p, t) for k, p in pairs.items()}
        for f in va + te
        for t in grid[0]
    }
    preds = {
        pt: [o.predict(s[f.id, pt[0]], CLASSES, *pt[1:]) for f in va]
        for pt in itertools.product(*grid)
    }
    out, yhats = {}, {}
    for crit, (t, th, mu) in o.choose([f.label for f in va], preds, grid).items():
        yhats[crit] = [o.predict(s[f.id, t], CLASSES, th, mu) for f in te]
        thresholds = {"tau": t, "theta": float(th), "mu": float(mu)}
        out[crit] = {"thresholds": thresholds, "test": evaluation(te, yhats[crit])}
    return out, yhats


def expected(fs: list, config: dict) -> dict | None:
    """The report of `main` by the oracle (None: empty validation set), with `images`,
    `part` (episode -> part) and `n_scored` (validation and test frames)."""
    part = parts(fs, config)
    used = [
        f
        for f in fs
        if f.status == "ok" and f.label not in (None, "invalid", "no_answer")
    ]
    x = {f.id: np.repeat(grey(f.y)[..., None], 3, axis=2) for f in used}
    tr, va, te = [[f for f in used if part[f.relpath] == i] for i in range(3)]
    if not va:
        return None
    qualified, pairs, images = build(tr, x, config)
    grid = (
        config["tau_grid"],
        *[[Fraction(t) for t in config[g]] for g in ("theta_grid", "mu_grid")],
    )
    report = {"qualified": qualified, "images": images, "part": part}
    report.update(split=[Counter(f.label for f in p) for p in (tr, va, te)])
    report["split"] = [{k: c[k] for k in LABELS} for c in report["split"]]
    labellers = {crit: [[f.label for f in te]] for crit in CRITERIA}
    for v in "abc":
        report[v], yhats = variant(pairs[v], x, va, te, grid)
        report[v]["pairs"] = pairs[v]
        for crit in CRITERIA:
            labellers[crit].append(yhats[crit])
    report["agreement_matrices"] = {
        c: o.matrices(CLASSES, labellers[c]) for c in CRITERIA
    }
    return {**report, "n_scored": len(va) + len(te)}
