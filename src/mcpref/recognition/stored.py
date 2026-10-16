"""Interface `recognition/stored.py` of specs/gui-masks.md: the files of the index, template
and features shards and of thresholds.json; the loading of the model (bar `model`)."""

import contextlib
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mcpref.gui.label_files import write_json
from mcpref.gui.screens import ScreenClass
from mcpref.recognition.drawn_masks import SCALE
from mcpref.recognition.masks import Pair
from mcpref.shards.store import shard_path, write_shard

INDEX, CATALOGUE, B, C = "index", "templates/catalogue", "templates/b", "templates/c"
FEATURES, THRESHOLDS = "features", "thresholds.json"
DRAWN, FOLDED = ("a", "d", "e", "f"), ("b", "c")
ALL = "all"
PARTS = ("train", "validation", "test")
TAU_THETA_MU = ("tau", "theta", "mu")
Model = dict[str, dict[str, dict[str, dict[str, Pair]]]]
"""(S): every template pair, build -> variant -> class -> column of (14) -> pair; build
`all` holds every variant, fold<l> the (b), (c) pairs from fold l."""


@contextlib.contextmanager
def shard_bar(
    desc: str, n_shards: int, h: int, total: int, initial: int, unit: str
) -> Iterator[Callable[..., object]]:
    """Interface Progress bars: the bar of a set of n_shards continued at shard h, or its
    line (`nothing to do`: no shard, or no item left; `already done`: every shard complete)."""
    if n_shards == 0 or (h < n_shards and total == initial):
        print(f"{desc}: nothing to do", file=sys.stderr)
        yield lambda n=1: None
    elif h == n_shards:
        print(f"{desc}: already done", file=sys.stderr)
        yield lambda n=1: None
    else:
        with tqdm(total=total, initial=initial, desc=desc, unit=unit) as bar:
            yield bar.update


def builds(k_folds: int) -> list[str]:
    """Notation: the builds b, `all`, then fold0 ... fold<K-1>."""
    return [ALL] + [f"fold{l}" for l in range(k_folds)]


def write_index(directory: Path, s: int, frames: list[dict], rng: dict) -> None:
    """Interface: index/<name(s)>/index.json = {frames, rng}."""
    with write_shard(directory, s) as staged:
        (staged / "index.json").write_text(json.dumps({"frames": frames, "rng": rng}))


def read_index(directory: Path, s: int) -> dict:
    """Interface: index/<name(s)>/index.json."""
    return json.loads((shard_path(directory, s) / "index.json").read_text())


def write_templates(
    directory: Path,
    s: int,
    name: str,
    arrays: Mapping[str, np.ndarray],
    pngs: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Interface templates/…/<name(s)>/: the .npz `name` (also without keys) and, per
    <variant>__<member or class> -> (variant, T, M), its _template.png (RGB; (f): 8-bit
    grey floor(T / 1000)) and _mask.png."""
    with write_shard(directory, s) as staged:
        np.savez(staged / name, **arrays)
        for png, (v, t, m) in pngs.items():
            image = Image.fromarray((t // SCALE).astype(np.uint8) if v == "f" else t)
            image.save(staged / f"{png}_template.png", format="PNG")
            mask = Image.fromarray(m.astype(np.uint8) * 255)
            mask.save(staged / f"{png}_mask.png", format="PNG")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Interface: every array of a .npz, read without pickles."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def read_arrays(path: Path, keys: Iterable[str]) -> list[np.ndarray]:
    """Interface: the arrays `keys` of a .npz, the others unread."""
    with np.load(path, allow_pickle=False) as z:
        return [z[key] for key in keys]


def b_path(out: Path, s: int) -> Path:
    """Interface: templates/b/<name(s)>/pairs.npz."""
    return shard_path(out / B, s) / "pairs.npz"


def b_builds(out: Path, s: int) -> list[str]:
    """Interface: the builds of (b) shard s, from its keys `<build>/T`, arrays unread."""
    with np.load(b_path(out, s), allow_pickle=False) as z:
        return [key.removesuffix("/T") for key in z.files if key.endswith("/T")]


def write_features(
    directory: Path, s: int, frames: list[dict], rows: list[dict], shapes: Mapping
) -> None:
    """Interface: features/<part>/<name(s)>/features.npz (numpy.savez): ids, y, g of
    the frames and, per key -> (shape, dtype) of `shapes`, the rows' values."""
    arrays = {
        "ids": np.array([f["id"] for f in frames], dtype=str),
        "y": np.array([f["label"] for f in frames], dtype=str),
        "g": np.array([f["g"] for f in frames], dtype=bool),
    }
    for key, (shape, dtype) in shapes.items():
        values = np.array([r[key] for r in rows], dtype=dtype)
        arrays[key] = values.reshape(len(rows), *shape)
    with write_shard(directory, s) as staged:
        np.savez(staged / "features.npz", **arrays)


def read_part(
    out: Path, part: int, counts: list[list[int]], desc: str, tally: Callable
) -> None:
    """(S) bar `thresholds`, (E) bar `report`: the features shards of part 0, 1 or 2
    (counts[s][part] frames) read one at a time into `tally`; never `already done`."""
    n = [c[part] for c in counts]
    with shard_bar(desc, len(n), 0, sum(n), 0, "frame") as update:
        for s in range(len(n)):
            z = read_npz(shard_path(out / FEATURES / PARTS[part], s) / "features.npz")
            tally(z)
            update(len(z["y"]))


def write_thresholds(path: Path, chosen: Mapping[str, Mapping[str, tuple]]) -> None:
    """Interface thresholds.json: per variant and threshold choice, {tau, theta, mu} of
    its (tau, theta, mu), theta and mu as written in the config."""
    points = {
        v: {c: dict(zip(TAU_THETA_MU, p)) for c, p in by.items()}
        for v, by in chosen.items()
    }
    write_json(path, points)


def read_thresholds(path: Path) -> dict[str, dict[str, tuple[int, Fraction, Fraction]]]:
    """Interface thresholds.json: per variant and threshold choice, (tau, theta, mu)."""
    chosen = json.loads(path.read_text())
    return {
        v: {
            c: (p["tau"], Fraction(p["theta"]), Fraction(p["mu"]))
            for c, p in by.items()
        }
        for v, by in chosen.items()
    }


def load_catalogue(pairs: dict, out: Path, s: int, k: str, cls: ScreenClass) -> None:
    """(1), (10)-(12) of class k from its catalogue shard into `pairs` (variant ->
    class -> column -> pair); (f) T with a channel axis of size 1 (Interface)."""
    z = read_npz(shard_path(out / CATALOGUE, s) / "pairs.npz")
    for v in DRAWN:
        for m in (m for m in cls.members if f"{v}/{m}/T" in z):
            t = z[f"{v}/{m}/T"]
            pair = (t[..., None] if v == "f" else t, z[f"{v}/{m}/M"])
            pairs[v].setdefault(k, {})[f"{k}__{m}"] = pair


def load_model(
    out: Path, classes: Mapping[str, ScreenClass], q: list[str], k_folds: int
) -> Model:
    """Behaviour (S): every catalogue, (b) and (c) shard read; bar `model`."""
    model = {b: {v: {} for v in FOLDED} for b in builds(k_folds)}
    model[ALL] |= {v: {} for v in DRAWN}
    templates = {}
    total = len(classes) + len(q) + k_folds + 1
    with tqdm(total=total, desc="model", unit="shard") as bar:
        for s, (k, cls) in enumerate(classes.items()):
            load_catalogue(model[ALL], out, s, k, cls)
            bar.update()
        for s, k in enumerate(q):
            for b in b_builds(out, s):
                t, m = read_arrays(b_path(out, s), (f"{b}/T", f"{b}/M"))
                templates[b, k], model[b]["b"][k] = t, {k: (t, m)}
            bar.update()
        for s, b in enumerate(builds(k_folds)):
            for k, m in read_npz(shard_path(out / C, s) / "masks.npz").items():
                model[b]["c"][k] = {k: (templates[b, k], m)}
            bar.update()
    return model
