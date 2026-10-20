"""(R), (I), Notation and (13) of specs/gui-masks.md: the checks, the index shards (bar
`index`), the state: the split, the counts, Q, the folds, the training frames."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mcpref import REPO
from mcpref.gui.label_files import (
    INVALID,
    LABELS,
    NO_ANSWER,
    OTHER,
    frame_size,
    labelled,
    read_labels,
    run_counts,
    yuv_path,
)
from mcpref.recognition.stored import INDEX, read_index, shard_bar, write_index
from mcpref.runs.files import check_fingerprints, finished_run, sha256, write_json
from mcpref.shards.store import complete_shards, shard_path
from mcpref.vpt.plan import generator, read_shard

if TYPE_CHECKING:
    from mcpref.recognition.evaluate import GuiMasksConfig


def check_labels(cfg: "GuiMasksConfig", frames: dict) -> None:
    """(R): N_S complete label shards, labelling the current frames."""
    n = complete_shards(REPO / cfg.labels_dir / "labels")
    if n != frames["shards"]:
        raise RuntimeError(
            f"{cfg.labels_dir} has {n} label shards, not {frames['shards']}"
        )
    labels_run = json.loads((REPO / cfg.labels_dir / "run.json").read_text())
    if labels_run["frames"] != frames:
        raise RuntimeError(
            f"{cfg.labels_dir} does not label the current {cfg.frames_dir}"
        )


def check_patches(cfg: "GuiMasksConfig", width: int, height: int) -> None:
    """(R): n_P <= the number of P x P patches of a W x H frame."""
    m = (width // cfg.patch) * (height // cfg.patch)
    if cfg.patches_per_class > m:
        n = cfg.patches_per_class
        raise RuntimeError(
            f"patches_per_class {n} exceeds the {m} patches of a {width}x{height} frame"
        )


def labels_sha256(labels_dir: Path, n_s: int) -> str:
    """(R) `labels`: SHA-256 of the label shards' labels.json concatenated in shard order."""
    digest = hashlib.sha256()
    for s in range(n_s):
        digest.update((shard_path(labels_dir / "labels", s) / LABELS).read_bytes())
    return digest.hexdigest()


def checks(cfg: "GuiMasksConfig", config: Path, raw: dict) -> dict:
    """(R): the checks in order, then run.json written; returns its contents."""
    counts = run_counts(finished_run(REPO / cfg.frames_dir, cfg.frames_dir, "frames"))
    frames = {"shards": complete_shards(REPO / cfg.frames_dir / "plan")} | counts
    check_labels(cfg, frames)
    check_patches(cfg, *frame_size(REPO / cfg.frames_dir))
    hashes = {"config": sha256(config), "screens": sha256(REPO / cfg.screens)}
    labels = labels_sha256(REPO / cfg.labels_dir, frames["shards"])
    path = REPO / cfg.out_dir / "run.json"
    check_fingerprints(path, hashes | {"frames": frames, "labels": labels}, cfg.out_dir)
    header = {"config": raw, "sha256": hashes, "frames": frames, "labels": labels}
    write_json(path, header)
    return header


def split_draw(rng: np.random.Generator, probs: Sequence[float]) -> int:
    """Notation: one draw rng.choice(3, p=split) of the split."""
    return int(rng.choice(3, p=probs))


def rebuilt(directory: Path, h: int) -> tuple[dict[str, int], int, dict | None]:
    """(I) continuation: the parts of the episodes of the complete index shards, their
    frames and the generator state of shard h - 1 (None when h = 0)."""
    parts, n, state = {}, 0, None
    for s in range(h):
        shard = read_index(directory, s)
        parts |= {f["episode"]: f["part"] for f in shard["frames"]}
        n, state = n + len(shard["frames"]), shard["rng"]
    return parts, n, state


def index_frames(
    cfg: "GuiMasksConfig", s: int, allowed: set[str], parts: dict, rng
) -> list[dict]:
    """(I): the `ok` frames of plan shard s (their .yuv exists) in order, joined to their
    label records by id; an episode's part drawn at its first appearance; a label not
    `allowed` aborts."""
    frames_dir, probs = REPO / cfg.frames_dir, [float(p) for p in cfg.split]
    labels = read_labels(REPO / cfg.labels_dir / "labels", s)["labels"]
    records, index = {r["id"]: r for r in labels}, []
    for f in read_shard(frames_dir / "plan", s)["frames"]:
        if not yuv_path(frames_dir, f).exists():
            continue
        r, e = records[f["id"]], f["relpath"]
        if r["label"] not in allowed:
            raise RuntimeError(
                f"label {r['label']} of {f['id']} is not a class of the catalogue"
            )
        if e not in parts:
            parts[e] = split_draw(rng, probs)
        g, part = r["gui_inventory"], parts[e]
        index.append(
            {"id": f["id"], "episode": e, "label": r["label"], "g": g, "part": part}
        )
    return index


def write_index_shards(cfg: "GuiMasksConfig", names: list[str], frames: dict) -> None:
    """(I): index shards h, ..., N_S - 1; bar `index`."""
    directory = REPO / cfg.out_dir / INDEX
    h, n_s = complete_shards(directory), frames["shards"]
    parts, initial, state = rebuilt(directory, h)
    rng = generator(cfg.seed, state)
    allowed = {*names, OTHER, INVALID, NO_ANSWER}
    with shard_bar("index", n_s, h, frames["ok"], initial, "frame") as update:
        for s in range(h, n_s):
            shard = index_frames(cfg, s, allowed, parts, rng)
            write_index(directory, s, shard, rng.bit_generator.state)
            update(len(shard))


def part_frames(out: Path, s: int, part: int) -> list[dict]:
    """(S): the frames of F^tr, F^va or F^te (part 0, 1, 2) in index shard s, in order."""
    frames = read_index(out / INDEX, s)["frames"]
    return [f for f in frames if f["part"] == part and labelled(f)]


def folds(relpaths: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """(13): episode e_{pi[r]} gets fold r mod K, pi = permutation(n) of
    Generator(PCG64(seed)); relpaths = e_0 ... e_{n-1}."""
    pi = np.random.Generator(np.random.PCG64(seed)).permutation(len(relpaths))
    return {relpaths[i]: r % k for r, i in enumerate(pi)}


@dataclass(frozen=True)
class State:
    """Behaviour (I): the state; size = (W, H); names = K; parts: episode -> split draw;
    counts[s][i]: labelled frames of part i in index shard s; per_label[i]: of part i
    by label; q = Q; fold_of: training episode -> fold; fold_frames[l] = |F^tr_(l)|;
    built: the built folds; train[k]: (id, fold) of each frame of F^tr_k, k in Q."""

    size: tuple[int, int]
    names: list[str]
    parts: dict[str, int]
    counts: list[list[int]]
    per_label: list[Counter]
    q: list[str]
    fold_of: dict[str, int]
    fold_frames: list[int]
    built: list[int]
    train: dict[str, list[tuple[str, int]]]


def scan(directory: Path, n_s: int, names: list[str]) -> tuple:
    """(I): the index shards read one at a time: parts, counts, per_label, the labelled
    frames of each training episode, (id, episode) of the frames of F^tr_k per k in K."""
    parts, counts, per_label = {}, [], [Counter() for _ in range(3)]
    in_train, train = Counter(), {k: [] for k in names}
    for s in range(n_s):
        frames = read_index(directory, s)["frames"]
        parts |= {f["episode"]: f["part"] for f in frames}
        counts.append([0, 0, 0])
        for f in filter(labelled, frames):
            counts[s][f["part"]] += 1
            per_label[f["part"]][f["label"]] += 1
            if f["part"] == 0:
                in_train[f["episode"]] += 1
                train.get(f["label"], []).append((f["id"], f["episode"]))
    return parts, counts, per_label, in_train, train


def read_state(cfg: "GuiMasksConfig", names: list[str], n_s: int) -> State:
    """(I): the state; an empty F^va aborts, before any template."""
    scanned = scan(REPO / cfg.out_dir / INDEX, n_s, names)
    parts, counts, per_label, in_train, train = scanned
    if not sum(c[1] for c in counts):
        raise RuntimeError("the validation part is empty")
    q = [k for k in names if len({e for _, e in train[k]}) >= cfg.min_episodes]
    fold_of = folds([e for e, p in parts.items() if p == 0], cfg.folds, cfg.fold_seed)
    fold_frames = [0] * cfg.folds
    for e, n in in_train.items():
        fold_frames[fold_of[e]] += n
    built = [l for l in range(cfg.folds) if fold_frames[l]]
    train_q = {k: [(i, fold_of[e]) for i, e in train[k]] for k in q}
    size = frame_size(REPO / cfg.frames_dir)
    return State(
        size, names, parts, counts, per_label, q, fold_of, fold_frames, built, train_q
    )
