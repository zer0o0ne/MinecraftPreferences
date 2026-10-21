"""Fake inputs of specs/pretrain.md written as their formats say: $C$, $Q_c$ (perception-data's
format, with clip files of $L$ state frames of $W \\times H$), $Q_\\ell$ (its contract) and the
index $X$ of train-index (`run.json`, `train/` and `validation/` shards of items); the
pretraining config. The clips also as `Clip`s built in memory for the oracles."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from mcpref.pretrain.clip import Clip

from .index_oracle import clips_fingerprint, items_bytes, labels_fingerprint, name, sha
from .index_oracle import shard_files as files_of
from .index_world import clips_run, end, pair, record, write_clips, write_json
from .index_world import write_labels as write_pairs
from .loop_modules import ACTIONS, FRAME, H, L, W

INDEX = {"clips_dir": "w/clips", "pair_clips_dir": "w/pair_clips",
         "pair_labels_dir": "w/pair_labels", "out_dir": "w/index", "validation": "0.3",
         "seed": 7, "items_per_shard": 3}  # fmt: skip
SIZES = {"config.width": W, "config.height": H, "config.states": L}
E1, E2, E3, E4 = "6.13/e1.mp4", "6.13/e2.mp4", "6.13/e3.mp4", "6.14/e4.mp4"
E5, E6 = "6.15/e5.mp4", "6.15/e6.mp4"
CLIPS = [  # C: 5 clips (one record with clips null)
    [record(E1, ("u0", "u1")), record(E2), record(E3, ("u2",))],
    [record(E4, ("u3", "u4"))],
]
PAIR_CLIPS = [[record(E1, ("q0",)), record(E5, ("q1", "q2"))], [record(E6, ("q3",))]]
PAIRS = [  # Q_l: q0 and q1 in two pairs each
    [pair(end(0, "q0", E1), end(0, "q1", E5), {"y": 1.0, "c": 1.0}),
     pair(end(0, "q1", E5), end(1, "q3", E6), {"y": 0.0, "c": 1.0})],
    [pair(end(0, "q2", E5), end(0, "q0", E1), {"y": 1.0, "c": 0.0}),
     pair(end(1, "q3", E6), end(0, "q2", E5), {"y": 0.0, "c": 0.0})],
]  # fmt: skip


def clip_path(written: str, shard: int, cid: str) -> str:
    return f"{written}/shards/{name(shard)}/clips/{cid}"


def u(i: int) -> dict:
    """The unlabeled item of clip u<i> of C."""
    return {"clip": clip_path("w/clips", 0 if i < 3 else 1, f"u{i}")}


def p(j: int) -> dict:
    """The labeled item of pair j of Q_l (in shard order)."""
    q = [x for shard in PAIRS for x in shard][j]
    ends = {k: clip_path("w/pair_clips", q[k]["shard"], q[k]["id"]) for k in "ab"}
    return {**ends, "labels": q["labels"]}


@dataclass(frozen=True)
class World:
    """The index: `train` and `validation` shards of items; `pairs`: X has pairs (the Q
    paths are strings) and Q_c, Q_l exist."""

    train: list
    validation: list = field(default_factory=list)
    pairs: bool = True

    @property
    def n_u(self) -> int:
        return sum("clip" in it for s in self.train for it in s)

    @property
    def n_r(self) -> int:
        return sum("clip" not in it for s in self.train for it in s)

    def counts(self) -> dict:
        part = {"train": self.train, "validation": self.validation}
        return {k: {"clips": sum("clip" in it for s in v for it in s),
                    "pairs": sum("clip" not in it for s in v for it in s), "episodes": 3}
                for k, v in part.items()}  # fmt: skip


WORLD = (
    World(  # N_U = 5, N_R = 3, B_X = 3: shards of 3, 3, 2 items; one pair in validation
        train=[[u(0), p(0), u(1)], [p(1), u(2), u(3)], [p(2), u(4)]],
        validation=[[p(3)]],
    )
)
NO_PAIRS = World(train=[[u(0), u(1), u(2)], [u(3), u(4)]], pairs=False)
VALIDATION_PAIRS = World(  # pairs in the index, none in the training part
    train=[[u(0), u(1), u(2)], [u(3), u(4)]], validation=[[p(0), p(1), p(2)], [p(3)]]
)


def content(path: str) -> tuple[bytes, dict, list]:
    """A clip's frames.yuv bytes, clip.json and lines.jsonl values, fixed by its path."""
    rng = np.random.default_rng(list(path.encode()))
    frames = rng.integers(0, 256, size=L * FRAME, dtype=np.uint8).tobytes()
    info = {"id": path.rsplit("/", 1)[1], "actions": rng.integers(0, ACTIONS, L).tolist(),
            "relpath": "é"}  # fmt: skip
    w = float(rng.uniform(0.5, 1.5))
    lines = [{"w": w, "s": "a b\x85c"}] + [{"tick": t} for t in range(L)]
    return frames, info, lines


def clip(path: str) -> Clip:
    """The Clip at `path` as Notation defines it, built from `content`."""
    frames, info, lines = content(path)
    array = np.frombuffer(frames, dtype=np.uint8).reshape(L, FRAME).copy()
    return Clip(frames=array, width=W, height=H, info=info, lines=lines)


def write_clip(directory: Path, path: str) -> None:
    """The three files of a clip (perception-data's format)."""
    frames, info, lines = content(path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "frames.yuv").write_bytes(frames)
    (directory / "clip.json").write_bytes(json.dumps(info, ensure_ascii=False).encode())
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    (directory / "lines.jsonl").write_bytes(text.encode("utf-8"))


def write_dataset(repo: Path, written: str, shards: list, run: dict) -> None:
    """A clips dataset (write_clips of the train-index tests) with real clip files."""
    write_clips(repo / written, shards, run)
    for k, records in enumerate(shards):
        for cid in [i for r in records for i in r["clips"] or ()]:
            path = clip_path(written, k, cid)
            write_clip(repo / path, path)


def records(repo: Path, written: str) -> list[bytes]:
    return files_of(repo / written / "shards", "records.json")


def write_inputs(repo: Path, pairs: bool) -> dict:
    """C, and with pairs Q_c and Q_l; their fingerprints (4) of specs/train-index.md."""
    c_run = clips_run(5, SIZES)
    write_dataset(repo, INDEX["clips_dir"], CLIPS, c_run)
    fp = {"clips": clips_fingerprint(c_run, records(repo, INDEX["clips_dir"])),
          "pair_clips": None, "pair_labels": None}  # fmt: skip
    if pairs:
        q_run = clips_run(4, SIZES | {"config.n_clips": 7}) | {"note": "Q_c"}
        write_dataset(repo, INDEX["pair_clips_dir"], PAIR_CLIPS, q_run)
        fp["pair_clips"] = clips_fingerprint(q_run, records(repo, "w/pair_clips"))
        labels_run = {"complete": True, "pair_clips": fp["pair_clips"]}
        write_pairs(repo / INDEX["pair_labels_dir"], PAIRS, labels_run)
        pairs_bytes = files_of(repo / "w/pair_labels/shards", "pairs.json")
        fp["pair_labels"] = labels_fingerprint(pairs_bytes)
    return fp


def index_run(world: World, fingerprints: dict) -> dict:
    """{X}/run.json as train-index writes it once complete."""
    config = INDEX if world.pairs else INDEX | dict.fromkeys(["pair_clips_dir", "pair_labels_dir"])  # fmt: skip
    return {"config": config, "sha256": {"config": sha(b"index"), **fingerprints},
            "states": L, "width": W, "height": H, "complete": True,
            "counts": world.counts()}  # fmt: skip


def write_world(repo: Path, world: World) -> None:
    """The inputs and the index X = w/index."""
    run = index_run(world, write_inputs(repo, world.pairs))
    write_json(repo / "w/index/run.json", run)
    for part, shards in [("train", world.train), ("validation", world.validation)]:
        for s, items in enumerate(shards):
            path = repo / "w/index" / part / name(s) / "items.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(items_bytes(items))


GROUPS = {  # in another order than the groups' (phi_A, phi_c, dual, pi, idle, pref)
    "pref": {"lr": 0.04, "weight_decay": 0.02},
    "pi": {"lr": 0.03, "weight_decay": 0.01},
    "idle": {"lr": 0.2, "weight_decay": 0.5},
    "dual": {"lr": 0.1, "weight_decay": 0.0},
    "phi_c": {"lr": 0.02, "weight_decay": 0.0},
    "phi_A": {"lr": 0.05, "weight_decay": 0.1},
}
CONFIG = {
    "index_dir": "w/index",
    "out_dir": "w/pretrain",
    "seed": 5,
    "epochs": 2,
    "batch": 2,
    "micro_batch": 1,
    "checkpoint_steps": 3,
    "keep_checkpoints": 2,
    "warmup_steps": 3,
    "w_pi": 0.7,
    "w_P": 1.9,
    "grad_clip": 1.0,
    "adam_eps": 1.0e-4,
    "lr_floor": 0.2,
    "betas": [0.8, 0.95],
    "groups": GROUPS,
    "device": "cpu",
}
NO_PREF = CONFIG | {"groups": {k: v for k, v in GROUPS.items() if k != "pref"}}


def config_text(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False)
