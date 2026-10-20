"""Fake inputs of specs/train-index.md written as its Sources and Consumes say: $C$ and
$Q_c$ in perception-data's format, $Q_\\ell$ in its contract, each with a staged shard the
block must ignore; a hand-built world and random ones; edits of their JSON files."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .index_oracle import HASHED, json_bytes, name, sha

CONFIG = {  # keys in another order than Notation's: run.json keeps the file's order
    "out_dir": "w/index",
    "clips_dir": "w/clips",
    "pair_clips_dir": "w/pair_clips",
    "pair_labels_dir": "w/pair_labels",
    "validation": "0.3",
    "seed": 7,
    "items_per_shard": 3,
}
NO_PAIRS = {**CONFIG, "pair_clips_dir": None, "pair_labels_dir": None}


def record(relpath: str, clips: tuple | None = None) -> dict:
    """A record of perception-data's records.json: valid with its clip ids, else
    `no_clip` with `clips` null."""
    valid = clips is not None
    return {
        "subset": "6.13",
        "relpath": relpath,
        "reason": None if valid else "no_clip",
        "options": {"autoJump": False},
        "starts_session": False,
        "overlay_start": False,
        "valid_starts": 4,
        "clips": list(clips) if valid else None,
    }


def end(shard: int, clip: str, episode: str) -> dict:
    """An element of a pair (Consumes): a clip of Q_c and its episode."""
    return {"shard": shard, "id": clip, "episode": episode}


def pair(a: dict, b: dict, labels: dict) -> dict:
    return {"a": a, "b": b, "labels": labels}


def place(run: dict, dotted: str) -> tuple[dict, str]:
    """The mapping holding a dotted name of a run.json, and its last key."""
    *path, key = dotted.split(".")
    for k in path:
        run = run[k]
    return run, key


def setting(dotted: str, value):
    """An edit of a run.json: the dotted name set to value."""

    def change(run: dict) -> None:
        holder, key = place(run, dotted)
        holder[key] = value

    return change


def drop(dotted: str):
    """An edit of a run.json: the dotted name removed."""

    def change(run: dict) -> None:
        holder, key = place(run, dotted)
        del holder[key]

    return change


def clips_run(n_clips: int, changes: dict | None = None) -> dict:
    """perception-data's run.json, complete; changes: dotted name -> value."""
    run = {
        "config": {
            "index_urls": {"6.13": "http://x/6.13.json"},
            "width": 640,
            "height": 360,
            "states": 16,
            "per_episode": 10,
            "n_clips": 100,
            "seed": 0,
            "workers": 4,
            "classifier_batch": 64,
            "out_dir": "data/perception",
        },
        "sha256": {k: sha(k.encode()) for k in HASHED},
        "classifier": "d:0123abcd",
        "episodes": {"null": 3, "no_clip": 1},
        "clips": n_clips,
    }
    for dotted, value in (changes or {}).items():
        setting(dotted, value)(run)
    return run


@dataclass(frozen=True)
class World:
    """The complete shards of C (records), of Q_c (records) and of Q_l (pairs)."""

    clips: list
    pair_clips: list | None = None
    pairs: list | None = None


A, B, C, X = "6.13/a.mp4", "6.14/b.mp4", "6.14/c.mp4", "6.14/x.mp4"
D, E, F, Z = "6.15/d.mp4", "6.15/e.mp4", "6.15/f.mp4", "6.15/z.mp4"
LABELS = [{"winner": "a", "why": "été"}, {"z": 0.1, "a": [1, {"b": None}]}, {}]
HAND = World(  # episodes a, b, c (U), d, e (R); groups {a}, {b, d, e}, {c}
    clips=[
        [record(A, ("a_10", "a_50")), record(X), record(B, ("b_0",))],
        [record(C, ("c_7",))],
    ],
    pair_clips=[
        [record(B, ("b_0", "b_90")), record(D, ("d_3",))],
        [record(E, ("e_5",)), record(F, ("f_8",))],  # f_8 is in no pair
    ],
    pairs=[
        [pair(end(0, "b_0", B), end(0, "d_3", D), LABELS[0]),
         pair(end(0, "d_3", D), end(1, "e_5", E), LABELS[1])],  # d_3 in two pairs
        [pair(end(0, "b_90", B), end(0, "b_0", B), LABELS[2])],
    ],
)  # fmt: skip
STAGED_CLIPS = [record(Z, ("z_1",))]  # a staged shard of C or Q_c, never read
STAGED_PAIRS = [pair(end(0, "b_0", B), end(0, "z_1", Z), {})]


def random_world(seed: int, n_u: int = 24, n_q: int = 16, n_pairs: int = 12) -> World:
    """Unlabeled episodes with 0-3 clips (0: `clips` null), pair clips on episodes of U
    and on new ones (ids also used in C), random pairs over them, 4 per shard."""
    rng = random.Random(seed)
    u = [record(f"6.13/u{i}.mp4", tuple(f"u{i}_{t}" for t in range(n)) or None)
         for i in range(n_u) for n in [rng.randrange(4)]]  # fmt: skip
    eps = [f"6.13/u{i}.mp4" for i in rng.sample(range(n_u), n_q // 2)]
    eps += [f"6.14/q{i}.mp4" for i in range(n_q - n_q // 2)]
    q = [record(e, tuple(f"{e[5:-4]}_{t}" for t in range(1 + rng.randrange(2))))
         for e in eps]  # fmt: skip
    q_shards = [q[k : k + 5] for k in range(0, len(q), 5)]
    ends = [end(k, c, r["relpath"]) for k, s in enumerate(q_shards) for r in s
            for c in r["clips"]]  # fmt: skip
    pairs = [pair(rng.choice(ends), rng.choice(ends), {"p": rng.random(), "i": i})
             for i in range(n_pairs)]  # fmt: skip
    return World(
        clips=[u[k : k + 5] for k in range(0, len(u), 5)],
        pair_clips=q_shards,
        pairs=[pairs[k : k + 4] for k in range(0, len(pairs), 4)],
    )


def n_clips(shards: list) -> int:
    return sum(len(r["clips"] or ()) for s in shards for r in s)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def read(path: Path):
    return json.loads(path.read_text())


def edit(path: Path, change) -> None:
    """Rewrites the JSON file at path with change(value) applied."""
    value = read(path)
    change(value)
    write_json(path, value)


def write_clips(directory: Path, shards: list, run: dict) -> None:
    """A clips dataset in perception-data's format: run.json, its complete shards
    (records.json, state.json, clips/<id>/) and a staged shard after them."""
    write_json(directory / "run.json", run)
    staged = [(name(len(shards)) + ".part", STAGED_CLIPS)]
    for k, records in [*((name(k), s) for k, s in enumerate(shards)), *staged]:
        shard = directory / "shards" / k
        write_json(shard / "records.json", records)
        write_json(shard / "state.json", {"generator": {}, "clips": 0})
        for i in [i for r in records for i in r["clips"] or ()]:
            write_json(shard / "clips" / i / "clip.json", {"id": i})
            (shard / "clips" / i / "frames.yuv").write_bytes(b"")
            (shard / "clips" / i / "lines.jsonl").write_bytes(b"")


def write_labels(directory: Path, shards: list, run: dict) -> None:
    """Q_l in its contract (Consumes): run.json, pairs.json of its complete shards and of
    a staged shard after them."""
    write_json(directory / "run.json", run)
    for k, pairs in enumerate(shards):
        write_json(directory / "shards" / name(k) / "pairs.json", pairs)
    staged = directory / "shards" / (name(len(shards)) + ".part")
    write_json(staged / "pairs.json", STAGED_PAIRS)
