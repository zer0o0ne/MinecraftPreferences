"""Fake inputs of specs/acyclic.md: $V_\\ell$ written as specs/vlm-labels.md says (its
`run.json` and the `answers.json`, `counts.json` of its complete shards, plus a staged
shard the block must ignore); hand worlds and random ones. A world: per shard, its
comparisons (a, b, labels), a and b clip references {"id", "episode"}, labels the 2s labels
in request order (order 0, then order 1)."""

import hashlib
import json
import random
from pathlib import Path

TOKENS = ("MUCH_BETTER", "BETTER", "SLIGHTLY_BETTER", "SLIGHTLY_WORSE", "WORSE",
          "MUCH_WORSE", "INCOMPARABLE")  # fmt: skip
MB, B, SB, SW, W, MW, INC = TOKENS
START = {  # the starting weights (Interface)
    MB: 1.0, B: 0.8, SB: 0.6, SW: 0.4, W: 0.2, MW: 0.0, INC: None,
}  # fmt: skip
DYADIC = {  # float64-exact costs: the exact transcription of (2) applies
    MB: 1, B: 0.75, SB: 0.625, SW: 0.375, W: 0.25, MW: 0, INC: None,
}  # fmt: skip
POOL = hashlib.sha256(b"fake Q_c").hexdigest()


def name(k: int) -> str:
    """specs/shards.md: name(k)."""
    return f"{k:06d}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    """The files of the blocks: json.dumps(value, indent=2) + "\\n"."""
    return (json.dumps(value, indent=2) + "\n").encode()


def ref(clip: str) -> dict:
    """A clip reference of comparisons.json: its id and its episode."""
    return {"id": clip, "episode": f"6.13/{clip.split('_')[0]}.mp4"}


def comparison(a: str, b: str, *labels: str) -> tuple:
    return ref(a), ref(b), list(labels)


def answer(label: str) -> str | None:
    """A VLM answer the label of specs/vlm-labels.md (2) reads as label."""
    if label == "no_answer":
        return None
    return "no idea" if label == "invalid" else f"A looks safer.\nAnswer: {label}"


def entries(shard: list, position: int) -> tuple[list, int]:
    """answers.json of a shard: per comparison its requests with seeds by position (seed 0)."""
    out = []
    for a, b, labels in shard:
        s = len(labels) // 2
        requests = [
            {"order": j // s, "sample": j % s, "seed": position + j,
             "answer": answer(label), "label": label}
            for j, label in enumerate(labels)
        ]  # fmt: skip
        out.append({"a": a, "b": b, "requests": requests})
        position += len(labels)
    return out, position


def label_counts(shards: list, tokens: tuple) -> dict:
    """{label: count} in the order of the tokens, then invalid, no_answer; those occurring."""
    labels = [t for shard in shards for _, _, ls in shard for t in ls]
    return {
        t: labels.count(t) for t in (*tokens, "invalid", "no_answer") if t in labels
    }


def votes_run(shards: list, tokens: tuple = TOKENS, pool: str = POOL) -> dict:
    """{V_l}/run.json of a complete run (specs/vlm-labels.md (I), (L))."""
    s = next((len(ls) // 2 for shard in shards for _, _, ls in shard), 1)
    config = {
        "pool_dir": "data/pair_clips", "vlm": "configs/vlm.yaml", "out_dir": "data/pair_votes",
        "instruction": "Judge.\n", "first": "A:\n", "second": "B:\n",
        "question": "Which?\n{answers}", "answer_line": "{token}: {description}",
        "prefix": "Answer: ", "inventory_line": "Inventory: {items}\n",
        "item_line": "{quantity} {type}", "no_items": "empty",
        "answers": [{"token": t, "description": t.lower()} for t in tokens],
        "frames": 2, "stride": 1, "samples": s, "seed": 0,
    }  # fmt: skip
    hashes = {
        "config": sha(b"c"),
        "vlm": sha(b"v"),
        "pool": pool,
        "comparisons": sha(b"x"),
    }
    counts = {
        "comparisons": sum(map(len, shards)),
        "labels": label_counts(shards, tokens),
    }
    return {"config": config, "sha256": hashes, "complete": True, "counts": counts}


def write_votes(directory: Path, shards: list, run: dict | None) -> None:
    """V_l: run.json (None: none), its complete shards and a staged one after them."""
    if run is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_bytes(json_bytes(run))
    position = 0
    staged = [comparison("ghost_1", "ghost_2", MB, MB)]
    named = [(name(k), shard) for k, shard in enumerate(shards)]
    for shard_name, shard in [*named, (f"{name(len(shards))}.part", staged)]:
        values, end = entries(shard, position)
        counts = {"comparisons": len(shard), "labels": label_counts([shard], TOKENS)}
        d = directory / "shards" / shard_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "answers.json").write_bytes(json_bytes(values))
        (d / "counts.json").write_bytes(json_bytes(counts | {"end": end}))
        position = end


HAND = [  # weights DYADIC, s = 1, tau = 1/2; derivations in tests/preferences/test_acyclic.py
    [
        comparison("q_1", "p_1", W, B),  # p = 1/4: arc p -> q, w = 3/4
        comparison("q_1", "r_1", B, W),  # arc q -> r, w = 3/4
        comparison("r_1", "p_1", B, W),  # arc r -> p, w = 3/4: the 3-cycle
        comparison("p_1", "t_1", INC, INC),  # incomparable
        comparison("t_1", "q_1", "invalid", "no_answer"),  # dropped
        comparison("t_1", "r_1", INC, MB),  # share 1/2 = tau; p = 0: arc r -> t, w = 1
        comparison("q_1", "t_1", SB, SW),  # arc q -> t, w = 5/8
        comparison("t_1", "s_1", SB, SB),  # p = 1/2: arc t -> s, w = 1/2, cost 0
    ],
    [],  # a shard without comparisons
    [comparison("p_2", "q_2", "invalid", "invalid")],  # comparisons, no pair
    [  # the graph "put back" of tests/graphs/test_feedback.py on clips v0..v3
        comparison("v0_3", "v3_3", SB, SB),  # p = 1/2: arc v0 -> v3, cost 0
        comparison("v1_3", "v0_3", SB, SW),  # arc v1 -> v0, w = 5/8
        comparison("v3_3", "v1_3", MB, W),  # arc v3 -> v1, w = 7/8
        comparison("v2_3", "v1_3", SB, SW),  # arc v2 -> v1, w = 5/8
        comparison("v2_3", "v0_3", MW, B),  # p = 1/8: arc v0 -> v2, w = 7/8
    ],
]


def random_labels(rng: random.Random, s: int) -> list[str]:
    """2s labels: mostly categories; some comparisons mostly refused, some mostly invalid."""
    mode = rng.random()
    pool = (
        [INC, INC, INC, MB, "invalid"] if mode < 0.15
        else ["invalid", "no_answer", "no_answer", B] if mode < 0.3
        else TOKENS[:-1]
    )  # fmt: skip
    return [rng.choice(pool) for _ in range(2 * s)]


def random_world(seed: int, n_shards: int = 4, s: int = 2) -> list:
    """Shards of 0-14 comparisons over 3-7 clips, at most one per pair of clips, in random
    orientation, with random_labels."""
    rng = random.Random(seed)
    world = []
    for k in range(n_shards):
        clips = [f"c{i}_{k}" for i in range(rng.randrange(3, 8))]
        pairs = [(x, y) for i, x in enumerate(clips) for y in clips[i + 1 :]]
        chosen = rng.sample(pairs, min(len(pairs), rng.randrange(15)))
        world.append([
            comparison(*(xy if rng.random() < 0.5 else xy[::-1]), *random_labels(rng, s))
            for xy in chosen
        ])  # fmt: skip
    return world
