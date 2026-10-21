"""Fake $Q_c$ of specs/vlm-labels.md, written as specs/preference-pool.md (Produces, Datasets)
says: run.json in perception-data's format with the counts, complete shards (records.json,
state.json, comparisons.json, clips/<id>/ with frames.yuv, lines.jsonl, clip.json) and a
staged shard after them; the fake VLM's answers with their labels by (2), by hand."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pretrain.index_oracle import json_bytes, name
from pretrain.index_world import clips_run, record

W, H, L = 4, 2, 6  # config.width, config.height, config.states of {Q_c}/run.json
E = {k: f"data/9.0/p{k}-0123456789ab-20220101-1{k}0000.mp4" for k in range(1, 9)}
DECOY = [{"type": "minecraft:diamond", "quantity": 9}]  # the lines before the last


def stack(*entries: tuple) -> list:
    return [{"type": t, "quantity": q} for t, q in entries]


# clip id -> (episode, inventory of its last line, iota of it with TINY by hand)
CLIPS = {
    "9.0_p1_40": (E[1], stack(("minecraft:oak_log", 3), ("minecraft:dirt", 5),
                              ("minecraft:oak_log", 2)),
                  "Inv: 5xminecraft:dirt, 5xminecraft:oak_log\n"),
    "9.0_p1_200": (E[1], [], "Inv: none\n"),
    "9.0_p2_16": (E[2], stack(("minecraft:stone_pickaxe", 1)),
                  "Inv: 1xminecraft:stone_pickaxe\n"),
    "9.0_p4_8": (E[4], stack(("b", 1), ("B", 2), ("a_b", 3), ("ab", 4), ("a", 5), ("b", 6)),
                 "Inv: 2xB, 5xa, 3xa_b, 4xab, 7xb\n"),
    "9.0_p4_96": (E[4], stack(("minecraft:bread", 64), ("minecraft:bread", 64)),
                  "Inv: 128xminecraft:bread\n"),
    "9.0_p5_0": (E[5], stack(("minecraft:torch", 16)), "Inv: 16xminecraft:torch\n"),
    "9.0_p6_32": (E[6], [], "Inv: none\n"),
    "9.0_p7_0": (E[7], [], "Inv: none\n"),
    "9.0_p7_9": (E[7], [], "Inv: none\n"),
}  # fmt: skip


@dataclass(frozen=True)
class Shard:
    records: list  # records.json
    comparisons: list  # (id of a, id of b), in the order of comparisons.json


HAND = [  # 3 + 0 + 1 + 1 comparisons; shard 1: a session with n < 2
    Shard(
        [record(E[1], ("9.0_p1_40", "9.0_p1_200")), record(E[2], ("9.0_p2_16",))],
        [("9.0_p1_40", "9.0_p2_16"), ("9.0_p1_200", "9.0_p1_40"),
         ("9.0_p2_16", "9.0_p1_200")],
    ),
    Shard([record(E[3], ()), record(E[8])], []),
    Shard([record(E[4], ("9.0_p4_8", "9.0_p4_96"))], [("9.0_p4_96", "9.0_p4_8")]),
    Shard([record(E[5], ("9.0_p5_0",)), record(E[6], ("9.0_p6_32",))],
          [("9.0_p5_0", "9.0_p6_32")]),
]  # fmt: skip
STAGED = Shard([record(E[7], ("9.0_p7_0", "9.0_p7_9"))], [("9.0_p7_0", "9.0_p7_9")])
ANSWERS = [  # by position: the fake VLM's answer, its label by (2) with TINY
    ("Answer: WORSE", "WORSE"),
    ("A has less.\n**Answer: WORSE**", "WORSE"),
    ("B has less.\nAnswer: BETTER.", "BETTER"),
    (None, "no_answer"),
    ("Answer: better", "invalid"),
    ("`Answer: INCOMPARABLE`\n", "INCOMPARABLE"),
    ("Answer:BETTER", "invalid"),
    ("```\nAnswer: INCOMPARABLE\n```", "INCOMPARABLE"),
    ('"Answer: BETTER"', "BETTER"),
    ("Answer: BETTER\nBecause of the pickaxe.", "invalid"),
    ("'Answer: WORSE'", "WORSE"),
    ("Answer: WORSE\n\n   \n", "WORSE"),
    ("Answer: SAME", "SAME"),
    (None, "no_answer"),
    ("Answer: BETTER", "BETTER"),
    ("Answer: INCOMPARABLE", "INCOMPARABLE"),
    ("Answer: WORSE", "WORSE"),
    ("Answer: BETTER", "BETTER"),
    ("**Answer: WORSE.**", "WORSE"),
    ("Answer: BETTER", "BETTER"),
]
# counts.json `labels` of HAND's shards with TINY, and their sums, by hand: tokens in
# the order of TINY_ANSWERS, then invalid, no_answer
HAND_LABELS = [
    {"BETTER": 2, "WORSE": 4, "INCOMPARABLE": 2, "invalid": 3, "no_answer": 1},
    {},
    {"BETTER": 1, "SAME": 1, "INCOMPARABLE": 1, "no_answer": 1},
    {"BETTER": 2, "WORSE": 2},
]
HAND_TOTAL = {"BETTER": 5, "WORSE": 6, "SAME": 1, "INCOMPARABLE": 3, "invalid": 3}
HAND_TOTAL["no_answer"] = 2
HAND_ENDS = [12, 12, 16, 20]  # `end`: 2 s = 4 requests per comparison


def code(clip: str, j: int) -> int:
    """Frame j of a clip: a distinct byte."""
    return 1 + list(CLIPS).index(clip) * L + j


def bits(c: int) -> np.ndarray:
    """(H, W): pixel (r, x) lit iff bit r W + x of c is set."""
    return np.array([[c >> (r * W + x) & 1 for x in range(W)] for r in range(H)])


def frame_yuv(c: int) -> bytes:
    """Raw yuv420p: Y 235 where lit, else 16; U, V (H/2 x W/2) 128."""
    return (16 + 219 * bits(c)).astype(np.uint8).tobytes() + bytes([128]) * (W * H // 2)


def frame_rgb(c: int) -> np.ndarray:
    """vpt-gui-frames (5) by hand: U = V = 128, so R = G = B = 255/219 (Y - 16)."""
    return np.repeat(bits(c)[..., None] * 255, 3, axis=2).astype(np.uint8)


def ref(clip: str) -> dict:
    """A clip reference of comparisons.json."""
    return {"id": clip, "episode": CLIPS[clip][0]}


def pool_run(shards: list, **changes) -> dict:
    """{Q_c}/run.json, complete: the clip-defining fields and the counts of (8')."""
    n = sum(len(r["clips"] or ()) for s in shards for r in s.records)
    run = clips_run(n, {"config.width": W, "config.height": H, "config.states": L})
    run |= {
        "sessions": len(shards),
        "comparisons": sum(len(s.comparisons) for s in shards),
    }
    for key, value in changes.items():
        run[key] = value
    return run


def write_clip(directory: Path, clip: str) -> None:
    episode, inventory, _ = CLIPS[clip]
    directory.mkdir(parents=True)
    (directory / "frames.yuv").write_bytes(
        b"".join(frame_yuv(code(clip, j)) for j in range(L))
    )
    lines = [{"tick": t, "inventory": DECOY, "isGuiOpen": False} for t in range(3)]
    lines.append({"tick": 3, "inventory": inventory, "isGuiOpen": False})
    (directory / "lines.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (directory / "clip.json").write_text(json.dumps({"id": clip, "relpath": episode}))


def write_shard(directory: Path, shard: Shard) -> None:
    directory.mkdir(parents=True)
    (directory / "records.json").write_bytes(json_bytes(shard.records))
    cmp = [{"a": ref(a), "b": ref(b)} for a, b in shard.comparisons]
    (directory / "comparisons.json").write_bytes(json_bytes(cmp))
    state = {"generator": {}, "clips": 0, "session": {"subset": "9xx", "start": "x"}}
    (directory / "state.json").write_bytes(json_bytes(state))
    for clip in [i for r in shard.records for i in r["clips"] or ()]:
        write_clip(directory / "clips" / clip, clip)


def write_pool(directory: Path, shards: list, run: dict) -> None:
    """Q_c: run.json, the complete shards, then a staged shard."""
    directory.mkdir(parents=True)
    (directory / "run.json").write_bytes(json_bytes(run))
    for k, shard in enumerate(shards):
        write_shard(directory / "shards" / name(k), shard)
    write_shard(directory / "shards" / f"{name(len(shards))}.part", STAGED)
