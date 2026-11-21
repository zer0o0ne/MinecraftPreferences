"""Small serialized shared dataset with gaps, idle spans and original edge aliases."""

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from mcpref import config as common_config
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.recovery import seal
from mcpref.datasets.continuous.timeline import identity
from mcpref.datasets.continuous.views import reader as payload_reader
from mcpref.gui.mask import Masks
from mcpref.infrastructure.runs import files
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import shard_path
from mcpref.training.pretrain.continuous.data import (
    config,
    fingerprints,
    graph,
    index,
    reader,
)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_world(tmp_path, monkeypatch, K=3, history=2):
    for module in (common_config, config, fingerprints, graph, index, reader, files):
        monkeypatch.setattr(module, "REPO", tmp_path)
    names = ("forward", "turn_up", "turn_down", *(f"hotbar_{i}" for i in range(1, 10)))
    masks = Masks(names, np.ones(len(names), bool), {}, 90)
    for module in (index, reader):
        monkeypatch.setattr(module, "load_masks", lambda _: masks)
        monkeypatch.setattr(module, "mask_config", lambda _: None)
    monkeypatch.setattr(
        reader,
        "reader_for",
        lambda root, w, h: payload_reader.ShardReader(root, "unused", 1, w, h),
    )

    def decoded(cfg, path, limit):
        payload = path.read_bytes()
        for i in range(len(payload) // 6):
            yield i, payload[6 * i : 6 * i + 6]

    monkeypatch.setattr(payload_reader, "decoded", decoded)
    root = tmp_path / "dataset"
    write_json(
        root / "run.json",
        {
            "format": "mcpref-pretraining-streams-v1",
            "complete": True,
            "width": 2,
            "height": 2,
        },
    )
    write_json(
        tmp_path / "train.yaml",
        {
            "tbptt_states": K,
            "preference": "preference.yaml",
            "action_mask": "mask.yaml",
        },
    )
    write_json(tmp_path / "preference.yaml", {"history": history})
    cfg = config.IndexConfig("dataset", "index", "train.yaml", None, None, 1)
    path = tmp_path / "index.yaml"
    write_json(path, dataclasses.asdict(cfg))
    positions, summaries, maps = {}, [], [[], [], []]
    shard = 0
    grids = ([0, 1, 2, 3, 5, 6, 7, 8, 9, 10], [0, 1, 2], [0, 1])
    for f, grid in enumerate(grids):
        first, observations, intervals = shard, [], []
        for ordinal, j in enumerate(grid):
            meta = {
                "inventory": [],
                "hotbar": 0,
                "yaw": 0,
                "pitch": -90 if f == 0 and j == 8 else 0,
                "xpos": 100 + j,
                "ypos": 64,
                "zpos": 200,
                "mouse": {"x": 0, "y": 0},
            }
            observations.append(
                {
                    "id": f"f{f}:{j}",
                    "fragment_id": f"f{f}",
                    "world_group": f"w{f}",
                    "grid_index": j,
                    "source": f"source-{ordinal // 3}",
                    "source_row": ordinal % 3,
                    "gui": False,
                    "screen_class": None,
                    "meta": meta,
                }
            )
            if ordinal and grid[ordinal - 1] + 1 == j:
                left = grid[ordinal - 1]
                actions = (
                    []
                    if f == 0 and left == 2
                    else ["turn_up"]
                    if f == 0 and left == 8
                    else ["forward"]
                )
                known = not (f == 0 and left == 5)
                intervals.append(
                    {
                        "grid_index": left,
                        "left": f"f{f}:{left}",
                        "right": f"f{f}:{j}",
                        "actions": actions,
                        "n": len(actions),
                        "known": known,
                        "valid": bool(actions) and known,
                        "reasons": [],
                    }
                )
        for start in range(0, len(observations), 2):
            directory = shard_path(root / "shards", shard)
            batch = observations[start : start + 2]
            ids = {row["id"] for row in batch}
            incoming = [row for row in intervals if row["right"] in ids]
            jsonl(directory / "observations.jsonl", batch)
            jsonl(directory / "intervals.jsonl", incoming)
            raw = b"".join(
                bytes([16 + start + i] * 4 + [128, 128]) for i in range(len(batch))
            )
            (directory / "frames.mkv").write_bytes(raw)
            write_json(
                directory / "state.json",
                {
                    "observations": len(batch),
                    "intervals": len(incoming),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "files": {
                        name: digest(directory / name)
                        for name in (
                            "observations.jsonl",
                            "intervals.jsonl",
                            "frames.mkv",
                        )
                    },
                },
            )
            for i, row in enumerate(batch):
                positions[row["id"]] = [shard, i]
            shard += 1
        summaries.append(
            {
                "streams": [
                    {
                        "id": f"f{f}",
                        "world_group": f"w{f}",
                        "origin": observations[0]["meta"],
                        "observations": len(grid),
                        "first_shard": first,
                        "next_shard": shard,
                    }
                ]
            }
        )
    histories = {
        "a": ["f0:0", "f0:1", "f0:2"],
        "alias_a": ["f0:0", "f0:1", "f0:2"],
        "short_a": ["f0:1", "f0:2"],
        "c": ["f0:6", "f0:7", "f0:8"],
        "b": ["f1:0", "f1:1", "f1:2"],
    }
    for alias, ids in histories.items():
        f = int(ids[0][1])
        info = {
            "id": identity(ids),
            "dataset": "dataset",
            "fragment_id": f"f{f}",
            "observation_ids": ids,
            "frames": [positions[key] for key in ids],
            "classical_valid": True,
        }
        clip = shard_path(root / "fragments", f) / "clips" / info["id"]
        write_json(clip / "clip.json", info)
        maps[f].append(
            {
                "status": "mapped",
                "endpoint": alias,
                "original_id": alias,
                "clip_ref": str(clip.relative_to(tmp_path)),
                "canonical_history_id": info["id"],
            }
        )
    for f, summary in enumerate(summaries):
        directory = shard_path(root / "fragments", f)
        write_json(directory / "summary.json", summary)
        jsonl(directory / "mappings.jsonl", maps[f])
        seal(directory)
    pairs = [("a", "b"), ("b", "a"), ("alias_a", "b"), ("a", "c"), ("a", "short_a")]
    jsonl(
        shard_path(root / "preference-map", 0) / "pairs.jsonl",
        [
            {"id": f"old:{i}", "a": a, "b": b, "labels": {"inc": i % 2, "p": i / 4}}
            for i, (a, b) in enumerate(pairs)
        ],
    )
    write_json(
        root / "split.json",
        {
            "fragment_parts": {"f0": "train", "f1": "train", "f2": "validation"},
            "groups": {"w0": "train", "w1": "train", "w2": "validation"},
        },
    )
    phi_cfg = SimpleNamespace(
        decimals=1,
        meta="{x},{y},{z}",
        inventory_line="{items}",
        item_line="{type}: {quantity}",
        no_items="empty",
        mouse_height=2,
    )
    return SimpleNamespace(
        cfg=cfg,
        path=path,
        root=root,
        out=tmp_path / "index",
        masks=masks,
        histories=histories,
        positions=positions,
        phi_cfg=phi_cfg,
    )
