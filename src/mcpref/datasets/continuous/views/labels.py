"""Attach additional grid-only labels while preserving the dataset world split."""

import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.timeline import identity
from mcpref.infrastructure.runs.files import clip_fields, finished_run
from mcpref.infrastructure.runs.files import fingerprint as dataset_fingerprint
from mcpref.infrastructure.shards.store import complete_shards, shard_path


def extra_labels(cfg, root: Path):
    if cfg.pair_clips_dir is None:
        return [], None
    pool = REPO / cfg.pair_clips_dir
    labels = REPO / cfg.pair_labels_dir
    pool_run = finished_run(pool, cfg.pair_clips_dir, "complete")
    label_run = finished_run(labels, cfg.pair_labels_dir, "complete")
    dataset = str(root.relative_to(REPO))
    if pool_run.get("shared_dataset") != dataset:
        raise ValueError("Additional preferences belong to a different shared dataset")
    if pool_run["shared_fingerprint"]["dataset"] != digest(root / "run.json"):
        raise ValueError("Additional preferences belong to a different dataset version")
    expected = dataset_fingerprint(
        cfg.pair_clips_dir, "records.json", clip_fields(pool_run), lambda: None
    )
    if label_run["pair_clips"] != expected:
        raise ValueError("Additional labels refer to a different preference pool")
    parts = json.loads((root / "split.json").read_text())["fragment_parts"]
    fingerprint = hashlib.sha256(
        json.dumps([pool_run, label_run], sort_keys=True).encode()
    )
    pairs, seen = [], {}
    total = complete_shards(labels / "shards")
    for k in tqdm(
        range(total),
        total=total,
        desc="Additional grid labels",
        unit="shard",
        mininterval=10,
    ):
        payload = (shard_path(labels / "shards", k) / "pairs.json").read_bytes()
        fingerprint.update(payload)
        for pair in json.loads(payload):
            endpoints = []
            for side in ("a", "b"):
                endpoint = pair[side]
                relative = (
                    shard_path(Path(cfg.pair_clips_dir) / "shards", endpoint["shard"])
                    / "clips"
                    / endpoint["id"]
                )
                key = str(relative)
                if key not in seen:
                    raw = (REPO / relative / "clip.json").read_bytes()
                    fingerprint.update(raw)
                    info = json.loads(raw)
                    if info.get("dataset") != dataset or not info["classical_valid"]:
                        raise ValueError(
                            "Additional endpoint is not an eligible shared history"
                        )
                    if info["id"] != identity(info["observation_ids"]):
                        raise ValueError("Additional endpoint identity differs")
                    seen[key] = (key, info, parts[info["fragment_id"]])
                endpoints.append(seen[key])
            a, b = endpoints
            if a[2] != b[2]:
                raise ValueError("New annotation crosses the frozen world split")
            if a[1]["observation_ids"] == b[1]["observation_ids"]:
                raise ValueError("New annotation compares an identical history")
            pairs.append(({"a": a[0], "b": b[0], "labels": pair["labels"]}, a[2]))
    return (list(seen.values()), pairs), fingerprint.hexdigest()


def attach(items, extra):
    if not extra:
        return
    known = set()
    for rows in items:
        for item in rows:
            if "clip" in item:
                info = json.loads((REPO / item["clip"] / "clip.json").read_text())
                known.add(info["id"])
    clips, pairs = extra
    for path, info, part in clips:
        if info["id"] not in known:
            items[("train", "validation").index(part)].append({"clip": path})
            known.add(info["id"])
    for pair, part in pairs:
        items[("train", "validation").index(part)].append(pair)
