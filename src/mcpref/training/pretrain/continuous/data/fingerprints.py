"""Bind indexes to completed shared data, masks and additional annotations."""

import hashlib
import json

from tqdm import tqdm

from mcpref.config import REPO, config_sha256, read_yaml
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.recovery import verify_files
from mcpref.infrastructure.runs.files import clip_fields, finished_run
from mcpref.infrastructure.runs.files import fingerprint as dataset_fingerprint
from mcpref.infrastructure.shards.store import complete_shards, shard_path

FORMAT = "mcpref-continuous-index-v1"


def additional_runs(cfg, root):
    if cfg.pair_clips_dir is None:
        return None
    pool = finished_run(REPO / cfg.pair_clips_dir, cfg.pair_clips_dir, "complete")
    labels = finished_run(REPO / cfg.pair_labels_dir, cfg.pair_labels_dir, "complete")
    if pool.get("shared_dataset") != cfg.dataset:
        raise ValueError("Additional preferences belong to a different dataset")
    if pool["shared_fingerprint"]["dataset"] != digest(root / "run.json"):
        raise ValueError("Additional preferences belong to a different dataset version")
    expected = dataset_fingerprint(
        cfg.pair_clips_dir, "records.json", clip_fields(pool), lambda: None
    )
    if labels["pair_clips"] != expected:
        raise ValueError("Additional labels belong to a different preference pool")
    return pool, labels


def sources(cfg, path, maths, masks):
    root = REPO / cfg.dataset
    run = finished_run(root, cfg.dataset, "complete")
    if run.get("format") != "mcpref-pretraining-streams-v1":
        raise ValueError("Continuous pretraining requires a shared stream dataset")
    value = hashlib.sha256()
    for name in ("run.json", "split.json"):
        value.update((root / name).read_bytes())
    count = complete_shards(root / "fragments")
    for k in tqdm(
        range(count), total=count, desc="Continuous source identity", unit="fragment"
    ):
        directory = shard_path(root / "fragments", k)
        verify_files(directory)
        value.update((directory / "state.json").read_bytes())
    for k in range(complete_shards(root / "shards")):
        value.update((shard_path(root / "shards", k) / "state.json").read_bytes())
    for k in range(complete_shards(root / "preference-map")):
        value.update(
            (shard_path(root / "preference-map", k) / "pairs.jsonl").read_bytes()
        )
    additional = additional_runs(cfg, root)
    if additional is not None:
        value.update(json.dumps(additional, sort_keys=True).encode())
        directory = REPO / cfg.pair_labels_dir / "shards"
        for k in range(complete_shards(directory)):
            value.update((shard_path(directory, k) / "pairs.json").read_bytes())
        pool = REPO / cfg.pair_clips_dir / "shards"
        for k in range(complete_shards(pool)):
            for clip in sorted((shard_path(pool, k) / "clips").glob("*/clip.json")):
                value.update(clip.read_bytes())
    train = read_yaml(REPO / cfg.training_config)
    mask_value = {
        "names": masks.names,
        "world": masks.world.tolist(),
        "screens": {k: v.tolist() for k, v in masks.screens.items()},
        "pitch_limit": masks.pitch_limit,
    }
    return {
        **maths,
        "index_config": config_sha256(path),
        "training_config": config_sha256(REPO / cfg.training_config),
        "preference_config": config_sha256(REPO / train["preference"]),
        "action_mask_sha256": hashlib.sha256(
            json.dumps(mask_value, sort_keys=True).encode()
        ).hexdigest(),
        "dataset": value.hexdigest(),
    }, run
