"""Frozen legacy provenance, labels, sessions and resume fingerprints."""

import dataclasses
import gzip
import hashlib
import json
from collections import defaultdict

from tqdm import tqdm

from mcpref.config import REPO, config_sha256
from mcpref.config_moves import MOVED
from mcpref.datasets.clips.collect import run_header
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.timeline import identity
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import complete_shards, shard_path


def legacy_inputs(cfg, manifest: dict):
    path = REPO / cfg.legacy_provenance
    expected = manifest["selection_basis"]["live_provenance_export"]["sha256"]
    if digest(path) != expected:
        raise ValueError("Legacy clip provenance differs from frozen acquisition plan")
    legacy, by_source = {}, defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for line in tqdm(
            stream, total=None, desc="Legacy provenance", unit="record", mininterval=10
        ):
            row = json.loads(line)
            if row.get("type") != "clip":
                continue
            clip = row["clip"]
            entry = {
                "path": row["path"],
                "sha256": row["sha256"],
                "kind": "preference" if row["dataset"] == "pair_clips" else "ordinary",
                "clip": {k: clip[k] for k in ("id", "subset", "relpath", "states")},
            }
            legacy[row["path"]] = entry
            by_source[clip["relpath"]].append(entry)
    return legacy, by_source


def label_inputs(cfg):
    root = REPO / cfg.pair_labels_dir
    run = json.loads((root / "run.json").read_text())
    if not run.get("complete"):
        raise ValueError("Preference correction is incomplete")
    value = hashlib.sha256()
    n = complete_shards(root / "shards")
    for k in tqdm(
        range(n), total=n, desc="Label fingerprints", unit="shard", mininterval=10
    ):
        value.update((shard_path(root / "shards", k) / "pairs.json").read_bytes())
    return value.hexdigest()


def annotation_worlds(cfg):
    root = REPO / cfg.pair_clips_dir / "sessions"
    n = complete_shards(root)
    worlds, digest = {}, hashlib.sha256()
    for k in tqdm(
        range(n), total=n, desc="Annotation worlds", unit="shard", mininterval=10
    ):
        payload = (shard_path(root, k) / "sessions.json").read_bytes()
        digest.update(payload)
        for session in json.loads(payload)["sessions"]:
            world = identity([session["subset"], session["pieces"][0]])
            for source in session["pieces"]:
                if source in worlds and worlds[source] != world:
                    raise ValueError("Source belongs to multiple annotation worlds")
                worlds[source] = world
    return worlds, digest.hexdigest()


def header(cfg, pipe, classifier_id, labels: str, sessions: str) -> dict:
    source = run_header(pipe, classifier_id)
    plan = REPO / cfg.acquisition_plan / "manifest.json"
    semantic_config = dataclasses.asdict(cfg)
    semantic_config.pop("transfer_config")
    fingerprint = {
        "config": semantic_config,
        "source_config": source,
        "acquisition_plan": digest(plan),
        "legacy_provenance": digest(REPO / cfg.legacy_provenance),
        "labels": labels,
        "sessions": sessions,
        "split_config": config_sha256(REPO / cfg.split_config),
    }
    out = REPO / cfg.out_dir
    path = out / "run.json"
    if path.exists():
        previous = json.loads(path.read_text())["fingerprint"]
        old_split = previous["config"]["split_config"]
        if MOVED.get(old_split, old_split) == MOVED.get(
            cfg.split_config, cfg.split_config
        ):
            semantic_config["split_config"] = old_split
        if previous != fingerprint:
            raise ValueError(
                "Continuous dataset inputs/config changed; use a new output directory"
            )
    result = {
        "format": "mcpref-pretraining-streams-v1",
        "fingerprint": fingerprint,
        "complete": False,
        "width": pipe.cfg.width,
        "height": pipe.cfg.height,
        "delta": pipe.clips.delta,
    }
    write_json(path, result)
    return result
