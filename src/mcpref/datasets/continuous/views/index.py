"""Classical eligibility view and index over the shared observation store."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref.config import REPO
from mcpref.config_moves import pretrain_config_sha256
from mcpref.datasets.continuous.timeline import identity
from mcpref.datasets.continuous.views.labels import attach, extra_labels
from mcpref.infrastructure.runs.files import check_fingerprints, write_json
from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.training.pretrain.classical.data.semantics import INDEX_SEMANTICS


def population(mappings, pairs, parts):
    clips = {
        m["canonical_history_id"]: m
        for m in mappings.values()
        if m["status"] == "mapped" and m["kind"] == "ordinary" and m["classical_valid"]
    }
    retained, excluded = [], []
    for pair in pairs:
        a, b = (mappings[pair[side]] for side in ("a", "b"))
        if not (a["classical_valid"] and b["classical_valid"]):
            excluded.append(
                {
                    "id": pair["id"],
                    "reason": "classical_invalid_endpoint",
                    "a": a["classical_reasons"],
                    "b": b["classical_reasons"],
                }
            )
            continue
        for clip in (a, b):
            clips.setdefault(clip["canonical_history_id"], clip)
        if parts[a["world_group"]] != parts[b["world_group"]]:
            raise ValueError("Preference pair crosses the shared split")
        retained.append(
            (
                {"a": a["clip_ref"], "b": b["clip_ref"], "labels": pair["labels"]},
                parts[a["world_group"]],
            )
        )
    entries = [
        ({"clip": m["clip_ref"]}, parts[m["world_group"]]) for m in clips.values()
    ]
    return entries + retained, excluded


def view_fingerprint(root: Path):
    run = json.loads((root / "run.json").read_text())
    value = hashlib.sha256(identity(run["fingerprint"]).encode())
    lengths = []
    count = complete_shards(root / "classical-view")
    for k in tqdm(
        range(count), total=count, desc="Shared view", unit="shard", mininterval=10
    ):
        path = shard_path(root / "classical-view", k) / "items.json"
        payload = path.read_bytes()
        value.update(payload)
        for item in json.loads(payload)["items"]:
            if "clip" in item:
                clip = (REPO / item["clip"] / "clip.json").read_bytes()
                value.update(clip)
                lengths.append(len(json.loads(clip)["states"]))
    value.update((root / "split.json").read_bytes())
    return value.hexdigest(), min(lengths) if lengths else 0


def write_view(cfg, out, mappings, pairs, parts, split_cfg):

    entries, excluded = population(mappings, pairs, parts)
    shuffled = [
        [item for item, part in entries if part == name]
        for name in ("train", "validation")
    ]
    rng = np.random.Generator(np.random.PCG64([split_cfg.seed, 2]))
    shuffled = [[items[i] for i in rng.permutation(len(items))] for items in shuffled]
    counts = {
        name: {
            "clips": sum("clip" in item for item in items),
            "pairs": sum("clip" not in item for item in items),
            "episodes": list(parts.values()).count(name),
        }
        for name, items in zip(("train", "validation"), shuffled)
    }
    # The same sharded payload is consumed by index generation and integrity checks.
    from mcpref.infrastructure.runs.files import json_bytes
    from mcpref.infrastructure.shards.store import write_shard

    all_items = [item | {"part": part} for item, part in entries]
    size = split_cfg.items_per_shard
    for k, start in enumerate(range(0, len(all_items), size)):
        payload = json_bytes({"items": all_items[start : start + size]})
        final = shard_path(out / "classical-view", k)
        if final.exists():
            if (final / "items.json").read_bytes() != payload:
                raise ValueError("Resumed classical view differs")
        else:
            with write_shard(out / "classical-view", k) as staged:
                (staged / "items.json").write_bytes(payload)
    write_json(
        out / "classical-report.json",
        {
            "counts": counts,
            "excluded_pairs": excluded,
            "excluded_clips": [
                m
                for m in mappings.values()
                if m["status"] == "mapped" and not m["classical_valid"]
            ],
        },
    )
    return counts


def build_index(cfg, path, root: Path):
    from mcpref.training.pretrain.classical.data.index import write_parts

    run = json.loads((root / "run.json").read_text())
    if not run.get("complete"):
        raise ValueError("Shared dataset is incomplete")
    split = json.loads((root / "split.json").read_text())
    if cfg.seed != split["seed"] or cfg.validation != split["validation"]:
        raise ValueError("Index must use the shared dataset split settings")
    fingerprint, states = view_fingerprint(root)
    extra, extra_fingerprint = extra_labels(cfg, root)
    sha = {
        "config": pretrain_config_sha256(path),
        "clip_metadata": fingerprint,
        "shared_dataset": fingerprint,
        "additional_labels": extra_fingerprint,
    }
    out = REPO / cfg.out_dir
    check_fingerprints(out / "run.json", sha, cfg.out_dir)
    items = [[], []]
    for k in range(complete_shards(root / "classical-view")):
        rows = json.loads(
            (shard_path(root / "classical-view", k) / "items.json").read_text()
        )["items"]
        for row in rows:
            item = dict(row)
            part = item.pop("part")
            items[("train", "validation").index(part)].append(item)
    attach(items, extra)
    if extra and extra[0]:
        states = min(states, *(len(info["states"]) for _, info, _ in extra[0]))
    rng = np.random.Generator(np.random.PCG64([cfg.seed, 2]))
    items = [[rows[i] for i in rng.permutation(len(rows))] for rows in items]
    import dataclasses

    head = {
        "semantics": INDEX_SEMANTICS,
        "config": dataclasses.asdict(cfg),
        "sha256": sha,
        "shared_dataset": str(root.relative_to(REPO)),
        "states": states,
        "width": run["width"],
        "height": run["height"],
    }
    write_json(out / "run.json", head)
    write_parts(out, items, cfg.items_per_shard)
    counts = {
        part: {
            "clips": sum("clip" in row for row in rows),
            "pairs": sum("clip" not in row for row in rows),
            "episodes": list(split["groups"].values()).count(part),
        }
        for part, rows in zip(("train", "validation"), items)
    }
    write_json(out / "run.json", head | {"complete": True, "counts": counts})
