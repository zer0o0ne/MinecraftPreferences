"""Uniform kept-scale shifts and unchanged original preference edges."""

import bisect
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mcpref.config import REPO
from mcpref.datasets.continuous.store import jsonl
from mcpref.datasets.continuous.timeline import identity
from mcpref.datasets.continuous.views.index import write_view
from mcpref.datasets.graphs.components import components
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import complete_shards, shard_path, write_shard


def map_clip(
    clip: dict, timeline: dict, source_offset: int, delta: int, maximum: int
) -> dict:
    original = clip["clip"]
    raw = [source_offset + t for t in original["states"]]
    positions = [bisect.bisect_left(timeline["kappa"], t) for t in raw]
    base = {
        "endpoint": clip["path"],
        "original_id": original["id"],
        "original_sha256": clip["sha256"],
        "source": original["relpath"],
        "original_states": original["states"],
        "kind": clip["kind"],
        "fragment_id": timeline["id"],
        "world_group": timeline["world_group"],
    }
    if any(
        p == len(timeline["kappa"]) or timeline["kappa"][p] != t
        for p, t in zip(positions, raw)
    ):
        return base | {"status": "excluded_original_state"}
    if positions != [positions[0] + i * delta for i in range(len(positions))]:
        return base | {"status": "nonuniform_kept_history"}
    shifts = sorted(range(-maximum, maximum + 1), key=lambda s: (abs(s), s))
    fitting = False
    for shift in shifts:
        moved = [p + shift for p in positions]
        if moved[0] % delta or moved[0] < 0 or moved[-1] >= len(timeline["kappa"]):
            continue
        fitting = True
        if all(p in timeline["observations"] for p in moved):
            ids = [timeline["observations"][p] for p in moved]
            return base | {
                "status": "mapped",
                "shift": shift,
                "original_kept_ordinals": positions,
                "observations": ids,
                "canonical_history_id": identity(ids),
            }
    return base | {
        "status": "excluded_grid_state" if fitting else "insufficient_fragment_extent"
    }


def unavailable(clip: dict, reason: str) -> dict:
    original = clip["clip"]
    return {
        "endpoint": clip["path"],
        "original_id": original["id"],
        "original_sha256": clip["sha256"],
        "source": original["relpath"],
        "original_states": original["states"],
        "kind": clip["kind"],
        "status": "unavailable_source",
        "reason": reason,
    }


def finish(cfg, out, legacy: dict, label_hash: str, split_cfg) -> dict:
    mappings, fragments = {}, []
    for k in range(complete_shards(out / "fragments")):
        directory = shard_path(out / "fragments", k)
        summary = json.loads((directory / "summary.json").read_text())
        fragments.extend(summary["streams"])
        for line in (directory / "mappings.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["endpoint"] in mappings:
                raise ValueError("Preference endpoint mapped more than once")
            mappings[record["endpoint"]] = record
    missing = set(legacy) - set(mappings)
    if missing:
        raise ValueError(
            f"Preference endpoints missing from acquisition: {len(missing)}"
        )
    report = {
        "endpoints": len(mappings),
        "endpoint_status": dict(Counter(m["status"] for m in mappings.values())),
        "shifts": dict(
            Counter(
                str(m["shift"]) for m in mappings.values() if m["status"] == "mapped"
            )
        ),
        "excluded_endpoints": [m for m in mappings.values() if m["status"] != "mapped"],
        "aliases": {},
        "pairs": 0,
        "retained_pairs": 0,
        "excluded_pairs": [],
    }
    aliases = {}
    for m in mappings.values():
        if m["status"] == "mapped":
            aliases.setdefault(m["canonical_history_id"], []).append(m["endpoint"])
    report["aliases"] = {key: ids for key, ids in aliases.items() if len(ids) > 1}
    edges, pairs = write_edges(cfg, out, mappings, label_hash, report)
    groups = list(dict.fromkeys(f["world_group"] for f in fragments))
    component = components(groups, edges)
    rng = np.random.Generator(np.random.PCG64([split_cfg.seed, 2]))
    validation = float(split_cfg.validation)
    drawn = [
        int(rng.choice(2, p=[1 - validation, validation], replace=True, size=None))
        for _ in set(component)
    ]
    parts = {g: ("train", "validation")[drawn[c]] for g, c in zip(groups, component)}
    write_json(
        out / "split.json",
        {
            "seed": split_cfg.seed,
            "validation": split_cfg.validation,
            "groups": parts,
            "fragment_parts": {f["id"]: parts[f["world_group"]] for f in fragments},
        },
    )
    write_json(out / "mapping-report.json", report)
    classical = write_view(cfg, out, mappings, pairs, parts, split_cfg)
    return {
        "endpoints": len(mappings),
        "pairs": report["pairs"],
        "retained_pairs": report["retained_pairs"],
        "excluded_pairs": len(report["excluded_pairs"]),
        "classical_clips": sum(p["clips"] for p in classical.values()),
        "classical_pairs": sum(p["pairs"] for p in classical.values()),
    }


def write_edges(cfg, out, mappings: dict, label_hash: str, report: dict) -> list:
    source, target, edges = (
        REPO / cfg.pair_labels_dir / "shards",
        out / "preference-map",
        [],
    )
    retained = []
    for k in range(complete_shards(source)):
        kept = []
        pairs = json.loads((shard_path(source, k) / "pairs.json").read_text())
        for i, pair in enumerate(pairs):
            endpoints = [
                str(
                    shard_path(Path(cfg.pair_clips_dir) / "shards", pair[s]["shard"])
                    / "clips"
                    / pair[s]["id"]
                )
                for s in ("a", "b")
            ]
            a, b = [mappings[e] for e in endpoints]
            record = {
                "id": f"{label_hash}:{k}:{i}",
                "original": pair,
                "a": endpoints[0],
                "b": endpoints[1],
                "labels": pair["labels"],
            }
            reason = (
                "unmapped_endpoint"
                if any(m["status"] != "mapped" for m in (a, b))
                else None
            )
            if reason is None and a["observations"] == b["observations"]:
                reason = "collapsed_pair"
            report["pairs"] += 1
            if reason:
                report["excluded_pairs"].append(record | {"reason": reason})
                continue
            kept.append(
                record
                | {
                    "a_history": a["canonical_history_id"],
                    "b_history": b["canonical_history_id"],
                }
            )
            edges.append((a["world_group"], b["world_group"]))
            report["retained_pairs"] += 1
        payload = jsonl(kept)
        retained.extend(kept)
        final = shard_path(target, k)
        if final.exists():
            if (final / "pairs.jsonl").read_bytes() != payload:
                raise ValueError("Resumed preference mapping differs")
        else:
            with write_shard(target, k) as staged:
                (staged / "pairs.jsonl").write_bytes(payload)
    return edges, retained
