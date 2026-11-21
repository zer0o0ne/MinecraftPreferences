"""Disk-backed endpoint aliases and original oriented edges; no label deduplication."""

import json
import sqlite3

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.timeline import identity
from mcpref.datasets.continuous.views.reader import ShardReader
from mcpref.infrastructure.shards.store import complete_shards, shard_path


def create(path):
    path.unlink(missing_ok=True)
    for suffix in ("-journal", "-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE fragments (id TEXT PRIMARY KEY, part TEXT, body TEXT);
        CREATE TABLE endpoints (id TEXT PRIMARY KEY, fragment TEXT, completion TEXT, body TEXT, degree INTEGER DEFAULT 0);
        CREATE INDEX completions ON endpoints(fragment, completion, id);
        CREATE TABLE aliases (path TEXT PRIMARY KEY, endpoint TEXT, original_id TEXT);
        CREATE INDEX endpoint_aliases ON aliases(endpoint);
        CREATE TABLE edges (id TEXT PRIMARY KEY, a TEXT, b TEXT, part TEXT, body TEXT);
        CREATE INDEX edges_a ON edges(a, id);
        CREATE INDEX edges_b ON edges(b, id);
        CREATE TABLE excluded (id TEXT PRIMARY KEY, reason TEXT, body TEXT);
    """)
    return db


def endpoint(db, info, alias, original, history, reader):
    ids = info["observation_ids"]
    if not ids or identity(ids) != info["id"]:
        raise ValueError("Continuous preference history identity differs")
    for expected, (shard, position) in zip(ids, info["frames"], strict=True):
        row = reader.metadata(shard)[position]
        if row["id"] != expected or row["fragment_id"] != info["fragment_id"]:
            raise ValueError(
                "Continuous preference endpoint references different observations"
            )
    fragment = info["fragment_id"]
    if db.execute("SELECT 1 FROM fragments WHERE id=?", (fragment,)).fetchone() is None:
        raise ValueError("Continuous preference endpoint has no fragment")
    body = json.dumps(
        {
            "id": info["id"],
            "fragment_id": fragment,
            "observation_ids": ids[-history:],
            "full_observation_ids": ids,
            "source": reader.metadata(info["frames"][0][0])[info["frames"][0][1]][
                "source"
            ],
        }
    )
    old = db.execute("SELECT body FROM endpoints WHERE id=?", (info["id"],)).fetchone()
    if old is not None and old[0] != body:
        raise ValueError("Aliased preference histories differ")
    db.execute(
        "INSERT OR IGNORE INTO endpoints(id,fragment,completion,body) VALUES (?,?,?,?)",
        (info["id"], fragment, ids[-1], body),
    )
    db.execute("INSERT INTO aliases VALUES (?,?,?)", (alias, info["id"], original))


def old_endpoints(db, cfg, history, reader):
    root = REPO / cfg.dataset / "fragments"
    count = complete_shards(root)
    for k in tqdm(
        range(count), total=count, desc="Continuous endpoints", unit="fragment"
    ):
        with (shard_path(root, k) / "mappings.jsonl").open() as stream:
            for line in stream:
                mapped = json.loads(line)
                if mapped["status"] != "mapped":
                    continue
                info = json.loads((REPO / mapped["clip_ref"] / "clip.json").read_text())
                if (
                    info["dataset"] != cfg.dataset
                    or info["id"] != mapped["canonical_history_id"]
                ):
                    raise ValueError(
                        "Continuous mapping references a different history"
                    )
                endpoint(
                    db, info, mapped["endpoint"], mapped["original_id"], history, reader
                )
        db.commit()


def edge(db, record):
    endpoints = []
    for side in ("a", "b"):
        found = db.execute(
            "SELECT endpoint FROM aliases WHERE path=?", (record[side],)
        ).fetchone()
        if found is None:
            raise ValueError("Continuous preference edge has no mapped endpoint")
        row = db.execute(
            "SELECT fragment,body FROM endpoints WHERE id=?", found
        ).fetchone()
        part = db.execute(
            "SELECT part FROM fragments WHERE id=?", (row[0],)
        ).fetchone()[0]
        endpoints.append((found[0], json.loads(row[1]), part))
    a, b = endpoints
    if a[2] != b[2]:
        raise ValueError("Continuous preference edge crosses the frozen split")
    if a[1]["observation_ids"] == b[1]["observation_ids"]:
        db.execute(
            "INSERT INTO excluded VALUES (?,?,?)",
            (record["id"], "collapsed_pair", json.dumps(record)),
        )
        return
    body = record | {
        "a": a[0],
        "b": b[0],
        "original_a": record["a"],
        "original_b": record["b"],
    }
    db.execute(
        "INSERT INTO edges VALUES (?,?,?,?,?)",
        (record["id"], a[0], b[0], a[2], json.dumps(body)),
    )
    for value in (a[0], b[0]):
        db.execute("UPDATE endpoints SET degree=degree+1 WHERE id=?", (value,))


def edges(db, cfg, history, reader):
    root = REPO / cfg.dataset / "preference-map"
    for k in tqdm(
        range(complete_shards(root)), desc="Continuous original edges", unit="shard"
    ):
        with (shard_path(root, k) / "pairs.jsonl").open() as stream:
            for line in stream:
                edge(db, json.loads(line))
        db.commit()
    if cfg.pair_labels_dir is None:
        return
    labels, pool = REPO / cfg.pair_labels_dir, REPO / cfg.pair_clips_dir
    label_id = digest(labels / "run.json")
    for k in tqdm(
        range(complete_shards(labels / "shards")),
        desc="Continuous additional edges",
        unit="shard",
    ):
        pairs = json.loads(
            (shard_path(labels / "shards", k) / "pairs.json").read_text()
        )
        for i, pair in enumerate(pairs):
            paths = []
            for side in ("a", "b"):
                ref = pair[side]
                path = shard_path(pool / "shards", ref["shard"]) / "clips" / ref["id"]
                alias = str(path.relative_to(REPO))
                if (
                    db.execute(
                        "SELECT 1 FROM aliases WHERE path=?", (alias,)
                    ).fetchone()
                    is None
                ):
                    info = json.loads((path / "clip.json").read_text())
                    if info["dataset"] != cfg.dataset or not info["classical_valid"]:
                        raise ValueError(
                            "Additional preference endpoint is not an eligible dataset history"
                        )
                    endpoint(db, info, alias, ref["id"], history, reader)
                paths.append(alias)
            edge(
                db,
                {
                    "id": f"additional:{label_id}:{k}:{i}",
                    "a": paths[0],
                    "b": paths[1],
                    "labels": pair["labels"],
                    "original": pair,
                },
            )
        db.commit()


def build(db, cfg, maths, run):
    reader = ShardReader(REPO / cfg.dataset, "", 0, run["width"], run["height"])
    old_endpoints(db, cfg, maths["preference_history"], reader)
    edges(db, cfg, maths["preference_history"], reader)
    db.commit()
