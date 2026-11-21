"""Continuous populations and chronological descriptors; specs/pretrain-continuous.md."""

import argparse
import dataclasses
import json
from pathlib import Path

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.gui.mask import load_config as mask_config
from mcpref.gui.mask import load_masks
from mcpref.infrastructure.runs.files import json_bytes, write_json
from mcpref.infrastructure.shards.store import (
    complete_shards,
    publish,
    shard_path,
    write_shard,
)
from mcpref.training.pretrain.continuous.data import graph, metadata
from mcpref.training.pretrain.continuous.data.config import (
    IndexConfig,
    load_config,
    mathematics,
)
from mcpref.training.pretrain.continuous.data.fingerprints import FORMAT, sources


def stored(out):
    for k in range(complete_shards(out / "shards")):
        directory = shard_path(out / "shards", k)
        state = json.loads((directory / "state.json").read_text())
        if digest(directory / "fragments.json") != state["sha256"]:
            raise ValueError("Continuous index shard checksum differs")
        yield from json.loads((directory / "fragments.json").read_text())


def save(out, k, items):
    payload = json_bytes(items)
    target = shard_path(out / "shards", k)
    if target.exists():
        if (target / "fragments.json").read_bytes() != payload:
            raise ValueError("Continuous index fragment order changed")
        return
    with write_shard(out / "shards", k) as staged:
        (staged / "fragments.json").write_bytes(payload)
        write_json(staged / "state.json", {"sha256": digest(staged / "fragments.json")})


def descriptors(cfg, maths, masks):
    root, out = REPO / cfg.dataset, REPO / cfg.out_dir
    split = json.loads((root / "split.json").read_text())
    existing, batch, k = iter(stored(out)), [], 0
    count = complete_shards(root / "fragments")
    for n in tqdm(
        range(count), total=count, desc="Continuous populations", unit="fragment"
    ):
        summary = json.loads(
            (shard_path(root / "fragments", n) / "summary.json").read_text()
        )
        for fragment in summary["streams"]:
            part = split["fragment_parts"][fragment["id"]]
            if part != split["groups"][fragment["world_group"]]:
                raise ValueError("Continuous fragment and world split differ")
            base = {
                key: fragment[key]
                for key in (
                    "id",
                    "world_group",
                    "first_shard",
                    "next_shard",
                    "observations",
                    "origin",
                )
            } | {"part": part}
            previous = next(existing, None)
            if previous is not None:
                if any(previous[key] != value for key, value in base.items()):
                    raise ValueError("Continuous source descriptor differs on resume")
                item = previous
            else:
                item = base | metadata.population(
                    root, fragment, masks, maths["tbptt_states"]
                )
            batch.append(item)
            if len(batch) == cfg.items_per_shard:
                save(out, k, batch)
                k, batch = k + 1, []
    if next(existing, None) is not None:
        raise ValueError("Continuous index has extra fragment descriptors")
    if batch:
        save(out, k, batch)


def database(cfg, maths, run):
    out = REPO / cfg.out_dir
    staged = out / "catalogue.sqlite.part"
    db = graph.create(staged)
    totals = {
        part: dict.fromkeys(
            ("N_phi", "N_pi", "N_P", "M", "observations", "fragments"), 0
        )
        for part in ("train", "validation")
    }
    try:
        for item in stored(out):
            db.execute(
                "INSERT INTO fragments VALUES (?,?,?)",
                (item["id"], item["part"], json.dumps(item)),
            )
            total = totals[item["part"]]
            for key in ("N_phi", "N_pi", "M", "observations"):
                total[key] += item[key]
            total["fragments"] += bool(item["observations"])
        db.commit()
        graph.build(db, cfg, maths, run)
        for part, count in db.execute("SELECT part,count(*) FROM edges GROUP BY part"):
            totals[part]["N_P"] = count
        excluded = db.execute("SELECT count(*) FROM excluded").fetchone()[0]
    finally:
        db.close()
    publish(staged, out / "catalogue.sqlite")
    return totals, excluded


def run(cfg: IndexConfig, path: Path):
    maths = mathematics(cfg)
    masks = load_masks(mask_config(REPO / maths["action_mask"]))
    fingerprint, dataset_run = sources(cfg, path, maths, masks)
    out, root = REPO / cfg.out_dir, REPO / cfg.dataset
    if out == root or out in root.parents or root in out.parents:
        raise ValueError("Continuous index and dataset destinations must be separate")
    out.mkdir(parents=True, exist_ok=True)
    run_path = out / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if previous.get("format") != FORMAT or previous["fingerprint"] != fingerprint:
            raise ValueError(
                "Continuous index inputs changed; use a new output directory"
            )
        if previous.get("complete"):
            if digest(out / "catalogue.sqlite") != previous["catalogue_sha256"]:
                raise ValueError("Continuous index catalogue checksum differs")
            for _ in stored(out):
                pass
            return previous
    header = {
        "format": FORMAT,
        "pretrain_mode": "continuous",
        "complete": False,
        "config": dataclasses.asdict(cfg),
        "mathematics": maths,
        "index_config": str(path.resolve().relative_to(REPO)),
        "fingerprint": fingerprint,
        "width": dataset_run["width"],
        "height": dataset_run["height"],
    }
    write_json(run_path, header)
    descriptors(cfg, maths, masks)
    totals, collapsed = database(cfg, maths, dataset_run)
    header |= {
        "complete": True,
        "populations": totals,
        "excluded_pairs": {"collapsed_pair": collapsed},
        "catalogue_sha256": digest(out / "catalogue.sqlite"),
    }
    write_json(run_path, header)
    return header


build_index = run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    path = parser.parse_args().config
    run(load_config(path), path)


if __name__ == "__main__":
    main()
