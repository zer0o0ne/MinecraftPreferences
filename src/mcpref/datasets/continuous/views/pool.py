"""Preference candidates selected exclusively from the completed shared dataset."""

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref.config import REPO, config_mapping, exact, read_yaml, relative
from mcpref.datasets.clips.collect import load_config as perception_config
from mcpref.datasets.clips.sample import uniform
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.timeline import identity
from mcpref.datasets.pool.graph import comparisons
from mcpref.datasets.pool.run import load_config as pool_config
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.runs.files import json_bytes, write_json
from mcpref.infrastructure.shards.store import complete_shards, shard_path, write_shard


@dataclasses.dataclass(frozen=True)
class Config:
    dataset: str
    pool: str
    out_dir: str


def catalogue(root: Path, states: int):
    worlds, seen = {}, set()
    n = complete_shards(root / "fragments")
    for k in tqdm(
        range(n), total=n, desc="Dataset candidates", unit="fragment", mininterval=10
    ):
        directory = shard_path(root / "fragments", k)
        summary = json.loads((directory / "summary.json").read_text())
        streams = {s["id"]: s for s in summary["streams"]}
        origins = {}
        for line in (directory / "mappings.jsonl").read_text().splitlines():
            mapped = json.loads(line)
            if mapped["status"] == "mapped":
                origins.setdefault(mapped["canonical_history_id"], mapped["source"])
        subset = summary["sources"][0]["piece"]["subset"]
        for path in sorted((directory / "clips").glob("*/clip.json")):
            info = json.loads(path.read_text())
            source = origins[info["id"]]
            if not info["classical_valid"] or len(info["states"]) < states:
                continue
            # A candidate is an existing history suffix; its pixels and targets stay referenced.
            count = len(info["states"]) - states
            for key in (
                "observation_ids",
                "frames",
                "gui",
                "classes",
                "pitch",
                "actions",
                "n",
            ):
                info[key] = info[key][count:]
            info["states"] = list(range(states))
            info["id"] = identity(info["observation_ids"])
            if info["id"] in seen:
                continue
            seen.add(info["id"])
            stream = streams[info["fragment_id"]]
            key = (subset, stream["annotation_world"])
            worlds.setdefault(key, defaultdict(list))[source].append(info)
    return worlds


def select(candidates: list, maximum: int, rng, occupied: set):
    remaining = [c for c in candidates if occupied.isdisjoint(c["observation_ids"])]
    chosen = []
    while remaining and len(chosen) < maximum:
        item = remaining.pop(uniform(rng, len(remaining)))
        chosen.append(item)
        occupied.update(item["observation_ids"])
        remaining = [c for c in remaining if occupied.isdisjoint(c["observation_ids"])]
    return chosen


def collect(cfg: Config):
    root, out = REPO / cfg.dataset, REPO / cfg.out_dir
    run = json.loads((root / "run.json").read_text())
    if not run.get("complete"):
        raise ValueError("Preference selection requires a complete shared dataset")
    settings = pool_config(REPO / cfg.pool)
    states = perception_config(REPO / settings.perception).states
    worlds = catalogue(root, states)
    fingerprint = {
        "dataset": digest(root / "run.json"),
        "pool": dataclasses.asdict(settings),
        "catalogue": identity(
            [
                (key, {f: [c["id"] for c in clips] for f, clips in fragments.items()})
                for key, fragments in worlds.items()
            ]
        ),
    }
    source = run["fingerprint"]["source_config"]
    head = source | {
        "config": source["config"] | dataclasses.asdict(settings) | {"states": states},
        "shared_dataset": cfg.dataset,
        "shared_fingerprint": fingerprint,
    }
    path = out / "run.json"
    if (
        path.exists()
        and json.loads(path.read_text())["shared_fingerprint"] != fingerprint
    ):
        raise ValueError("Preference source changed; use a new output directory")
    write_json(path, head)
    pools = defaultdict(list)
    for key in worlds:
        pools[key[0]].append(key)
    rng = np.random.Generator(np.random.PCG64([settings.seed, 3]))
    count, pairs, k = 0, 0, 0
    with tqdm(
        total=min(
            settings.n_clips,
            sum(len(cs) for fs in worlds.values() for cs in fs.values()),
        ),
        desc="Dataset preference clips",
        unit="clip",
        mininterval=10,
    ) as bar:
        while count < settings.n_clips and any(pools.values()):
            active = [s for s, pool in pools.items() if pool]
            subset = active[uniform(rng, len(active))]
            key = pools[subset].pop(uniform(rng, len(pools[subset])))
            fragments = list(worlds[key])
            if len(fragments) > settings.episodes:
                fragments = [
                    fragments[i]
                    for i in sorted(
                        rng.permutation(len(fragments))[: settings.episodes]
                    )
                ]
            clips, selected_by_source, occupied = [], {}, set()
            for j, fragment in enumerate(fragments):
                g = np.random.Generator(np.random.PCG64([settings.seed, 4, k, j]))
                chosen = select(
                    worlds[key][fragment], settings.per_episode, g, occupied
                )
                selected_by_source[fragment] = {c["id"] for c in chosen}
                clips.extend(chosen)
            if len(clips) < 2:
                clips = []
            g = np.random.Generator(np.random.PCG64([settings.seed, 5, k]))
            edges = (
                comparisons(
                    len(clips), settings.partners, exact("cycles", settings.cycles), g
                )
                if clips
                else []
            )
            refs = [{"id": c["id"], "episode": c["fragment_id"]} for c in clips]
            records = [
                {
                    "subset": subset,
                    "relpath": f,
                    "reason": None,
                    "clips": [
                        c["id"]
                        for c in clips
                        if c["id"] in selected_by_source.get(f, set())
                    ],
                    "shared_dataset_fingerprint": fingerprint["dataset"],
                }
                for f in fragments
            ]
            files = {
                "records.json": records,
                "comparisons.json": [{"a": refs[a], "b": refs[b]} for a, b in edges],
            }
            files |= {f"clips/{c['id']}/clip.json": c for c in clips}
            final = shard_path(out / "shards", k)
            if final.exists():
                for name, value in files.items():
                    if (final / name).read_bytes() != json_bytes(value):
                        raise ValueError("Resumed preference candidates differ")
            else:
                with write_shard(out / "shards", k) as staged:
                    for name, value in files.items():
                        target = staged / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(json_bytes(value))
            count += len(clips)
            pairs += len(edges)
            k += 1
            bar.update(len(clips))
    write_json(path, head | {"clips": count, "comparisons": pairs, "complete": True})


def main(argv):
    parser = argparse.ArgumentParser(
        prog="python -m mcpref.datasets.continuous.views.pool"
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = config_mapping(read_yaml(args.config), args.config, Config)
        for key, value in raw.items():
            relative(key, value)
        collect(Config(**raw))
    except FAILURES as error:
        return failed(parser.prog, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
