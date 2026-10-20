"""Config, Behaviour and `main` of specs/train-index.md: (R), (R'), (W); (I), (1)-(3),
(4) in items.py."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from mcpref import REPO, config_mapping, positive_int
from mcpref.cli.failure import FAILURES, failed
from mcpref.pretrain.items import PARTS, read_inputs, split
from mcpref.recognition.stored import shard_bar
from mcpref.runs.files import check_fingerprints, finished_run, sha256, write_json
from mcpref.shards.store import complete_shards, write_shard

PROG = "mcpref.pretrain.index"
HASHED = ("actions", "clips", "screens", "gui_actions", "overlay", "overlay_screens")
CLIP_FIELDS = ("config.width", "config.height", "config.states")
CLIP_FIELDS += (*(f"sha256.{k}" for k in HASHED), "classifier")
PAIR_DIRS = ("pair_clips_dir", "pair_labels_dir")
FINGERPRINTS = ("config", "clips", "pair_clips", "pair_labels")
ABSENT = object()


@dataclass(frozen=True)
class TrainIndexConfig:
    """Notation: configs/train-index.yaml as written; paths relative to the repository
    root, resolved against it where read or written; pair_* both null without pairs."""

    clips_dir: str
    pair_clips_dir: str | None
    pair_labels_dir: str | None
    out_dir: str
    validation: str
    seed: int
    items_per_shard: int


def relative(name: str, value: object) -> None:
    """Notation: a non-empty relative path string without a `..` component."""
    path = PurePosixPath(value) if type(value) is str and value else None
    if path is None or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{name} {value!r} is not a non-empty relative path without .."
        )


def load_config(path: Path) -> TrainIndexConfig:
    """Interface: the YAML file, validated, into TrainIndexConfig as written."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, TrainIndexConfig)
    paired = any(raw[k] is not None for k in PAIR_DIRS)
    for k in ("clips_dir", "out_dir", *(PAIR_DIRS if paired else ())):
        relative(k, raw[k])
    v = raw["validation"]
    if type(v) is not str or not 0 <= float(v) < 1:
        raise ValueError(f"validation {v!r} is not a string of a number in [0, 1)")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**32:
        raise ValueError(f"seed {raw['seed']!r} is not an integer in [0, 2^32)")
    positive_int("items_per_shard", raw["items_per_shard"])
    return TrainIndexConfig(**raw)


def clip_fields(run: dict) -> dict:
    """Notation: the clip-defining fields present in run.json, {dotted name: value}."""
    fields = {}
    for name in CLIP_FIELDS:
        node = run
        for key in name.split("."):
            node = node.get(key, ABSENT) if isinstance(node, dict) else ABSENT
        if node is not ABSENT:
            fields[name] = node
    return fields


def checks(cfg: TrainIndexConfig) -> tuple[list[dict], dict | None]:
    """(R) 1-4 in order: the clip-defining fields of C (and Q_c); Q_l's run.json."""
    paired = cfg.pair_clips_dir is not None
    dirs = [cfg.clips_dir, *([cfg.pair_clips_dir] if paired else [])]
    runs = [finished_run(REPO / d, d, "clips") for d in dirs]
    fields = [clip_fields(run) for run in runs]
    for d, f in zip(dirs, fields):
        lacking = [n for n in CLIP_FIELDS if n not in f]
        if lacking:
            raise RuntimeError(f"{d} lacks {lacking}")
    if not paired:
        return fields, None
    c, q = fields
    differing = [n for n in CLIP_FIELDS if json.dumps(q[n]) != json.dumps(c[n])]
    if differing:
        raise RuntimeError(
            f"{cfg.pair_clips_dir} differs from {cfg.clips_dir} in {differing}"
        )
    labels = finished_run(REPO / cfg.pair_labels_dir, cfg.pair_labels_dir, "complete")
    return fields, labels


def header(
    cfg: TrainIndexConfig, path: Path, c: dict, fingerprints: list, labels: dict | None
) -> dict:
    """(R'): Q_l labels Q_c; check_fingerprints of an existing {out_dir}/run.json; then
    run.json written: the config as written, the fingerprints (4), L, W, H of C."""
    if labels is not None and labels["pair_clips"] != fingerprints[1]:
        raise RuntimeError(
            f"{cfg.pair_labels_dir} labels other clips than {cfg.pair_clips_dir}"
        )
    sha = dict(zip(FINGERPRINTS, [sha256(path), *fingerprints]))
    check_fingerprints(REPO / cfg.out_dir / "run.json", sha, cfg.out_dir)
    run = {
        "config": yaml.safe_load(path.read_text()),
        "sha256": sha,
        "states": c["config.states"],
        "width": c["config.width"],
        "height": c["config.height"],
    }
    write_json(REPO / cfg.out_dir / "run.json", run)
    return run


def write_parts(out: Path, shuffled: list[list[dict]], B: int) -> None:
    """(W): shards h_x .. ceil(N_x / B) - 1 of train/, then validation/, each items.json
    by write_shard (bar `index shards`, initial: the items of the complete shards)."""
    dirs = [out / p for p in PARTS]
    h = [complete_shards(d) for d in dirs]
    n = [-(-len(items) // B) for items in shuffled]
    total = sum(len(items) for items in shuffled)
    initial = sum(min(h_x * B, len(items)) for h_x, items in zip(h, shuffled))
    with shard_bar("index shards", sum(n), sum(h), total, initial, "item") as update:
        for d, items, h_x, n_x in zip(dirs, shuffled, h, n):
            for s in range(h_x, n_x):
                shard = items[s * B : (s + 1) * B]
                with write_shard(d, s) as staged:
                    text = json.dumps({"items": shard}, indent=2) + "\n"
                    (staged / "items.json").write_text(text)
                update(len(shard))


def run(cfg: TrainIndexConfig, path: Path) -> None:
    """Behaviour: (R), (I), (1)-(3), (R'), (W) in order; run.json complete at last."""
    fields, labels = checks(cfg)
    entries, fingerprints = read_inputs(cfg, fields)
    shuffled, counts = split(cfg, entries)
    head = header(cfg, path, fields[0], fingerprints, labels)
    write_parts(REPO / cfg.out_dir, shuffled, cfg.items_per_shard)
    write_json(
        REPO / cfg.out_dir / "run.json", head | {"complete": True, "counts": counts}
    )


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.pretrain.index <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.config)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
