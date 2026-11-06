"""Independent fixed-reference selection oracle of specs/pretrain-diagnostics.md (10), (17)."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from mcpref.pretrain.checks import index_run
from mcpref.pretrain.config import load_config
from mcpref.pretrain.diagnostics import GlobalFit, Stratum
from mcpref.pretrain.reference import select_reference

from .index_run import repo_root
from .loop_world import World, p


def labelled_world(
    train: list[list[float | None]], validation: list[list[float | None]]
) -> World:
    def items(shards):
        return [[{**p(i % 4), "labels": {"p": value, "inc": int(value is None)}}
                 for i, value in enumerate(shard)] for shard in shards]  # fmt: skip

    return World(train=items(train), validation=items(validation))


def full_world() -> World:
    train = [[0.0, 0.5, 1.0, None][i % 4] for i in range(64)]
    validation = [[0.0, 0.5, 1.0, None][(3 - i) % 4] for i in range(64)]
    chunks = lambda values: [values[i : i + 3] for i in range(0, 64, 3)]
    return labelled_world(chunks(train), chunks(validation))


def stratum(item: dict) -> tuple[str, float | None]:
    labels = item["labels"]
    return (
        ("inc", None)
        if labels["inc"]
        else (str(float(labels["p"])), float(labels["p"]))
    )


def fit_of(world: World) -> GlobalFit:
    parts = {"train": world.train, "validation": world.validation}
    values = sorted({value for shards in parts.values() for items in shards for item in items
                     for key, value in [stratum(item)] if value is not None})  # fmt: skip
    keys = [(str(float(value)), value) for value in values] + [("inc", None)]
    counts = {part: Counter(stratum(item)[0] for items in shards for item in items)
              for part, shards in parts.items()}  # fmt: skip
    strata = tuple(Stratum(key, "inc" if value is None else "comparable", value,
                           counts["train"][key], counts["validation"][key]) for key, value in keys)  # fmt: skip
    train = [item for items in world.train for item in items]
    comparable = [item for item in train if not item["labels"]["inc"]]
    n, nc = len(train), len(comparable)
    u = (n - nc) / n
    v = sum(item["labels"]["p"] for item in comparable) / nc if nc else None
    logit = lambda x: None if x is None or x in (0, 1) else math.log(x / (1 - x))
    return GlobalFit(n, n - nc, nc, sum((item["labels"]["p"] for item in comparable), 0.0) if nc else None,
                     u, v, logit(u), logit(v), strata)  # fmt: skip


def allocation(
    counts: Counter, total: int, pairs: int, order: list[str]
) -> dict[str, int]:
    if total < pairs:
        raise RuntimeError(f"reference needs {pairs}")
    raw = {key: pairs * counts[key] / total for key in order}
    result = {key: math.floor(raw[key]) for key in order}
    for key in sorted(order, key=lambda k: (-(raw[k] - result[k]), order.index(k)))[
        : pairs - sum(result.values())
    ]:
        result[key] += 1
    return result


def selection_of(world: World, seed: int, pairs: int) -> tuple[dict, GlobalFit]:
    fit = fit_of(world)
    order = [s.key for s in fit.strata]
    rng = np.random.Generator(np.random.PCG64([seed, 4]))
    selected = []
    for part, shards in (("train", world.train), ("validation", world.validation)):
        counts = Counter(stratum(item)[0] for items in shards for item in items)
        total = sum(counts.values())
        allocated = allocation(counts, total, pairs, order)
        for key in order:
            positions = sorted(rng.permutation(counts[key])[: allocated[key]].tolist())
            wanted = set(positions)
            ordinal = 0
            for shard, items in enumerate(shards):
                for item_index, item in enumerate(items):
                    if stratum(item)[0] != key:
                        continue
                    if ordinal in wanted:
                        selected.append(
                            {
                                **item,
                                "part": part,
                                "stratum": key,
                                "shard": shard,
                                "item_index": item_index,
                            }
                        )
                    ordinal += 1
    return {"version": 1, "seed": seed, "pairs_per_part": pairs, "items": selected}, fit


def select(loop):
    with repo_root(loop.repo):
        cfg = load_config(loop.path)
        return select_reference(cfg, index_run(cfg)), cfg, index_run(cfg)


def write_selection(directory: Path, selection: dict, fit: GlobalFit) -> None:
    directory.mkdir(parents=True)
    (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (directory / "baselines.json").write_text(
        json.dumps(__import__("dataclasses").asdict(fit), indent=2) + "\n"
    )


def count_item_reads(monkeypatch, fail_at: int | None = None) -> list[str]:
    real = Path.read_bytes
    reads = []

    def read(path, *args, **kwargs):
        if path.name == "items.json":
            reads.append(str(path))
            if fail_at is not None and len(reads) == fail_at:
                raise OSError("injected index read failure")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    return reads
