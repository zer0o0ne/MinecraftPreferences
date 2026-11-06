"""Equation (1), C3: exact disk shuffle, source order, sharding, tail, and resume."""

import numpy as np
import pytest
import tqdm
from pretrain.index_run import repo_root
from progress_bars import screen, states

from mcpref.rl.index import build_index, read_item

from .helpers import (
    config,
    pair_items,
    read_json,
    repository,
    shard_items,
    write_json,
    write_pairs,
)


def prepare(tmp_path, repo, cycle=0, pairs=2, steps=2, horizon=2, **changes):
    root = tmp_path / "repo"
    repository(root, repo, steps=steps, horizon=horizon, pairs=pairs)
    cfg = config(tmp_path / "rl.yaml", **changes)
    pair_dir = root / f"data/rl_preference_pairs/{cycle:06d}"
    write_pairs(pair_dir, pair_items(pairs))
    if cycle:
        write_json(
            root / f"data/rl_cycle/{cycle - 1:06d}/tail/shards/000000/items.json",
            {
                "next_step": cycle * steps,
                "indices": list(
                    range(max(0, cycle * steps - horizon + 1), cycle * steps)
                ),
            },
        )
    return root, cfg, pair_dir


def source(cycle, steps, horizon, pairs):
    fresh = [
        {"kind": "fresh", "index": t} for t in range(cycle * steps, (cycle + 1) * steps)
    ]
    u = (cycle + 1) * steps
    pending = (
        []
        if not cycle
        else [
            {"kind": "pending", "index": t}
            for t in range(max(0, cycle * steps - horizon + 1), cycle * steps)
            if t + horizon <= u
        ]
    )
    preferences = [{"kind": "preference", **item} for item in pair_items(pairs)]
    return [*fresh, *pending, *preferences], pending


def test_exact_generator_permutation_source_order_shards_and_counts(tmp_path, repo):
    root, cfg, pair_dir = prepare(tmp_path, repo, cycle=1, pairs=3, items_per_shard=2,
                                  batch=2, epochs=2)  # fmt: skip
    with repo_root(root):
        out = build_index(cfg, 1, 2, pair_dir)
    records, _ = source(1, 2, 2, 3)
    order = np.random.Generator(np.random.PCG64([cfg.seed, 2])).permutation(
        len(records), axis=0
    )
    expected = [records[int(i)] for i in order]
    order_shards = shard_items(out.parent / "order" / "shards", "indices.json")
    assert [
        item for shard in order_shards for item in shard["indices"]
    ] == order.tolist()
    assert [len(shard["indices"]) for shard in order_shards] == [2, 2, 2]
    mixed = shard_items(out / "shards")
    assert [item for shard in mixed for item in shard["items"]] == expected
    for k, shard in enumerate(mixed):
        for j, item in enumerate(shard["items"]):
            assert read_item(out, k, j) == item
    run = read_json(out / "run.json")
    assert run["complete"] and run["counts"] == {
        "fresh": 2,
        "pending": 1,
        "ppo": 2,
        "preference": 3,
        "items": 6,
    }
    assert run["batches"] == 3 and run["steps"] == 6
    assert read_json(out.parent / "tail/shards/000000/items.json") == {
        "next_step": 4,
        "indices": [3],
    }
    assert not list(out.parent.glob("*.npy"))


def test_cycle_zero_and_zero_denominators(tmp_path, repo):
    root, cfg, pair_dir = prepare(tmp_path, repo, pairs=0, horizon=3)
    with repo_root(root):
        out = build_index(cfg, 0, 0, pair_dir)
    run = read_json(out / "run.json")
    assert run["counts"] == {
        "fresh": 2,
        "pending": 0,
        "ppo": 0,
        "preference": 0,
        "items": 2,
    }
    assert read_json(out.parent / "tail/shards/000000/items.json") == {
        "next_step": 2,
        "indices": [0, 1],
    }


def test_disk_shuffle_uses_memmap_not_generator_permutation_or_full_arange(
    tmp_path, repo, monkeypatch
):
    root, cfg, pair_dir = prepare(tmp_path, repo, pairs=20, items_per_shard=7)
    records, _ = source(0, 2, 2, 20)
    expected = np.random.Generator(np.random.PCG64([cfg.seed, 2])).permutation(
        len(records)
    )
    maps = []
    real_memmap, real_arange = np.memmap, np.arange

    def memmap(*args, **kwargs):
        maps.append((args, kwargs))
        return real_memmap(*args, **kwargs)

    real_generator = np.random.Generator

    class GuardedGenerator:
        def __init__(self, bit_generator):
            self.inner = real_generator(bit_generator)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def permutation(self, *args, **kwargs):
            raise AssertionError("the index must use memmap.shuffle, not permutation")

    def arange(start, stop=None, *args, **kwargs):
        size = start if stop is None else stop - start
        assert size != len(records), "a whole-index arange is forbidden"
        return real_arange(start, stop, *args, **kwargs)

    monkeypatch.setattr("mcpref.rl.index.np.memmap", memmap)
    monkeypatch.setattr("mcpref.rl.index.np.random.Generator", GuardedGenerator)
    monkeypatch.setattr("mcpref.rl.index.np.arange", arange)
    with repo_root(root):
        out = build_index(cfg, 0, 0, pair_dir)
    got = [i for shard in shard_items(out.parent / "order/shards", "indices.json")
           for i in shard["indices"]]  # fmt: skip
    assert got == expected.tolist()
    assert maps and maps[0][1]["shape"] == (len(records),)
    assert maps[0][1]["dtype"] == np.int64 and maps[0][1]["mode"] == "w+"


class Interrupted(RuntimeError):
    pass


def interrupt_index(monkeypatch, at):
    real = tqdm.tqdm.update
    raised = False

    def update(self, n=1):
        nonlocal raised
        result = real(self, n)
        if self.desc == "rl index" and self.n == at and not raised:
            raised = True
            raise Interrupted
        return result

    monkeypatch.setattr(tqdm.tqdm, "update", update)


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file() and p.name != "run.json"}  # fmt: skip


def test_interrupted_memmap_resume_and_complete_already_done(
    tmp_path, repo, monkeypatch, capsys
):
    baseline, cfg, pairs = prepare(tmp_path / "whole", repo, pairs=4, items_per_shard=2)
    with repo_root(baseline):
        build_index(cfg, 0, 0, pairs)
    expected = tree(baseline)
    root, cfg, pairs = prepare(tmp_path / "cut", repo, pairs=4, items_per_shard=2)
    interrupt_index(monkeypatch, 8)
    with pytest.raises(Interrupted), repo_root(root):
        build_index(cfg, 0, 0, pairs)
    capsys.readouterr()
    assert (root / "data/rl_cycle/000000/order/shards/000000/indices.json").exists()
    with repo_root(root):
        out = build_index(cfg, 0, 0, pairs)
    resumed = capsys.readouterr()
    assert tree(root) == expected
    assert states(resumed.err, "rl index")[0].n > 0
    with repo_root(root):
        build_index(cfg, 0, 0, pairs)
    assert screen(capsys.readouterr().err) == ["rl index: already done"]
    assert out == root / "data/rl_cycle/000000/index"


@pytest.mark.parametrize("indices,next_step", [([0, 2], 3), ([1, 1], 3), ([1, 0], 2)])
def test_incoming_tail_continuity_and_uniqueness(tmp_path, repo, indices, next_step):
    root, cfg, pairs = prepare(tmp_path, repo, cycle=1)
    write_json(root / "data/rl_cycle/000000/tail/shards/000000/items.json",
               {"next_step": next_step, "indices": indices})  # fmt: skip
    with pytest.raises((RuntimeError, ValueError)), repo_root(root):
        build_index(cfg, 1, 2, pairs)


def test_pair_fingerprint_covers_immutable_items_not_mutable_run(tmp_path, repo):
    root, cfg, pairs = prepare(tmp_path, repo)
    write_json(pairs / "run.json", {"phase": "correction", "mutable": 1})
    with repo_root(root):
        out = build_index(cfg, 0, 0, pairs)
    before = read_json(out / "run.json")["sha256"]
    write_json(pairs / "run.json", {"phase": "history", "mutable": 999})
    with repo_root(root):
        build_index(cfg, 0, 0, pairs)
    assert read_json(out / "run.json")["sha256"] == before
    assert isinstance(before, str) and len(before) == 64
