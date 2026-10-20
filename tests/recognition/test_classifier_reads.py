"""`load_classifier` of src/mcpref/recognition/classifier.py against specs/gui-classifier.md
(L), Datasets and the bar `classifier`: each catalogue shard opened once, only its (d) arrays
read, nothing else of {masks_dir} but run.json and thresholds.json; gamma reading nothing
from disk; the bar on every load, and its line for an empty catalogue."""

import shutil

import numpy as np
import pytest
from classifier_reads import THRESHOLDS, H, W, array_reads, opens, world
from classifier_world import Masks, load, shard_arrays
from progress_bars import Bar, assert_progress, screen, states

pytestmark = pytest.mark.usefixtures("away")


def test_each_shard_opened_once_only_d_arrays_read(tmp_path, repo):
    """Datasets: each catalogue shard opened once, its (d) arrays alone read; nothing else
    of {masks_dir} but run.json and thresholds.json (a (b) shard is never opened)."""
    masks = world(tmp_path, repo)
    decoy = masks.out / "templates/b/000000/pairs.npz"
    decoy.parent.mkdir(parents=True)
    np.savez(decoy, **shard_arrays(masks.classes["inv"], W, H))
    with opens() as rec, array_reads() as keys:
        load(masks)
    under = [p for p in rec.paths if masks.out.resolve() in p.parents]
    catalogue = masks.out.resolve() / "templates/catalogue"
    shards = [catalogue / f"{s:06d}" / "pairs.npz" for s in range(4)]
    assert sorted(p for p in under if p.suffix == ".npz") == shards
    assert set(under) == {*shards, masks.out.resolve() / "run.json"} | {
        masks.out.resolve() / "thresholds.json"
    }
    d_keys = {f"d/{m}/{a}" for ms in masks.classes.values() for m, p in ms.items() if p for a in "TM"}  # fmt: skip
    assert set(keys) == d_keys


def test_gamma_reads_nothing_from_disk(tmp_path, repo):
    """(L): gamma holds its model; with the outputs and the catalogue moved away it opens
    no file and labels as before."""
    masks = world(tmp_path, repo)
    gamma, _ = load(masks)
    rng = np.random.default_rng(3)
    frames = [
        masks.classes["inv"]["m1"][0],
        *rng.integers(0, 256, (3, H, W, 3), np.uint8),
    ]
    before = gamma(frames)
    shutil.move(masks.root, tmp_path / "moved")
    with opens() as rec, array_reads() as keys:
        assert gamma(frames) == before
    assert rec.paths == [] and keys == []


def test_bar(tmp_path, repo, capfd):
    """Bar `classifier` (unit shard): total |K| = 4 from 0, no postfix, drawn on every
    load (never `already done`); nothing else printed."""
    masks = world(tmp_path, repo)
    for _ in range(2):
        load(masks)
        out, err = capfd.readouterr()
        assert out == ""
        assert_progress(err)
        assert states(err)[0] == Bar("classifier", 0, 4, "shard")
        assert screen(err) == [Bar("classifier", 4, 4, "shard")]


def test_no_class(tmp_path, repo, capfd):
    """|K| = 0: the line `classifier: nothing to do`; every frame `other`."""
    masks = Masks(tmp_path, repo, {}, THRESHOLDS, W, H).write()
    gamma, _ = load(masks)
    assert gamma([np.zeros((H, W, 3), np.uint8)] * 2) == ["other"] * 2
    assert capfd.readouterr() == ("", "classifier: nothing to do\n")
