"""`load_classifier` of src/mcpref/recognition/classifier.py against specs/gui-classifier.md
(L): each check, its RuntimeError and message (paths as written) and their order; the
catalogue loaded after its hash check; the bar from step 3 on, closed on an abort."""

import numpy as np
import pytest
from classifier_reads import H, W, opens, pair, rewrite, world
from classifier_world import Masks, catalogue_text, load
from progress_bars import Bar, assert_progress, screen

pytestmark = pytest.mark.usefixtures("away")


def wrong_size(masks: Masks, k: str = "inv", member: str = "m1") -> None:
    """A (d) pair of H + 1 rows: its T is W x (H + 1)."""
    rewrite(masks, k, member, pair(np.random.default_rng(1), H + 1, W))


def empty_mask(masks: Masks, k: str = "inv", member: str = "m3") -> None:
    t, m = masks.classes[k][member]
    rewrite(masks, k, member, (t, np.zeros_like(m)))


def unloadable(masks: Masks) -> None:
    """A catalogue load_screens rejects (an unknown key), other bytes than run.json's."""
    masks.catalogue.write_text(catalogue_text(masks.classes) + "unknown: 1\n")


def no_run(masks: Masks) -> None:
    (masks.out / "run.json").unlink()


SIZE = f"holds {W}x{H + 1} templates, not {W}x{H}"
CHECKS = [  # (L) in order; each case also breaks every later check
    ("{d} is not complete", no_run),
    ("{d} was built from another catalogue than {s}", unloadable),
    ("{d} " + SIZE, wrong_size),  # inv/m1, before inv/m3 in member order
    ("{d}: the mask of d/m3 of inv is empty", empty_mask),
]


@pytest.mark.parametrize("i", range(len(CHECKS)), ids=[c[1].__name__ for c in CHECKS])
def test_checks_in_order(tmp_path, repo, capfd, i):
    """(L): the first failing check aborts with its RuntimeError, paths as written; the
    bar only from step 3, left closed on the abort."""
    masks = world(tmp_path, repo)
    for _, breaking in CHECKS[i:]:
        breaking(masks)
    d, s = f"./{masks.masks_dir}", f"./{masks.screens}"
    with pytest.raises(RuntimeError) as info:
        load(masks, masks_dir=d, screens=s)
    assert type(info.value) is RuntimeError
    assert str(info.value) == CHECKS[i][0].format(d=d, s=s)
    out, err = capfd.readouterr()
    assert out == "" and (err == "") == (i < 2), repr(err)
    if err:
        assert_progress(err)
        assert screen(err) == [Bar("classifier", 0, 4, "shard")]


@pytest.mark.parametrize("complete", [None, False])
def test_incomplete_run(tmp_path, repo, complete):
    """(L) step 1: run.json as written at the start of a gui-masks run, or not true."""
    masks = world(tmp_path, repo)
    masks.write_run(complete)
    with pytest.raises(RuntimeError, match=f"^{masks.masks_dir} is not complete$"):
        load(masks)


def test_catalogue_loaded_after_its_hash_check(tmp_path, repo, capfd):
    """(L) step 2: a catalogue load_screens rejects, whose hash run.json holds, fails in
    load_screens (ValueError) before any shard is opened or the bar drawn."""
    masks = world(tmp_path, repo)
    unloadable(masks)
    masks.write_run(True)
    with opens() as rec, pytest.raises(ValueError):
        load(masks)
    assert not any(p.suffix == ".npz" for p in rec.paths)
    assert capfd.readouterr() == ("", "")


def test_templates_of_another_size(tmp_path, repo):
    """(L) step 3: W x H templates against a caller's H x W: <w>x<h> from T's shape."""
    masks = world(tmp_path, repo)
    with pytest.raises(RuntimeError) as info:
        load(masks, width=H, height=W)
    assert str(info.value) == f"{masks.masks_dir} holds {W}x{H} templates, not {H}x{W}"


@pytest.mark.parametrize(
    "breaks, message",
    [
        ([(wrong_size, "inv", "m1"), (empty_mask, "inv", "m1")], SIZE),  # same pair
        ([(empty_mask, "inv", "m1"), (wrong_size, "inv", "m3")], "the mask of d/m1 of inv"),
        ([(empty_mask, "inv", "m3"), (wrong_size, "craft", "main")], "the mask of d/m3"),
        ([(empty_mask, "craft", "é1"), (wrong_size, "chest", "main")], SIZE),
        ([(empty_mask, "craft", "é1")], "the mask of d/é1 of craft is empty"),
    ],
)  # fmt: skip
def test_shard_checks_in_shard_then_member_order(tmp_path, repo, breaks, message):
    """(L) step 3: per shard in order, per pair in member order (the files list members
    in reverse), the size, then the mask."""
    masks = world(tmp_path, repo)
    for breaking, k, m in breaks:
        breaking(masks, k, m)
    with pytest.raises(RuntimeError, match=message):
        load(masks)


def test_thresholds_read_after_the_shards(tmp_path, repo):
    """(L): step 3 fails before step 4 would read thresholds.json."""
    masks = world(tmp_path, repo)
    empty_mask(masks)
    (masks.out / "thresholds.json").unlink()
    with pytest.raises(RuntimeError, match="the mask of d/m3 of inv is empty"):
        load(masks)
