"""The fingerprints (4) of specs/train-index.md, read from `sha256` of {out_dir}/run.json
and through the rerun check of (R'): one `json.dumps` of the clip-defining fields (default
separators and `ensure_ascii`), then the records.json of the complete shards; the pairs.json
of Q_l's complete shards; keys outside the clip-defining fields free."""

import hashlib
import json

import pytest

from .index_oracle import CLIP_FIELDS, sha
from .index_run import aborted
from .index_world import (
    HAND,
    NO_PAIRS,
    World,
    clips_run,
    edit,
    read,
    record,
    setting,
    write_json,
)


def stored(site) -> dict:
    return read(site.out / "run.json")["sha256"]


def test_clips_fingerprint_by_hand(make_site):
    """C's classifier 'd:été': the head is the ASCII text below (default separators,
    non-ASCII escaped), then records.json of shards 000000, 000001 (not the staged one)."""
    site = make_site(c_run=clips_run(4, {"classifier": "d:été"}), **NO_PAIRS)
    assert site.run().code == 0
    hashes = ", ".join(f'"sha256.{k}": "{sha(k.encode())}"' for k in
                       ("actions", "clips", "screens", "gui_actions", "overlay", "overlay_screens"))  # fmt: skip
    head = '{"config.width": 640, "config.height": 360, "config.states": 16, '
    head += hashes + ', "classifier": "d:\\u00e9t\\u00e9"}'
    shards = site.repo / "w/clips/shards"
    data = head.encode() + b"".join(
        (shards / k / "records.json").read_bytes() for k in ("000000", "000001")
    )
    assert stored(site) == {
        "config": sha(site.path.read_bytes()),
        "clips": hashlib.sha256(data).hexdigest(),
        "pair_clips": None,
        "pair_labels": None,
    }


def test_labels_fingerprint_by_hand(make_site):
    """pair_labels: the pairs.json of shards 000000, 000001 (not the staged one);
    pair_clips: Q_c's (4), as Q_l's run.json holds it."""
    site = make_site()
    assert site.run().code == 0
    shards = site.repo / "w/pair_labels/shards"
    data = b"".join(
        (shards / k / "pairs.json").read_bytes() for k in ("000000", "000001")
    )
    assert stored(site)["pair_labels"] == sha(data)
    pair_clips = read(site.repo / "w/pair_labels/run.json")["pair_clips"]
    assert stored(site)["pair_clips"] == pair_clips == site.fingerprints()["pair_clips"]


def test_pairs_without_complete_shards(make_site):
    """Q_c and Q_l without complete shards: no pair; pair_labels = SHA-256 of no bytes."""
    site = make_site(world=World(HAND.clips, [], []), validation="0")
    r = site.run()
    assert r.code == 0, r.err
    assert stored(site)["pair_labels"] == sha(b"")
    assert site.tree() == site.expected_tree()


FREE = {  # C's run.json: changes outside the clip-defining fields
    "config.workers": ("config.workers", 9),
    "config.out_dir": ("config.out_dir", "elsewhere"),
    "counts": ("clips", 99),
    "episodes": ("episodes", {"null": 1}),
    "an extra key": ("extra", [1, 2]),
}


@pytest.mark.parametrize("change", FREE.values(), ids=FREE)
def test_other_keys_keep_the_clips_fingerprint(make_site, change):
    """A complete run, then a change of another key of C's run.json: the rerun passes
    (R'), `sha256` unchanged."""
    site = make_site(**NO_PAIRS)
    assert site.run().code == 0
    before = stored(site)
    edit(site.repo / "w/clips/run.json", setting(*change))
    r = site.run()
    assert r.code == 0, r.err
    assert stored(site) == before


def test_other_keys_keep_the_pair_clips_fingerprint(make_site):
    """An extra key and other counts in Q_c's run.json: Q_l still labels these clips (R')
    and the rerun passes."""
    site = make_site()
    assert site.run().code == 0
    before = stored(site)
    edit(site.repo / "w/pair_clips/run.json", setting("producer", "preference-pool"))
    edit(site.repo / "w/pair_clips/run.json", setting("clips", 1))
    r = site.run()
    assert r.code == 0, r.err
    assert stored(site) == before


NEW = {
    "config.width": 641,
    "config.height": 361,
    "config.states": 17,
    "classifier": "d:new",
} | {f: sha(b"new") for f in CLIP_FIELDS[3:9]}
CHANGED = {f: (f, NEW[f]) for f in CLIP_FIELDS} | {
    "width as 640.0": ("config.width", 640.0)
}


@pytest.mark.parametrize("change", CHANGED.values(), ids=CHANGED)
def test_clip_defining_field_changes_the_fingerprint(make_site, change):
    """A complete run, then a clip-defining field of C's run.json changed (also its JSON
    text alone): the rerun aborts with `clips changed`."""
    site = make_site(**NO_PAIRS)
    assert site.run().code == 0
    edit(site.repo / "w/clips/run.json", setting(*change))
    aborted(site.run(), "clips changed since the run started; delete w/index")


def test_new_complete_shard_changes_the_fingerprint(make_site):
    """A shard of C completed since the run started: `clips changed`."""
    site = make_site(**NO_PAIRS)
    assert site.run().code == 0
    write_json(site.repo / "w/clips/shards/000002/records.json", [record("6.16/n.mp4")])
    aborted(site.run(), "clips changed since the run started; delete w/index")


@pytest.mark.parametrize("staged", ["w/clips/shards/000002.part/records.json",
                                    "w/pair_clips/shards/000002.part/records.json",
                                    "w/pair_labels/shards/000002.part/pairs.json"])  # fmt: skip
def test_staged_shards_left_out(make_site, staged):
    """A staged shard's file changed since the run started: no fingerprint changes."""
    site = make_site()
    assert site.run().code == 0
    before = stored(site)
    path = site.repo / staged
    path.write_bytes(json.dumps([]).encode())
    r = site.run()
    assert r.code == 0, r.err
    assert stored(site) == before
