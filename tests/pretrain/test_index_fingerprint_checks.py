"""(R') of specs/train-index.md, the checks after the pass of (I) (its bar drawn): `labels
other clips`, then `check_fingerprints` in the order config, clips, pair_clips, pair_labels
(null without pairs); {out_dir} unchanged by an abort."""

import pytest

from .index_oracle import sha
from .index_run import INPUTS, aborted, interrupting
from .index_world import NO_PAIRS, edit, setting


def test_labels_other_clips(make_site):
    """(R'): pair_clips of Q_l other than the fingerprint (4) of Q_c: the abort after the
    pass of (I) (its bar drawn); nothing written to {out_dir}."""
    site = make_site()
    edit(site.repo / "w/pair_labels/run.json", setting("pair_clips", sha(b"other")))
    above = aborted(site.run(), "w/pair_labels labels other clips than w/pair_clips")
    assert above == [INPUTS]
    assert not site.out.exists()


def test_labels_other_clips_than_their_fingerprint(make_site):
    """(R'): Q_c's records changed after Q_l labelled it (a byte of a records.json): its
    fingerprint differs from pair_clips."""
    site = make_site()
    path = site.repo / "w/pair_clips/shards/000001/records.json"
    path.write_bytes(path.read_bytes() + b" ")
    above = aborted(site.run(), "w/pair_labels labels other clips than w/pair_clips")
    assert above == [INPUTS]


def change_config(site) -> None:
    """The config's bytes, not its mapping (an appended comment)."""
    site.write_config("# changed\n")


def change_clips(site) -> None:
    path = site.repo / "w/clips/shards/000000/records.json"
    path.write_bytes(path.read_bytes() + b"\n")


def change_pair_clips(site) -> None:
    """A byte of a records.json of Q_c; Q_l labels the changed Q_c."""
    path = site.repo / "w/pair_clips/shards/000000/records.json"
    path.write_bytes(path.read_bytes() + b"\n")
    pair_clips = site.fingerprints()["pair_clips"]
    edit(site.repo / "w/pair_labels/run.json", setting("pair_clips", pair_clips))


def change_pair_labels(site) -> None:
    path = site.repo / "w/pair_labels/shards/000001/pairs.json"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\n "))


CHANGES = {
    "config": change_config,
    "clips": change_clips,
    "pair_clips": change_pair_clips,
    "pair_labels": change_pair_labels,
}
CHANGED = {  # changes applied together; the first name in the order reported
    "config": ["config"],
    "clips": ["clips"],
    "pair_clips": ["pair_clips"],
    "pair_labels": ["pair_labels"],
    "config, pair_labels": ["pair_labels", "config"],
    "clips, pair_clips": ["pair_clips", "clips"],
    "pair_clips, pair_labels": ["pair_labels", "pair_clips"],
}


@pytest.mark.parametrize("changed", CHANGED.values(), ids=CHANGED)
def test_changed_since_the_run_started(make_site, changed):
    """(R'): a complete run, then changes: `check_fingerprints` names the first of config,
    clips, pair_clips, pair_labels; after the pass of (I); {out_dir} unchanged."""
    site = make_site()
    assert site.run().code == 0
    before = site.tree()
    for name in changed:
        CHANGES[name](site)
    first = min(changed, key=list(CHANGES).index)
    above = aborted(
        site.run(), f"{first} changed since the run started; delete w/index"
    )
    assert above == [INPUTS]
    assert site.tree() == before


def test_changed_after_an_interrupted_run(make_site):
    """(R') against the run.json of an interrupted run (no `complete`)."""
    site = make_site()
    assert site.run(interrupting("index shards", 1)).code == 1
    before = site.tree()
    change_clips(site)
    aborted(site.run(), "clips changed since the run started; delete w/index")
    assert site.tree() == before


def test_changed_without_pairs(make_site):
    """Without pairs (pair_clips, pair_labels null): an unchanged rerun passes, a change
    of C is named; out_dir as written."""
    site = make_site(**{**NO_PAIRS, "out_dir": "w/index/"})
    assert site.run().code == 0
    assert site.run().code == 0
    change_clips(site)
    aborted(site.run(), "clips changed since the run started; delete w/index/")


def test_labels_checked_before_the_fingerprints(make_site):
    """(R'): `labels other clips` before `check_fingerprints` (here: config changed)."""
    site = make_site()
    assert site.run().code == 0
    before = site.tree()
    change_config(site)
    edit(site.repo / "w/pair_labels/run.json", setting("pair_clips", sha(b"other")))
    aborted(site.run(), "w/pair_labels labels other clips than w/pair_clips")
    assert site.tree() == before
