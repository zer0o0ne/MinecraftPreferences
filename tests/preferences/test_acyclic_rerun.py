"""Datasets of specs/acyclic.md (Interface): a run interrupted inside the shards or the
inputs, then continued, gives byte-identical shards; the complete shards are kept, a
leftover staged shard replaced; a rerun of a complete run changes nothing."""

import pytest
from pretrain.index_run import interrupting
from progress_bars import Bar, states

from .acyclic_site import make_site

# HAND: comparisons per shard 8, 0, 1, 5; the bar reaches 8 after shard 0, 9 after shard 2
SHARDS_DONE = {1: 1, 9: 3}  # interrupted at -> complete shards of {out_dir}
INITIAL = {1: 8, 9: 9}


@pytest.mark.parametrize("at", [1, 9])
def test_interrupted_inside_the_shards(tmp_path, at):
    """The bar raising once a shard is written: the shards written so far are complete and
    final; the rerun starts its bar at their comparisons and ends with every file equal to
    an uninterrupted run's."""
    site = make_site(tmp_path)
    assert site.run(interrupting("correction shards", at)).code == 1
    expected = site.expected_tree()
    kept = {k: v for k, v in site.tree().items() if k != "run.json"}
    shards = {k.split("/")[1] for k in kept}
    assert shards == {f"{k:06d}" for k in range(SHARDS_DONE[at])}
    assert all(expected[k] == v for k, v in kept.items())
    r = site.run()
    assert r.code == 0
    assert states(r.err, "correction shards")[0] == Bar(
        "correction shards", INITIAL[at], 14, "comparison"
    )
    assert site.tree() == expected


def test_interrupted_inside_the_inputs(tmp_path):
    """The inputs bar raising: nothing written to {out_dir}; the rerun completes it."""
    site = make_site(tmp_path)
    assert site.run(interrupting("correction inputs", 2)).code == 1
    assert site.tree() == {}
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()


def test_leftover_staged_shard_replaced(tmp_path):
    """A staged shard left by an interrupted write is replaced by the shard."""
    site = make_site(tmp_path)
    assert site.run(interrupting("correction shards", 1)).code == 1
    staged = site.out / "shards" / "000001.part"
    staged.mkdir()
    (staged / "pairs.json").write_text("half")
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()


def test_rerun_of_a_complete_run(tmp_path):
    """A rerun after a complete run: exit 0, every file unchanged."""
    site = make_site(tmp_path)
    assert site.run().code == 0
    before = site.tree()
    assert site.run().code == 0
    assert site.tree() == before == site.expected_tree()
