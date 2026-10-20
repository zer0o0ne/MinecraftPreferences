"""`main` of specs/train-index.md over runs (Behaviour (R'), (W); Interface: Datasets,
Progress bars): run.json after (R') and after (W); runs interrupted inside a bar and
continued (byte-identical outputs, complete shards kept, `initial`); a finished run rerun;
the bars and their lines."""

import itertools

import pytest
from progress_bars import Bar, assert_aborted, assert_progress, screen, states

from .index_oracle import PARTS, name
from .index_run import INPUTS, interrupting
from .index_world import NO_PAIRS, World, random_world, record

RANDOM = {"seed": 0, "validation": "0.3", "items_per_shard": 4}  # with random_world(0)


def inodes(out) -> dict:
    """items.json of each complete shard -> its inode."""
    return {
        f"{part}/{p.parent.name}": p.stat().st_ino
        for part in PARTS
        for p in sorted((out / part).glob("[0-9]*/items.json"))
    }


def inputs(site) -> Bar:
    """The bar `index inputs` at its end: the complete shards of C, Q_c and Q_l."""
    w = site.world
    n = len(w.clips) + len(w.pair_clips) + len(w.pairs)
    return Bar("index inputs", n, n, "shard")


def cuts(site) -> list[int]:
    """The bar `index shards` after each shard: train's shards, then validation's."""
    sizes = [len(s) for part in site.expected().shards for s in part]
    return list(itertools.accumulate(sizes))


def test_bars_of_a_run(make_site):
    """`index inputs` over the complete shards of C, Q_c, Q_l (staged ones not counted),
    then `index shards` over N_0 + N_1 items, advanced by each shard's items; no stdout."""
    site = make_site()
    r = site.run()
    assert r.code == 0 and r.stdout == "", r.err
    assert_progress(r.err)
    assert screen(r.err) == [INPUTS, Bar("index shards", 7, 7, "item")]
    assert {b.n for b in states(r.err, "index shards")} <= {0, *cuts(site)}


def test_bars_without_pairs(make_site):
    """Without pairs: `index inputs` over C's complete shards alone."""
    site = make_site(**{**NO_PAIRS, "items_per_shard": 2})
    r = site.run()
    assert r.code == 0, r.err
    expected = [Bar("index inputs", 2, 2, "shard"), Bar("index shards", 4, 4, "item")]
    assert screen(r.err) == expected


def test_run_json_after_the_checks(make_site):
    """(R'): run.json written before the first shard, without `complete` and `counts`."""
    site = make_site()
    r = site.run(interrupting("index shards", 1))
    assert assert_aborted(r.code, r.err) == [INPUTS, Bar("index shards", 3, 7, "item")]
    assert (site.out / "run.json").read_bytes() == site.expected_run(complete=False)


def test_rerun_of_a_finished_run(make_site):
    """Every shard complete: the inputs read again, `index shards: already done`, the
    shards kept (not rewritten), run.json rewritten to the same bytes."""
    site = make_site()
    assert site.run().code == 0
    before, ino = site.tree(), inodes(site.out)
    r = site.run()
    assert r.code == 0, r.err
    assert screen(r.err) == [INPUTS, "index shards: already done"]
    assert site.tree() == before == site.expected_tree()
    assert inodes(site.out) == ino


@pytest.mark.parametrize("at", ["first train shard", "train", "first validation shard"])
def test_interrupted_and_continued(make_site, at):
    """Interrupted after some shards (a raise inside `index shards`): the complete shards
    are those of an uninterrupted run; the rerun keeps them, draws its bar from their
    items (`initial`) and ends byte-identical to an uninterrupted run."""
    site = make_site(world=random_world(0), **RANDOM)
    train, validation = site.expected().shards
    assert len(train) > 1 and len(validation) > 1
    done = {"first train shard": 1, "train": len(train),
            "first validation shard": len(train) + 1}[at]  # fmt: skip
    total, start = cuts(site)[-1], cuts(site)[done - 1]
    first = site.run(interrupting("index shards", start))
    assert assert_aborted(first.code, first.err) == [
        inputs(site),
        Bar("index shards", start, total, "item"),
    ]
    complete = {k: v for k, v in site.expected_tree().items() if k != "run.json"}
    kept = dict(itertools.islice(complete.items(), done))
    assert site.tree() == {"run.json": site.expected_run(complete=False), **kept}
    ino = inodes(site.out)
    again = site.run()
    assert again.code == 0, again.err
    assert screen(again.err) == [
        inputs(site),
        Bar("index shards", total, total, "item"),
    ]
    drawn = states(again.err, "index shards")
    assert drawn[0].n == start and all(a.n <= b.n for a, b in itertools.pairwise(drawn))
    assert site.tree() == site.expected_tree()
    assert {k: v for k, v in inodes(site.out).items() if k in ino} == ino


def test_leftover_staged_shard_replaced(make_site):
    """A staged shard left by an interrupted writer: the rerun writes the shard anew."""
    site = make_site(world=random_world(0), **RANDOM)
    total, start = cuts(site)[-1], cuts(site)[0]
    assert site.run(interrupting("index shards", start)).code == 1
    staged = site.out / "train" / (name(1) + ".part")
    staged.mkdir()
    (staged / "items.json").write_text("{}")
    again = site.run()
    assert again.code == 0, again.err
    assert site.tree() == site.expected_tree()
    assert screen(again.err)[-1] == Bar("index shards", total, total, "item")


def test_interrupted_inside_the_inputs(make_site):
    """A raise inside `index inputs`: nothing written to {out_dir}; a rerun writes all."""
    site = make_site()
    r = site.run(interrupting("index inputs", 2))
    assert assert_aborted(r.code, r.err) == [Bar("index inputs", 2, 6, "shard")]
    assert not site.out.exists()
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()


def test_inputs_total(make_site):
    """random_world(0): C 5, Q_c 4, Q_l 3 complete shards: the bar's total is 12."""
    site = make_site(world=random_world(0), **RANDOM)
    r = site.run()
    assert r.code == 0, r.err
    assert screen(r.err)[0] == inputs(site) == Bar("index inputs", 12, 12, "shard")


def test_nothing_to_do(make_site):
    """No complete shard of C (only a staged one) and no pairs: both lines; run.json
    complete with zero counts; the same lines on a rerun."""
    site = make_site(world=World([]), **NO_PAIRS)
    zero = {"clips": 0, "pairs": 0, "episodes": 0}
    for _ in range(2):
        r = site.run()
        assert r.code == 0 and r.stdout == "", r.err
        assert r.err == "index inputs: nothing to do\nindex shards: nothing to do\n"
        assert site.tree() == {"run.json": site.expected_run()}
        assert site.expected().counts == {"train": zero, "validation": zero}


def test_no_item(make_site):
    """C's complete shard holds only records with `clips` null: the inputs bar, then
    `index shards: nothing to do`."""
    site = make_site(world=World([[record("6.13/n.mp4")]]), **NO_PAIRS)
    r = site.run()
    assert r.code == 0, r.err
    assert screen(r.err) == [Bar("index inputs", 1, 1, "shard"), "index shards: nothing to do"]  # fmt: skip
    assert site.tree() == {"run.json": site.expected_run()}
