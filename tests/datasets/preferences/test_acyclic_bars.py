"""Progress bars of specs/acyclic.md and block 33 target diagnostics."""

import tqdm
from progress_bars import Bar, assert_bars, screen, states
from training.pretrain.data.index_run import interrupting

from datasets.preferences.acyclic_site import make_site
from datasets.preferences.acyclic_world import HAND, MB, comparison, votes_run

INPUTS = Bar("correction inputs", 4, 4, "shard")
SHARDS = Bar("correction shards", 14, 14, "comparison")
TARGETS = Bar("diagnostics targets", 4, 4, "shard")


def test_bars_of_a_run(tmp_path):
    """Producer and diagnostic bars start at zero and close at their totals."""
    r = make_site(tmp_path).run()
    assert (r.code, r.stdout) == (0, "")
    assert_bars(r.err, [INPUTS, TARGETS, SHARDS])


def test_shards_total_from_the_votes_run(tmp_path):
    """The total of `correction shards` is `counts.comparisons` of {V_l}/run.json."""
    run = votes_run(HAND)
    run["counts"]["comparisons"] = 20
    r = make_site(tmp_path, run=run).run()
    assert states(r.err, SHARDS.desc)[0] == Bar(SHARDS.desc, 0, 20, "comparison")
    assert states(r.err, SHARDS.desc)[-1] == Bar(SHARDS.desc, 14, 20, "comparison")


def recording(log: list, site):
    """A patch of tqdm's `update` logging (desc, n, complete shards of {out_dir})."""
    real = tqdm.tqdm.update

    def update(self, n=1):
        shown = real(self, n)
        d = site.out / "shards"
        done = (
            sum(1 for p in d.iterdir() if not p.name.endswith(".part"))
            if d.exists()
            else 0
        )
        log.append((self.desc, self.n, done))
        return shown

    return lambda mp: mp.setattr(tqdm.tqdm, "update", update)


def test_advanced_per_shard(tmp_path):
    """`correction inputs` and `correction shards` advance after their source events."""
    site = make_site(tmp_path)
    log = []
    assert site.run(recording(log, site)).code == 0
    assert [n for desc, n, _ in log if desc == INPUTS.desc] == [1, 2, 3, 4]
    written = [0, 8, 8, 9, 14]
    shards = [(n, done) for desc, n, done in log if desc == SHARDS.desc]
    assert shards and all(n == written[done] for n, done in shards), shards
    assert shards[-1] == (14, 4)


def test_nothing_to_do_without_shards(tmp_path):
    r = make_site(tmp_path, []).run()
    assert r.code == 0
    assert screen(r.err) == [
        "correction inputs: nothing to do",
        "correction shards: nothing to do",
        "diagnostics targets: nothing to do",
    ]


def test_already_done(tmp_path):
    site = make_site(tmp_path)
    assert site.run().code == 0
    r = site.run()
    assert screen(r.err) == [
        INPUTS,
        "correction shards: already done",
        "diagnostics targets: already done",
    ]


def test_nothing_to_do_when_the_shards_left_have_no_comparison(tmp_path):
    world = [[comparison("a_1", "b_1", MB, MB)], [], []]
    site = make_site(tmp_path, world)
    assert site.run(interrupting("correction shards", 1)).code == 1
    r = site.run()
    assert r.code == 0
    inputs = Bar("correction inputs", 3, 3, "shard")
    assert screen(r.err) == [
        inputs,
        "correction shards: nothing to do",
        Bar("diagnostics targets", 3, 3, "shard"),
    ]
    assert site.tree() == site.expected_tree()


def test_nothing_to_do_in_shards_without_comparisons(tmp_path):
    site = make_site(tmp_path, [[], []])
    r = site.run()
    assert screen(r.err) == [Bar("correction inputs", 2, 2, "shard"),
                             "correction shards: nothing to do",
                             Bar("diagnostics targets", 2, 2, "shard")]  # fmt: skip
    assert site.tree() == site.expected_tree()
