"""Progress bars of specs/acyclic.md (Interface), drawn by `shard_bar`: `correction inputs`
(unit shard, total K, advanced after each shard) and `correction shards` (unit comparison,
total `counts.comparisons` of V_l, initial the complete shards' counts, advanced once a
shard is written); their idle lines; nothing on stdout."""

import tqdm
from pretrain.index_run import interrupting
from progress_bars import Bar, assert_progress, screen, states

from .acyclic_site import make_site
from .acyclic_world import HAND, MB, comparison, votes_run

INPUTS = Bar("correction inputs", 4, 4, "shard")
SHARDS = Bar("correction shards", 14, 14, "comparison")  # HAND: 8 + 0 + 1 + 5


def test_bars_of_a_run(tmp_path):
    """Both bars, from 0 to their totals, each closed on a line of its own."""
    r = make_site(tmp_path).run()
    assert (r.code, r.stdout) == (0, "")
    assert_progress(r.err)
    assert screen(r.err) == [INPUTS, SHARDS]
    assert states(r.err, INPUTS.desc)[0] == Bar(INPUTS.desc, 0, 4, "shard")
    assert states(r.err, SHARDS.desc)[0] == Bar(SHARDS.desc, 0, 14, "comparison")


def test_shards_total_from_the_votes_run(tmp_path):
    """The total of `correction shards` is `counts.comparisons` of {V_l}/run.json (here
    written apart from the shards' 14 comparisons to tell the two sources apart)."""
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
    """`correction inputs` advanced by one per shard; `correction shards` shows at every
    update the comparisons of the shards already written."""
    site = make_site(tmp_path)
    log = []
    assert site.run(recording(log, site)).code == 0
    assert [n for desc, n, _ in log if desc == INPUTS.desc] == [1, 2, 3, 4]
    written = [0, 8, 8, 9, 14]  # comparisons of shards 0..j-1, j complete
    shards = [(n, done) for desc, n, done in log if desc == SHARDS.desc]
    assert shards and all(n == written[done] for n, done in shards), shards
    assert shards[-1] == (14, 4)


def test_nothing_to_do_without_shards(tmp_path):
    """K = 0: two lines, no bar."""
    r = make_site(tmp_path, []).run()
    assert r.code == 0
    assert screen(r.err) == [
        "correction inputs: nothing to do",
        "correction shards: nothing to do",
    ]


def test_already_done(tmp_path):
    """Every shard complete: the inputs bar, then `correction shards: already done`."""
    site = make_site(tmp_path)
    assert site.run().code == 0
    r = site.run()
    assert screen(r.err) == [INPUTS, "correction shards: already done"]


def test_nothing_to_do_when_the_shards_left_have_no_comparison(tmp_path):
    """The shards left hold no comparison: `correction shards: nothing to do`, and they are
    written."""
    world = [[comparison("a_1", "b_1", MB, MB)], [], []]
    site = make_site(tmp_path, world)
    assert site.run(interrupting("correction shards", 1)).code == 1
    r = site.run()
    assert r.code == 0
    inputs = Bar("correction inputs", 3, 3, "shard")
    assert screen(r.err) == [inputs, "correction shards: nothing to do"]
    assert site.tree() == site.expected_tree()


def test_nothing_to_do_in_shards_without_comparisons(tmp_path):
    """Shards without comparisons only: `correction shards: nothing to do`; both written."""
    site = make_site(tmp_path, [[], []])
    r = site.run()
    assert screen(r.err) == [Bar("correction inputs", 2, 2, "shard"),
                             "correction shards: nothing to do"]  # fmt: skip
    assert site.tree() == site.expected_tree()
