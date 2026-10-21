"""`main` of specs/vlm-labels.md on a fake $Q_c$ with a fake `ask_many` (Behaviour (I), (L);
Interface: Produces, Datasets): one call per shard with its requests, prompts (1) and seeds
(positions across shards), none for a shard without comparisons; the label shards,
`counts.json`, `end`; run.json after (I) and after (L); the shown frames."""

import pytest
from progress_bars import assert_progress, screen, states

from .labels_oracle import SHOWN, inputs_bar, label_bar, same_prompts
from .labels_world import HAND, HAND_ENDS, HAND_LABELS, HAND_TOTAL, Shard, record


def test_one_call_per_shard(make_site):
    """Shards 0, 2, 3 asked, each in one call: its comparisons in order, then (o, i) in
    order, (a, b) for o = 0 and (b, a) for o = 1; seeds `seed` + 2 s m + o s + i with m
    over the shards; the client config of `vlm`; shard 1 (no comparison) not asked."""
    site = make_site()
    r = site.run()
    assert (r.code, r.stdout) == (0, ""), r.err
    calls = site.fake.calls
    assert [c.seeds for c in calls] == [
        list(range(1000, 1012)),
        list(range(1012, 1016)),
        list(range(1016, 1020)),
    ]
    for call, (prompts, seeds) in zip(calls, site.expected_calls(), strict=True):
        assert call.seeds == seeds and same_prompts(call.prompts, prompts)
        assert call.cfg == site.vlm_config()


def test_label_shards(make_site):
    """{out_dir}: run.json and one label shard per shard of Q_c (the staged one not
    labelled): answers.json per comparison with its requests, counts.json with the labels
    in the order of the tokens, then invalid, no_answer (those that occur) and `end`; a
    shard without comparisons: `[]`, zero comparisons, `end` unchanged."""
    site = make_site()
    assert site.run().code == 0
    assert site.tree() == site.expected_tree(HAND_LABELS, HAND_TOTAL, HAND_ENDS)
    assert site.tree()["shards/000001/answers.json"] == b"[]\n"


def test_run_json_after_inputs(make_site):
    """(I): run.json = {config as written (file order kept), sha256: config, vlm, pool,
    comparisons}, present at the first call, written before any label shard."""
    site = make_site()
    assert site.run().code == 0
    first = site.fake.calls[0]
    assert (first.run_json, first.shards) == (site.expected_run(), [])
    assert [c.shards for c in site.fake.calls[1:]] == [
        ["000000", "000001"],
        ["000000", "000001", "000002"],
    ]


def test_run_json_after_labels(make_site):
    """(L): `complete` true, then `counts` = {comparisons, labels} summed over the label
    shards, labels in the key order of (L)."""
    site = make_site()
    assert site.run().code == 0
    assert (site.out / "run.json").read_bytes() == site.expected_run(HAND_TOTAL)


def test_bars(make_site):
    """`label inputs` over 2 K shards, then `label` over the comparisons of Q_c's
    run.json, advanced by each shard's comparisons once written; nothing on stdout."""
    site = make_site()
    r = site.run()
    assert (r.code, r.stdout) == (0, ""), r.err
    assert_progress(r.err)
    assert screen(r.err) == [inputs_bar(4), label_bar(5, 5)]
    assert {b.n for b in states(r.err, "label")} <= {0, 3, 4, 5}


@pytest.mark.parametrize("frames, stride", list(SHOWN))
def test_shown_frames(make_site, frames, stride):
    """Notation: j_q = L - 1 - (k_V - 1 - q) s_V, oldest first; (k_V - 1) s_V = L - 1
    included (2, 5) and all frames (6, 1); one sample per order."""
    site = make_site(frames=frames, stride=stride, samples=1)
    assert site.run().code == 0
    expected = site.expected_calls()
    assert [c.seeds for c in site.fake.calls] == [s for _, s in expected]
    assert all(
        same_prompts(c.prompts, p) for c, (p, _) in zip(site.fake.calls, expected)
    )


def test_pool_without_shards(make_site):
    """K = 0 (`clips` = 0 holds the marker): both bars `nothing to do`, no call, run.json
    complete with zero counts."""
    site = make_site(shards=[])
    r = site.run()
    assert r.code == 0, r.err
    assert screen(r.err) == ["label inputs: nothing to do", "label: nothing to do"]
    assert site.fake.calls == []
    assert site.tree() == {"run.json": site.expected_run({})}


def test_pool_without_comparisons(make_site):
    """K = 2 shards without comparisons: `label: nothing to do`, no call, both label shards
    written with `end` 0."""
    shards = [Shard([record("data/9.0/a.mp4", ())], []), Shard([record("b.mp4")], [])]
    site = make_site(shards=shards)
    r = site.run()
    assert r.code == 0, r.err
    assert screen(r.err) == [inputs_bar(2), "label: nothing to do"]
    assert site.fake.calls == []
    assert site.tree() == site.expected_tree([{}, {}], {}, [0, 0])


def test_seed_at_its_bound(make_site):
    """Seeds `seed` + position with `seed` = 2^32 - 1."""
    site = make_site(seed=2**32 - 1, samples=1, shards=HAND[2:])
    assert site.run().code == 0
    assert [c.seeds for c in site.fake.calls] == [
        [2**32 - 1, 2**32],
        [2**32 + 1, 2**32 + 2],
    ]
