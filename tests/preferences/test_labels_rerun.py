"""`main` of specs/vlm-labels.md over runs (Behaviour (I), (L); Interface: Datasets,
Progress bars): runs interrupted and continued (equal calls, `no_answer` final, `initial`),
a finished run rerun, the aborts of `check_fingerprints` in order."""

import json

import pytest
from pretrain.index_run import interrupting
from progress_bars import assert_aborted, screen, states

from .labels_oracle import inputs_bar, label_bar, same_prompts
from .labels_run import PREFIX
from .labels_world import HAND, HAND_ENDS, HAND_LABELS, HAND_TOTAL


def hand_tree(site) -> dict:
    return site.expected_tree(HAND_LABELS, HAND_TOTAL, HAND_ENDS)


def shards_of(tree: dict, n: int) -> dict:
    """The files of label shards 0..n-1 of a tree."""
    return {k: v for k, v in tree.items() if "shards/" <= k[:13] < f"shards/{n:06d}"}


def inodes(site) -> dict:
    return {p: (site.out / p).stat().st_ino for p in site.tree() if p != "run.json"}


def test_interrupted_by_ask_many_and_continued(make_site):
    """ask_many raising in shard 2's call: exit 1 with its message; label shards 0 and 1
    kept, run.json of (I); the rerun asks from shard 2 on (shard 0, with its `no_answer`,
    never again) with `initial` 3, and ends as an uninterrupted run."""
    site = make_site()
    site.fake.fail_on = 2
    first = site.run()
    above = assert_aborted(first.code, first.err)
    assert first.err.splitlines()[-1] == f"{PREFIX}RuntimeError: HTTP 400: rejected"
    assert above == [inputs_bar(4), label_bar(3, 5)]
    expected = hand_tree(site)
    assert site.tree() == {"run.json": site.expected_run(), **shards_of(expected, 2)}
    kept = inodes(site)
    again = site.run()
    assert again.code == 0, again.err
    assert screen(again.err) == [inputs_bar(4), label_bar(5, 5)]
    assert states(again.err, "label")[0] == label_bar(3, 5)
    calls = site.fake.calls[:1] + site.fake.calls[2:]
    asked = [(c.seeds, c.prompts) for c in calls]
    wanted = site.expected_calls()
    assert [s for s, _ in asked] == [s for _, s in wanted]
    assert all(same_prompts(a, p) for (_, a), (p, _) in zip(asked, wanted))
    assert site.fake.calls[1].seeds == site.fake.calls[2].seeds
    assert site.tree() == expected
    assert {p: i for p, i in inodes(site).items() if p in kept} == kept


def test_interrupted_after_the_last_shard(make_site):
    """Interrupted after the last label shard is written (a raise in the bar `label`):
    run.json of (I); the rerun asks nothing (`label: already done`) and marks it
    complete."""
    site = make_site()
    first = site.run(interrupting("label", 5))
    assert assert_aborted(first.code, first.err) == [inputs_bar(4), label_bar(5, 5)]
    assert site.tree() == {**hand_tree(site), "run.json": site.expected_run()}
    again = site.run()
    assert again.code == 0, again.err
    assert screen(again.err) == [inputs_bar(4), "label: already done"]
    assert len(site.fake.calls) == 3 and site.tree() == hand_tree(site)


def test_rest_without_comparisons(make_site):
    """Interrupted after shard 0 with only shard 1 (no comparison) left: the rerun draws
    `label: nothing to do`, asks nothing and writes label shard 1 (`end` 12)."""
    site = make_site(shards=HAND[:2])
    assert site.run(interrupting("label", 3)).code == 1
    again = site.run()
    assert again.code == 0, again.err
    assert screen(again.err) == [inputs_bar(2), "label: nothing to do"]
    assert len(site.fake.calls) == 1
    total = HAND_LABELS[0]
    assert site.tree() == site.expected_tree(HAND_LABELS[:2], total, HAND_ENDS[:2])


def test_rerun_of_a_finished_run(make_site):
    """Every label shard complete: the inputs read again, no call, `label: already done`,
    the shards kept, run.json rewritten to the same bytes."""
    site = make_site()
    assert site.run().code == 0
    kept = inodes(site)
    again = site.run()
    assert again.code == 0, again.err
    assert screen(again.err) == [inputs_bar(4), "label: already done"]
    assert len(site.fake.calls) == 3
    assert site.tree() == hand_tree(site) and inodes(site) == kept


def test_leftover_staged_label_shard(make_site):
    """A staged label shard left over: the rerun writes the shard anew."""
    site = make_site()
    site.fake.fail_on = 2
    assert site.run().code == 1
    (site.out / "shards/000002.part").mkdir()
    (site.out / "shards/000002.part/answers.json").write_text("{}")
    assert site.run().code == 0
    assert site.tree() == hand_tree(site)


def edit_json(path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, indent=2) + "\n")


def append(path, text: str):
    return lambda: path.write_text(path.read_text() + text)


def changes(site) -> dict:
    """A change of each input, by fingerprint name."""
    pool = site.at("pool_dir")
    records = pool / "shards/000002/records.json"
    cmp = pool / "shards/000003/comparisons.json"
    width = pool / "run.json"
    return {
        "config": append(site.path, "# edited\n"),
        "vlm": append(site.at("vlm"), "# edited\n"),
        "pool": append(records, " "),
        "pool.width": lambda: edit_json(width, lambda r: r["config"].update(width=8)),
        "classifier": lambda: edit_json(width, lambda r: r.update(classifier="d:1")),
        "comparisons": lambda: edit_json(
            cmp, lambda c: c[0].update(a=c[0]["b"], b=c[0]["a"])
        ),
    }


@pytest.mark.parametrize(
    "changed, name",
    [
        (["config"], "config"),
        (["vlm"], "vlm"),
        (["pool"], "pool"),
        (["pool.width"], "pool"),
        (["classifier"], "pool"),
        (["comparisons"], "comparisons"),
        (["config", "vlm", "pool", "comparisons"], "config"),
        (["vlm", "pool", "comparisons"], "vlm"),
        (["pool", "comparisons"], "pool"),
    ],
)
def test_fingerprint_aborts_in_order(make_site, changed, name):
    """(I): an existing run.json checked after the pass over the inputs (bar drawn), in
    the order config, vlm, pool, comparisons; the first that differs aborts with
    `{name} changed since the run started; delete {out_dir}`; nothing rewritten."""
    site = make_site()
    site.fake.fail_on = 2
    assert site.run().code == 1
    before = site.tree()
    for key in changed:
        changes(site)[key]()
    r = site.run()
    assert assert_aborted(r.code, r.err) == [inputs_bar(4)]
    message = f"{name} changed since the run started; delete w/votes"
    assert r.err.splitlines()[-1] == f"{PREFIX}RuntimeError: {message}"
    assert site.tree() == before and len(site.fake.calls) == 2


@pytest.mark.parametrize(
    "where", ["config.workers", "episodes", "staged records", "staged comparisons"]
)
def test_unfingerprinted_changes_continue(make_site, where):
    """A key of Q_c's run.json outside the clip-defining fields, or a staged shard of
    Q_c, changed: the run continues."""
    site = make_site()
    site.fake.fail_on = 2
    assert site.run().code == 1
    pool = site.at("pool_dir")
    staged = pool / "shards/000004.part"
    edit = {
        "config.workers": lambda r: r["config"].update(workers=9),
        "episodes": lambda r: r.update(episodes={"null": 1}),
    }
    if where in edit:
        edit_json(pool / "run.json", edit[where])
    else:
        file = "records.json" if where == "staged records" else "comparisons.json"
        (staged / file).write_text("[]\n")
    assert site.run().code == 0
    assert site.tree() == hand_tree(site)
