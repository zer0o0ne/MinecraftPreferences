"""`main` of src/mcpref/gui/labels.py against specs/gui-labels.md: reruns continuing at label
shard h, `no_answer` final, (4') and run.json on every run, the bar's `initial` and
`label: already done`; fake ask_many, fake plan shards and frames (conftest.py, labels_world.py)."""

import json
import shutil

import pytest
from labels_world import (
    ENDS,
    LABELS,
    NAMES,
    SHARDS,
    calls,
    check_bar,
    label_bar,
    ok_of,
    summary,
)
from progress_bars import assert_aborted


@pytest.mark.parametrize("j", [1, 2, 3])
def test_interrupted_run_calls_equal_uninterrupted(lab, j):
    """ask_many raising on call j, then a rerun: the raising call repeated, the others once;
    the label shards as uninterrupted."""
    lab.fake.fail_on = j
    assert_aborted(*lab.run())
    lab.fake.fail_on = None
    code, err = lab.run()
    assert code == 0, err
    assert lab.asked() == calls()[:j] + calls()[j - 1 :]
    assert [lab.shard(k) for k in range(5)] == [lab.label_shard(k) for k in range(5)]
    assert lab.names() == NAMES


@pytest.mark.parametrize("h", range(len(SHARDS)))
def test_continuation_at_every_shard_boundary(lab, h):
    """Label shards h, ... gone (as after an interruption there): the rerun asks exactly the
    calls of plan shards h, ... with their seeds (positions continue from `end` of shard h - 1);
    the bar starts at the records of shards 0, ..., h - 1."""
    assert lab.run()[0] == 0
    for k in range(h, len(SHARDS)):
        shutil.rmtree(lab.root / "out" / "labels" / NAMES[k])
    code, err = lab.run()
    assert code == 0, err
    assert lab.asked(first=3) == calls(first=h)
    assert [lab.shard(k)["end"] for k in range(5)] == ENDS
    check_bar(err, sum(len(ok_of(k)) for k in range(h)))


def test_interruption_then_no_answer_never_asked_again(lab):
    """Shard 3 holds the no_answer frame 7; the call of shard 4 raises; the rerun asks shard 4
    only and keeps shards 0-3 as they are; a further rerun asks nothing."""
    lab.fake.fail_on = 3
    assert assert_aborted(*lab.run()) == [label_bar(6)]
    assert lab.names() == NAMES[:4]
    assert lab.shard(3) == lab.label_shard(3)
    assert lab.shard(3)["labels"][3]["label"] == "no_answer"
    before = {k: v for k, v in lab.outputs().items() if k.startswith("labels/")}
    lab.fake.fail_on = None
    code, err = lab.run()
    assert code == 0, err
    assert lab.asked(first=3) == calls(first=4) == [([8], [13])]
    check_bar(err, 6)
    assert all(lab.outputs()[k] == v for k, v in before.items())
    assert [lab.shard(k) for k in range(5)] == [lab.label_shard(k) for k in range(5)]
    assert lab.run() == (0, "label: already done\n") and len(lab.fake.calls) == 4


def test_finished_rerun_asks_nothing_and_is_already_done(lab):
    """h = H: no ask, outputs unchanged, no bar but the line `label: already done`."""
    assert lab.run()[0] == 0
    before = lab.outputs()
    assert lab.run() == (0, "label: already done\n") and lab.out == ""
    assert len(lab.fake.calls) == 3 and lab.outputs() == before


def test_summary_only_at_the_end_of_a_run_exiting_0(lab):
    """(4'): none after an interruption; then over every label shard, those of the
    interrupted run included."""
    lab.fake.fail_on = 2
    assert_aborted(*lab.run())
    assert not (lab.root / "out" / "summary.json").exists()
    lab.fake.fail_on = None
    assert lab.run()[0] == 0
    assert lab.read("summary.json") == summary(LABELS, 1, 1, 2, 1)


def test_summary_recomputed_from_the_label_shards_on_every_run(lab):
    """(4'), (5): a run with nothing left to label recomputes summary.json from the label
    records, g_f as stored there (frame 8's record edited: inventory, flag false)."""
    assert lab.run()[0] == 0
    path = lab.root / "out" / "labels" / NAMES[4] / "labels.json"
    shard = json.loads(path.read_text())
    shard["labels"][0].update(label="inventory", gui_inventory=False)
    path.write_text(json.dumps(shard))
    (lab.root / "out" / "summary.json").write_text("{}")
    assert lab.run() == (0, "label: already done\n")
    labels = [*LABELS[:-1], "inventory"]
    assert lab.read("summary.json") == summary(labels, 1, 2, 1, 1)


@pytest.mark.parametrize("finished", [False, True])
def test_run_json_rewritten_at_the_start_of_every_run(lab, finished):
    """Written by specs/shards.md (4): replaced (a handle opened before reads the old text),
    before the first call of the rerun."""
    lab.fake.fail_on = None if finished else 2
    lab.run()
    lab.fake.fail_on = None
    path = lab.root / "out" / "run.json"
    original = path.read_text()
    path.write_text(json.dumps({**json.loads(original), "stale": 1}))
    with path.open() as handle:
        stale = handle.read()
        assert lab.run()[0] == 0
        handle.seek(0)
        assert handle.read() == stale
    assert path.read_text() == original
    assert all(c.run_json == original for c in lab.fake.calls)


def test_run_json_written_before_an_interruption(lab):
    """The first call raises: run.json is in place, no label shard is."""
    lab.fake.fail_on = 1
    assert_aborted(*lab.run())
    assert lab.fake.calls[0].run_json is not None
    assert lab.read("run.json")["frames"] == {"shards": 5, "ok": 7, "failed": 3}
    assert [n for n in lab.names() if not n.endswith(".part")] == []
