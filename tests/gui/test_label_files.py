"""`main` of src/mcpref/gui/labels.py against specs/gui-labels.md on a fresh {out_dir}: (2),
(4) calls and label shards, (4'), (5), run.json, the bar `label`, aborts before labelling;
fake ask_many, fake plan shards and frames (conftest.py, labels_world.py). Reruns:
test_label_files_rerun.py, test_label_files_checks.py."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from labels_world import (
    ENDS,
    FINGERPRINT,
    HASHED,
    LABELS,
    NAMES,
    QUESTION,
    TEXTS,
    calls,
    check_bar,
    label_bar,
    summary,
)
from progress_bars import assert_aborted

from mcpref.vlm.client import load_config as load_vlm_config


def test_one_call_per_plan_shard_with_its_ok_frames_and_seeds(lab):
    """(2), (4): plan shards 0, 3, 4 have ok frames; 1 (no frame) and 2 (failed only) none;
    seed = seed + position over all shards, failed frames included."""
    code, err = lab.run()
    assert code == 0, err
    assert lab.asked() == [
        ([1, 2], [6, 7]),
        ([4, 5, 6, 7], [9, 10, 11, 12]),
        ([8], [13]),
    ]
    assert lab.asked() == calls()
    assert all(type(s) is int for c in lab.fake.calls for s in c.seeds)


def test_prompts_and_vlm_config(lab):
    """(1) per frame: [instruction, its RGB view (the fake knows each frame's), question]."""
    assert lab.run()[0] == 0
    prompts = [q for c in lab.fake.calls for q in c.prompts]
    assert all(len(q) == 3 and q[0] == TEXTS["instruction"] for q in prompts)
    assert all(q[2] == QUESTION for q in prompts)
    vlm = load_vlm_config(lab.root / "vlm.yaml")[0]
    assert all(c.cfg == vlm for c in lab.fake.calls)


def test_label_shards_contents_and_end(lab):
    """One label shard per plan shard, file labels.json: records of its ok frames in order
    (g_f from `line`), `end` after its last frame (unchanged after a shard without frames)."""
    assert lab.run()[0] == 0
    assert lab.names() == NAMES
    assert [lab.shard(k) for k in range(5)] == [lab.label_shard(k) for k in range(5)]
    assert [lab.shard(k)["end"] for k in range(5)] == ENDS == [3, 3, 4, 8, 10]
    assert lab.shard(1) == {"labels": [], "end": 3} and lab.shard(2)["labels"] == []
    for name in NAMES:
        assert os.listdir(lab.root / "out" / "labels" / name) == ["labels.json"]
    assert sorted(os.listdir(lab.root / "out")) == [
        "labels",
        "run.json",
        "summary.json",
    ]


def test_each_label_shard_written_after_its_call(lab):
    """At each call: the label shards of the earlier plan shards, nothing of its own."""
    assert lab.run()[0] == 0
    assert [c.labels for c in lab.fake.calls] == [[], NAMES[:3], NAMES[:4]]


def test_summary_and_vpt_check(lab):
    """(5) over class labels only: inventory (g T), inventory_book (F); other: chest3 (T),
    other (F), other (T); invalid and no_answer excluded."""
    assert lab.run()[0] == 0
    assert lab.read("summary.json") == summary(LABELS, 1, 1, 2, 1)


def test_run_json_contents(lab):
    assert lab.run()[0] == 0
    run = lab.read("run.json")
    assert set(run) == {"config", "sha256", "frames", "prompt"}
    assert run["config"] == lab.cfg
    files = {k: (lab.root / v).read_bytes() for k, v in HASHED.items()}
    assert run["sha256"] == {k: hashlib.sha256(b).hexdigest() for k, b in files.items()}
    assert run["frames"] == FINGERPRINT
    assert run["prompt"] == [TEXTS["instruction"], "<image>", QUESTION]


def test_run_json_written_before_the_first_call(lab):
    assert lab.run()[0] == 0
    final = (lab.root / "out" / "run.json").read_text()
    assert [c.run_json for c in lab.fake.calls] == [final] * 3


def test_failed_count_zero_when_absent(lab):
    """Frames' run.json without `failed` (none occurred): fingerprint failed = 0; a rerun
    finds it equal."""
    lab.frames(frames=[("ok", True), ("ok", False)], shards=[[0], [1]])
    assert lab.run()[0] == 0
    assert lab.read("run.json")["frames"] == {"shards": 2, "ok": 2, "failed": 0}
    assert lab.run() == (0, "label: already done\n")


def test_bar_total_and_final_count(lab):
    """Bar `label` on stderr only, from 0 to its total, the ok count of the frames' run.json;
    nothing on stdout."""
    code, err = lab.run()
    assert code == 0 and lab.out == ""
    check_bar(err, 0)


@pytest.mark.parametrize("j, written", [(1, 0), (2, 2), (3, 6)])
def test_bar_closed_at_the_shards_written(lab, j, written):
    """Advanced by a shard's records once written: closed at the records of the shards before
    the shard whose call raises (a state drawn whatever the timing), the message below it."""
    lab.fake.fail_on = j
    assert assert_aborted(*lab.run()) == [label_bar(written)]


def test_abort_while_frames_lack_counts(lab):
    """{frames_dir}/run.json as at the start of a vpt-gui-frames run: frames not complete."""
    lab.frames(counts=False)
    assert_aborted(*lab.run())
    assert lab.fake.calls == [] and not (lab.root / "out").exists()


@pytest.mark.parametrize("case", ["failed only", "no frame", "ok count 0"])
def test_abort_without_ok_frame(lab, case):
    frames = [("failed", True), ("failed", False)] if case != "no frame" else []
    lab.frames(frames=frames, shards=[[0], [1]] if frames else [])
    if case == "ok count 0":
        run = json.loads((lab.root / "frames/run.json").read_text())
        run["frames"]["ok"] = 0
        (lab.root / "frames/run.json").write_text(json.dumps(run))
    assert_aborted(*lab.run())
    assert lab.fake.calls == [] and not (lab.root / "out").exists()


UNREADABLE = ["gui-labels.yaml", "gui-screens.yaml", "vlm.yaml", "frames/run.json"]


@pytest.mark.parametrize("missing", UNREADABLE)
def test_abort_on_unreadable_file(lab, missing):
    (lab.root / missing).unlink()
    assert assert_aborted(*lab.run()) == []
    assert lab.fake.calls == [] and not (lab.root / "out/run.json").exists()


def test_abort_on_unreadable_plan_shard_after_the_shards_before(lab):
    """Plan shard k is read in the loop: shards 0-2 are labelled before shard 3 fails."""
    (lab.root / "frames/plan/000003/shard.json").unlink()
    assert assert_aborted(*lab.run()) == [label_bar(2)]
    assert [n for n in lab.names() if not n.endswith(".part")] == NAMES[:3]
    assert lab.asked() == calls()[:1]


def test_abort_on_foreign_entry_in_plan(lab):
    """complete_shards of {frames_dir}/plan raises."""
    (lab.root / "frames/plan/notes").write_text("")
    assert_aborted(*lab.run())
    assert lab.fake.calls == []


def test_runs_as_module(lab):
    """`python -m mcpref.gui.labels <config path>`, here on frames without an ok one."""
    lab.frames(frames=[("failed", True)], shards=[[0]])
    argv = [sys.executable, "-m", "mcpref.gui.labels", lab.root / "gui-labels.yaml"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=8, check=False)
    assert_aborted(proc.returncode, proc.stderr)
    assert not (lab.root / "out").exists()
