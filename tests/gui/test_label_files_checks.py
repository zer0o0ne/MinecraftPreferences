"""`main` of src/mcpref/gui/labels.py against specs/gui-labels.md: the checks of (4) on a rerun
(hashes, fingerprint of the frames, frames complete), a rerun without run.json, and the aborts
of complete_shards on {out_dir}/labels; fake ask_many, fake plan shards and frames
(conftest.py, labels_world.py)."""

import shutil

import pytest
from labels_world import FRAMES, HASHED, calls
from progress_bars import assert_aborted

TURNED_OK = [
    *FRAMES[:3],
    ("ok", *FRAMES[3][1:]),
    *FRAMES[4:],
]  # frame 3 extracted later
CHANGES = {  # fingerprint changes of {frames_dir}: Lab.frames arguments
    "a frame turned ok": {"frames": TURNED_OK},
    "another K, same frames": {
        "shards": [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]],
        "draws": 4,
    },
}


def labelled(lab, finished: bool) -> None:
    """A complete labelling, or one interrupted at the call of plan shard 3."""
    lab.fake.fail_on = None if finished else 2
    lab.run()
    lab.fake.fail_on = None


def changed_message(lab) -> str:
    """The fingerprint abort, paths as written in the config."""
    frames, out = lab.cfg["frames_dir"], lab.cfg["out_dir"]
    return f"{frames} changed since labelling started; delete {out}"


def assert_refused(lab, calls_before: int, outputs_before: dict) -> str:
    """Aborted before any bar, no call, {out_dir} unchanged; returns the message."""
    code, err = lab.run()
    assert assert_aborted(code, err) == []
    assert len(lab.fake.calls) == calls_before and lab.outputs() == outputs_before
    return err[:-1].rsplit("\n", 1)[-1]


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("changed", HASHED.values())
def test_rerun_aborts_when_a_hash_differs(lab, changed, finished):
    labelled(lab, finished)
    before = (len(lab.fake.calls), lab.outputs())
    path = lab.root / changed
    path.write_text(path.read_text() + "# changed\n")
    assert not assert_refused(lab, *before).endswith(changed_message(lab))


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("change", CHANGES)
def test_rerun_aborts_on_a_changed_fingerprint(lab, change, finished):
    labelled(lab, finished)
    before = (len(lab.fake.calls), lab.outputs())
    lab.frames(**CHANGES[change])
    assert assert_refused(lab, *before).endswith(changed_message(lab))


def test_hashes_compared_before_the_fingerprint(lab):
    labelled(lab, True)
    before = (len(lab.fake.calls), lab.outputs())
    lab.frames(frames=TURNED_OK)
    path = lab.root / "vlm.yaml"
    path.write_text(path.read_text() + "# changed\n")
    assert not assert_refused(lab, *before).endswith(changed_message(lab))


@pytest.mark.parametrize("finished", [False, True])
def test_rerun_aborts_while_frames_lack_counts(lab, finished):
    """Frames' run.json as at the start of a vpt-gui-frames rerun: checked before the
    fingerprint (which then differs too)."""
    labelled(lab, finished)
    before = (len(lab.fake.calls), lab.outputs())
    lab.frames(counts=False)
    assert not assert_refused(lab, *before).endswith(changed_message(lab))


def test_no_run_json_but_label_shards_kept_without_check(lab):
    """run.json deleted after an interruption, the config changed: no check, the rerun
    continues at the next shard, keeps the label shards and writes run.json."""
    labelled(lab, False)
    shards = {k: v for k, v in lab.outputs().items() if k.startswith("labels/")}
    (lab.root / "out" / "run.json").unlink()
    path = lab.root / "gui-labels.yaml"
    path.write_text(path.read_text() + "# changed\n")
    code, err = lab.run()
    assert code == 0, err
    assert lab.asked(first=2) == calls()[1:]
    assert all(lab.outputs()[k] == v for k, v in shards.items())
    assert lab.read("run.json")["frames"] == {"shards": 5, "ok": 7, "failed": 3}


@pytest.mark.parametrize("fault", ["gap", "foreign entry"])
def test_abort_when_complete_shards_raises_on_labels(lab, fault):
    labelled(lab, fault == "gap")
    labels = lab.root / "out" / "labels"
    if fault == "gap":
        shutil.rmtree(labels / "000001")
    else:
        (labels / "notes.txt").write_text("")
    calls_before = len(lab.fake.calls)
    assert_aborted(*lab.run())
    assert len(lab.fake.calls) == calls_before
