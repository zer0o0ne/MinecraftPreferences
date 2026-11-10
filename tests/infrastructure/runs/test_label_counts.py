"""`run_counts` and `frame_counts` of src/mcpref/gui/label_files.py against the Interface of
specs/runs.md: the counts of a finished frames run.json, the two aborts."""

import json

import pytest

from mcpref.gui.label_files import frame_counts, run_counts

RUN_COUNTS = [  # run["frames"] -> {status: run["frames"].get(status, 0)} for ok, failed
    ({"ok": 3, "failed": 2}, {"ok": 3, "failed": 2}),
    ({"ok": 3}, {"ok": 3, "failed": 0}),
    ({}, {"ok": 0, "failed": 0}),
    ({"failed": 1, "ok": 4, "other": 9}, {"ok": 4, "failed": 1}),
]


@pytest.mark.parametrize("frames, counts", RUN_COUNTS)
def test_run_counts(frames, counts):
    assert run_counts({"config": {}, "frames": frames}) == counts


def frames_dir_with(tmp_path, run):
    """A resolved {frames_dir} whose run.json holds `run` (None: no run.json)."""
    frames_dir = tmp_path.resolve() / "frames"
    frames_dir.mkdir()
    if run is not None:
        (frames_dir / "run.json").write_text(json.dumps(run))
    return frames_dir


def test_frame_counts(tmp_path):
    frames_dir = frames_dir_with(tmp_path, {"config": {}, "frames": {"ok": 2}})
    assert frame_counts(frames_dir, "w/frames") == {"ok": 2, "failed": 0}


@pytest.mark.parametrize("run", [None, {"config": {}}, {"frames": None}])
def test_frame_counts_of_frames_not_complete(tmp_path, run):
    """finished_run, marker frames: the message with {frames_dir} as written."""
    with pytest.raises(RuntimeError) as error:
        frame_counts(frames_dir_with(tmp_path, run), "w/frames")
    assert str(error.value) == "w/frames is not complete"


@pytest.mark.parametrize("frames", [{}, {"failed": 3}, {"ok": 0, "failed": 1}])
def test_frame_counts_without_ok_frame(tmp_path, frames):
    """ok < 1: RuntimeError naming the resolved {frames_dir}/run.json."""
    frames_dir = frames_dir_with(tmp_path, {"config": {}, "frames": frames})
    with pytest.raises(RuntimeError) as error:
        frame_counts(frames_dir, "w/frames")
    assert str(error.value) == f"no ok frame in {frames_dir / 'run.json'}"
