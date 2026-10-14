"""`main` of src/mcpref/gui/evaluate.py against specs/gui-masks.md (Interface): exit 1 with a
one-line message as the last line of stderr, below the bars drawn before the error; no report."""

import masks_world as mw
import pytest
from masks_run import bars, run_main
from progress_bars import Bar

AT_START = {  # config changes, deleted files
    "patches_per_class above the number of patches": ({"patches_per_class": 9}, ()),
    "unreadable labels": ({}, ["labels/labels.jsonl"]),
}


@pytest.mark.parametrize("changes, remove", AT_START.values(), ids=AT_START)
def test_failures_at_the_start(tmp_path, repo, changes, remove):
    """16 x 8 frames, P = 4: 8 patches, checked when W, H are read; labels needed for the
    split: no bar, nothing but the message."""
    run = run_main(tmp_path, repo, {**mw.CONFIG, **changes}, remove=remove)
    assert run.failed() == [] and run.report is None


def test_empty_validation_set_fails_before_any_template(tmp_path, repo):
    """Checked right after the split: no bar and no file under out_dir before the message."""
    run = run_main(tmp_path, repo, {**mw.CONFIG, "split": ["0.5", "0", "0.5"]})
    assert run.failed() == []
    assert not run.out.exists() or not [p for p in run.out.rglob("*") if p.is_file()]


def test_abort_message_below_the_bar(tmp_path, repo):
    """A training frame of delta (in Q) unreadable: the bar `templates b`, drawn from 0 of its
    total, stops below it and is closed (leave=True); the message is the last line."""
    fs = mw.frames()
    part = mw.parts(fs, mw.CONFIG)
    lost = [f for f in fs if f.label == "delta" and part[f.relpath] == 0][-1]
    total = bars(mw.expected(fs, mw.CONFIG), mw.CONFIG)[0].total
    run = run_main(tmp_path, repo, mw.CONFIG, remove=[f"frames/frames/{lost.id}.yuv"])
    (above,) = run.failed()
    assert above == Bar("templates b", above.n, total, "step") and above.n < total, (
        above
    )
    assert run.report is None
