"""`main` of src/mcpref/recognition/evaluate.py against specs/gui-masks.md (Interface): exit 1
with a one-line message as the last line of stderr, below the bars drawn before the error; no
report."""

import masks_world as mw
import pytest
from masks_run import bars, expected_of, run_main
from progress_bars import Bar

AT_START = {  # config changes, deleted files
    "patches_per_class above the number of patches": ({"patches_per_class": 31}, ()),
    "unreadable labels": ({}, ["labels/labels.jsonl"]),
}


@pytest.mark.parametrize("changes, remove", AT_START.values(), ids=AT_START)
def test_failures_at_the_start(tmp_path, repo, changes, remove):
    """24 x 20 frames, P = 4: 30 patches, checked when W, H are read; labels needed for the
    split: no bar, nothing but the message."""
    config = {**mw.CONFIG, **changes}
    run = run_main(tmp_path, repo, config, mw.frames(), remove=remove)
    assert run.failed() == [] and run.report is None


def test_empty_validation_set_fails_before_any_template(tmp_path, repo):
    """Checked right after the split: no bar and no file under out_dir before the message."""
    config = {**mw.CONFIG, "split": ["0.5", "0", "0.5"]}
    run = run_main(tmp_path, repo, config, mw.frames())
    assert run.failed() == []
    assert not run.out.exists() or not [p for p in run.out.rglob("*") if p.is_file()]


def test_missing_font_fails(tmp_path, repo):
    """(9) reads font/ascii.png under the textures directory: without it, the message."""
    run = run_main(
        tmp_path, repo, mw.CONFIG, mw.frames(), remove=["tex/font/ascii.png"]
    )
    run.failed()
    assert run.report is None


def test_abort_message_below_the_bars(tmp_path, repo):
    """A training frame of delta (in Q) unreadable: `templates catalogue` complete, then the
    bar `templates b`, drawn from 0 of its total, stops below it and is closed (leave=True);
    the message is the last line."""
    exp = expected_of("base")
    lost = [f for f in exp["frames"]["train"] if f.label == "delta"][-1]
    catalogue, b = bars(exp)[:2]
    remove = [f"frames/frames/{lost.id}.yuv"]
    run = run_main(tmp_path, repo, mw.CONFIG, mw.frames(), remove=remove)
    first, above = run.failed()
    assert first == catalogue == Bar("templates catalogue", 4, 4, "class")
    assert above == Bar("templates b", above.n, b.total, "step"), above
    assert above.n < b.total, above
    assert run.report is None
