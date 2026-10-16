"""`main` of src/mcpref/recognition/evaluate.py against specs/gui-masks.md (R), (I), Interface:
exit 1 with a one-line message `mcpref.recognition.evaluate: <type>: <message>` as the last
line of stderr, below the bars drawn before the error; no report."""

import dataclasses
import json
import os

import masks_world as mw
import pytest
import yaml
from masks_files import plan_shards, write
from masks_run import expected_of, rerun, run_main
from progress_bars import Bar

PREFIX = "mcpref.recognition.evaluate: RuntimeError: "


def edit_json(path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def no_counts(root) -> None:
    """{frames_dir}/run.json as vpt-gui-frames writes it at the start of its run."""
    edit_json(root / "frames/run.json", lambda d: [d.pop("episodes"), d.pop("frames")])


def label_shard_lost(root) -> None:
    os.rename(root / "labels/labels/000005", root / "labels/labels/000005.part")


def other_frames(root) -> None:
    edit_json(root / "labels/run.json", lambda d: d["frames"].update(ok=53))


def too_many_patches(root) -> None:
    """24 x 20 frames, P = 4: 30 patches; 31 exceeds them."""
    cfg = yaml.safe_load((root / "gui-masks.yaml").read_text())
    (root / "gui-masks.yaml").write_text(
        yaml.safe_dump({**cfg, "patches_per_class": 31})
    )


def stale_out(root) -> None:
    """An {out_dir}/run.json whose every fingerprint differs."""
    run = {"config": {}, "sha256": {"config": "0" * 64, "screens": "0" * 64}}
    run.update(frames={"shards": 1, "ok": 1, "failed": 0}, labels="0" * 64)
    (root / "out").mkdir()
    (root / "out/run.json").write_text(json.dumps(run))


CHECKS = [  # (R) in order; each case also breaks every later check
    ("{f}/run.json lacks the counts: the frames are not complete", no_counts),
    ("{l} has 5 label shards, not 6", label_shard_lost),
    ("{l} does not label the current {f}", other_frames),
    ("patches_per_class 31 exceeds the 30 patches of a 24x20 frame", too_many_patches),
    ("config changed since the run started; delete {o}", stale_out),
]


@pytest.mark.parametrize("i", range(len(CHECKS)), ids=[c[1].__name__ for c in CHECKS])
def test_checks_in_order(tmp_path, repo, i):
    """(R): the first failing check aborts with its message, paths as written in the
    config, before any bar and any file written."""
    path = write(tmp_path, repo, mw.frames(), mw.CONFIG)
    for _, breaking in CHECKS[i:]:
        breaking(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    paths = {k[0]: cfg[f"{k}_dir"] for k in ("frames", "labels", "out")}
    run = rerun(tmp_path)
    assert run.failed() == [] and run.report is None
    assert run.err == PREFIX + CHECKS[i][0].format(**paths) + "\n"
    files = {p.name for p in run.out.rglob("*")} if run.out.exists() else set()
    assert files <= {"run.json"}


def test_more_label_shards_than_plan_shards(tmp_path, repo):
    """(R): an extra label shard is counted like a missing one."""
    path = write(tmp_path, repo, mw.frames(), mw.CONFIG)
    extra = tmp_path / "labels/labels/000006/labels.json"
    extra.parent.mkdir()
    extra.write_text(json.dumps({"labels": [], "end": 56}))
    run = rerun(tmp_path)
    labels = yaml.safe_load(path.read_text())["labels_dir"]
    assert (
        run.failed() == []
        and run.err == PREFIX + f"{labels} has 7 label shards, not 6\n"
    )


def bump_failed(root) -> None:
    """Another frames fingerprint, still labelled by the label shards' run.json."""
    for path in (root / "frames/run.json", root / "labels/run.json"):
        edit_json(path, lambda d: d["frames"].update(failed=d["frames"]["failed"] + 1))


def relabel_one(root) -> None:
    path = root / "labels/labels/000004/labels.json"
    edit_json(path, lambda d: d["labels"][0].update(seed=d["labels"][0]["seed"] + 1))


def append(name: str):
    return lambda root: (root / name).write_text((root / name).read_text() + "# x\n")


FINGERPRINTS = [  # (R), compared in this order
    ("config", append("gui-masks.yaml")),
    ("screens", append("screens.yaml")),
    ("frames", bump_failed),
    ("labels", relabel_one),
]


@pytest.mark.parametrize(
    "i", range(len(FINGERPRINTS)), ids=[f[0] for f in FINGERPRINTS]
)
def test_fingerprints_in_order(tmp_path, repo, i):
    """(R): a rerun with this and every later input changed names the first one: the config
    and catalogue files' bytes, the frames' counts, the label shards' bytes."""
    path = write(tmp_path, repo, mw.frames(), mw.CONFIG)
    font = tmp_path / "tex/font/ascii.png"
    os.rename(font, font.with_suffix(".bak"))
    assert rerun(tmp_path).failed()  # aborts in (T), after (R) and (I)
    os.rename(font.with_suffix(".bak"), font)
    for _, change in FINGERPRINTS[i:]:
        change(tmp_path)
    out = yaml.safe_load(path.read_text())["out_dir"]
    run = rerun(tmp_path)
    assert run.failed() == [] and run.report is None
    msg = f"{FINGERPRINTS[i][0]} changed since the run started; delete {out}"
    assert run.err == PREFIX + msg + "\n"


def test_unreadable_label_shard(tmp_path, repo):
    """The `labels` fingerprint of (R) reads every label shard: no bar, the message."""
    lost = ["labels/labels/000003/labels.json"]
    run = run_main(tmp_path, repo, mw.CONFIG, mw.frames(), remove=lost)
    assert run.failed() == [] and run.report is None


def test_label_outside_the_catalogue(tmp_path, repo):
    """(I): index shards 0-3 written (18 frames), then the message for shard 4's frame."""
    fs = mw.frames()
    bad = plan_shards(fs)[4][2]
    fs[fs.index(bad)] = dataclasses.replace(bad, label="zeta")
    run = run_main(tmp_path, repo, mw.CONFIG, fs)
    assert run.failed() == [Bar("index", 18, 54, "frame")]
    msg = f"label zeta of {bad.id} is not a class of the catalogue"
    assert run.err.endswith(PREFIX + msg + "\n") and run.report is None
    assert sorted(p.name for p in (run.out / "index").iterdir()) == [
        f"{k:06d}" for k in range(4)
    ]


def test_empty_validation_set_fails_before_any_template(tmp_path, repo):
    """(I): checked once the index is written, before any template."""
    config = {**mw.CONFIG, "split": ["0.5", "0", "0.5"]}
    run = run_main(tmp_path, repo, config, mw.frames())
    assert run.failed() == [Bar("index", 54, 54, "frame")]
    assert run.err.endswith(PREFIX + "the validation part is empty\n")
    assert not (run.out / "templates").exists()


def test_missing_font_fails(tmp_path, repo):
    """(9) reads font/ascii.png under the textures directory: without it, the message."""
    run = run_main(
        tmp_path, repo, mw.CONFIG, mw.frames(), remove=["tex/font/ascii.png"]
    )
    index, catalogue = run.failed()
    assert index == Bar("index", 54, 54, "frame")
    assert catalogue == Bar("templates catalogue", 0, 4, "class")
    assert run.report is None


def test_abort_message_below_the_bars(tmp_path, repo):
    """A training frame of delta (the third class of Q) unreadable: `templates b` stops
    after the 2 x 4 row blocks of alpha and beta and is closed (leave=True); the message is
    the last line."""
    exp = expected_of("base")
    lost = [f for f in exp["frames"]["train"] if f.label == "delta"][-1]
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    (tmp_path / f"frames/frames/{lost.id}.yuv").chmod(0)
    run = rerun(tmp_path)
    *_, above = run.failed()
    assert above == Bar("templates b", 8, 12, "step")
    assert run.report is None


@pytest.mark.parametrize("directory", ["frames/plan", "labels/labels"])
def test_foreign_entry_in_input_shards(tmp_path, repo, directory):
    """Interface: `complete_shards` raising (specs/shards.md (2)) on the plan or label shards
    aborts with its ValueError before any bar."""
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    (tmp_path / directory / "junk").write_text("")
    run = rerun(tmp_path)
    assert run.failed() == [] and run.report is None
    assert run.err.startswith("mcpref.recognition.evaluate: ValueError: ")
    assert run.err.endswith(": foreign entry junk\n")
