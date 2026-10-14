"""`main` of src/mcpref/gui/labels.py (with its file helpers in label_files.py) against
specs/gui-labels.md: (2)-(5), output files, the bar `label`, reruns and aborts; fake ask_many,
fake frames (conftest.py, labels_world.py)."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
import yaml
from labels_world import (
    BATCHES,
    FRAMES,
    HASHED,
    OK,
    PATHS,
    QUESTION,
    SEED,
    TEXTS,
    Lab,
    jsonl,
    label_bar,
    summary,
)
from progress_bars import assert_aborted, assert_progress, screen, tqdm_defaults

from mcpref.vlm.client import load_config as load_vlm_config


@pytest.fixture
def lab(
    tmp_path,
    repo,
    catalogue,
    write_config,
    write_frames,
    fake_ask_many,
    monkeypatch,
    capsys,
):
    """Catalogue inventory, inventory_book, chest3; FRAMES; a vlm.yaml copy whose server refuses
    (an unfaked ask_many fails); config paths relative to the repository root; cwd elsewhere;
    tqdm's defaults (no TQDM_* variable)."""
    alpha, beta = catalogue["classes"]["alpha"], catalogue["classes"]["beta"]
    book = {**copy.deepcopy(alpha), "description": "Alpha with the book."}
    catalogue["classes"] = {"inventory": alpha, "inventory_book": book, "chest3": beta}
    write_config(catalogue)
    vlm = yaml.safe_load((repo / "configs/vlm.yaml").read_text())
    vlm["client"].update(base_url="http://127.0.0.1:9", max_resends=0)
    (tmp_path / "vlm.yaml").write_text(yaml.safe_dump(vlm))
    ids = write_frames(tmp_path / "frames", [f[:2] for f in FRAMES])
    paths = {k: os.path.relpath(tmp_path / v, repo) for k, v in PATHS.items()}
    cfg = {**paths, **TEXTS, "seed": SEED, "batch": len(BATCHES[0])}
    (tmp_path / "gui-labels.yaml").write_text(yaml.safe_dump(cfg))
    answers = {p: [a for a, _ in f[2]] for p, f in enumerate(FRAMES) if f[0] == "ok"}
    fake = fake_ask_many(answers, tmp_path / "out")
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    tqdm_defaults(monkeypatch)
    return Lab(tmp_path, ids, fake, capsys, write_frames)


def test_records_seeds_batches_prompts(lab):
    """(1)-(4): one ask_many call per batch of ok frames in frames.jsonl order, seeds by (2)."""
    code, err = lab.run()
    assert code == 0, err
    assert lab.read("labels.jsonl") == lab.records(OK)
    assert lab.asked() == [(b, [SEED + p for p in b]) for b in BATCHES]
    assert all(isinstance(s, int) for c in lab.fake.calls for s in c.seeds)
    prompts = [q for c in lab.fake.calls for q in c.prompts]
    assert all(len(q) == 3 and q[0] == TEXTS["instruction"] for q in prompts)
    assert all(q[2] == QUESTION for q in prompts)
    vlm = load_vlm_config(lab.root / "vlm.yaml")[0]
    assert all(c.cfg == vlm for c in lab.fake.calls)


def test_outputs_rewritten_after_every_batch(lab):
    """(4): only finished frames; each rewrite replaces the file (temporary file, then rename),
    so a handle opened before still reads the old text."""
    assert lab.run()[0] == 0
    files = [c.files for c in lab.fake.calls]
    labels = [jsonl(f["labels.jsonl"][0]) if "labels.jsonl" in f else [] for f in files]
    assert labels == [[], lab.records([1, 2, 4]), lab.records([1, 2, 4, 5, 6, 7])]
    first = ["inventory", "inventory_book", "chest3"]
    assert [json.loads(f["summary.json"][0]) for f in files[1:]] == [
        summary(first, 1, 1, 1, 0),
        summary([*first, "other", "invalid", "no_answer"], 1, 1, 1, 1),
    ]
    assert all(handle.read() == text for f in files for text, handle in f.values())
    assert sorted(lab.outputs()) == ["labels.jsonl", "run.json", "summary.json"]


def test_bar_total_and_final_count(lab):
    """Bar `label` on stderr only, from 0 to its total, the 7 ok frames (failed ones excluded).
    States between are drawn as tqdm's timing allows; the batch steps are checked through the
    state drawn at close after an abort (below)."""
    code, err = lab.run()
    assert code == 0 and lab.out == ""
    assert screen(err) == [label_bar(7, 7)]
    assert_progress(err)


def test_summary_and_vpt_check(lab):
    """(5) over class labels only, g_f from `line`; labels lists the labels that occurred."""
    assert lab.run()[0] == 0
    labels = ["inventory", "inventory_book", "chest3", "other", "invalid", "no_answer"]
    assert lab.read("summary.json") == summary([*labels, "other"], 1, 1, 2, 1)


def test_rerun_retries_no_answer_with_its_seed(lab):
    assert lab.run()[0] == 0
    code, err = lab.run()
    assert code == 0, err
    assert lab.asked(first=3) == [([7], [SEED + 7])]
    assert lab.read("labels.jsonl") == lab.records(OK, asks=2)
    labels = ["inventory", "inventory_book", "chest3", "chest3", "other", "other"]
    assert lab.read("summary.json") == summary([*labels, "invalid"], 1, 1, 2, 2)


def test_rerun_bar_counts_the_no_answer_frame(lab):
    """Bar total = frames still to label: the one no_answer frame."""
    assert lab.run()[0] == 0
    code, err = lab.run()
    assert code == 0 and screen(err) == [label_bar(1, 1)]
    assert_progress(err)


def test_finished_rerun_asks_nothing_and_is_already_done(lab):
    """Nothing left to label: no ask, files unchanged, no bar but the line `label: already done`."""
    assert lab.run()[0] == 0 and lab.run()[0] == 0
    before = lab.outputs()
    assert lab.run() == (0, "label: already done\n") and lab.out == ""
    assert len(lab.fake.calls) == 4 and lab.outputs() == before


def test_run_json(lab):
    assert lab.run()[0] == 0
    run = lab.read("run.json")
    assert set(run) == {"config", "sha256", "prompt"}
    files = {k: (lab.root / v).read_bytes() for k, v in HASHED.items()}
    assert run["sha256"] == {k: hashlib.sha256(b).hexdigest() for k, b in files.items()}
    assert run["prompt"] == [TEXTS["instruction"], "<image>", QUESTION]
    assert set(run["config"]) == {*PATHS, *TEXTS, "seed", "batch"}
    values = {k: run["config"][k] for k in (*TEXTS, "seed", "batch")}
    assert values == {**TEXTS, "seed": SEED, "batch": len(BATCHES[0])}


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("changed", HASHED.values())
def test_rerun_aborts_when_a_hash_differs(lab, changed, finished):
    for _ in range(1 + finished):
        assert lab.run()[0] == 0
    calls, before = len(lab.fake.calls), lab.outputs()
    path = lab.root / changed
    path.write_text(path.read_text() + "# changed\n")
    assert_aborted(*lab.run())
    assert len(lab.fake.calls) == calls and lab.outputs() == before


def test_abort_when_ask_many_raises_then_resume(lab):
    """Advanced by the first batch's size once written, the failed batch not counted: the bar
    is closed at 3 (a state drawn whatever the timing), the message below it; the resumed run's
    bar counts the 4 frames left."""
    lab.fake.fail_on = 2
    assert assert_aborted(*lab.run()) == [label_bar(3, 7)]
    assert lab.read("labels.jsonl") == lab.records([1, 2, 4])
    lab.fake.fail_on = None
    code, err = lab.run()
    assert code == 0 and screen(err) == [label_bar(4, 4)]
    assert_progress(err)
    assert [ps for ps, _ in lab.asked(first=2)] == [[5, 6, 7], [8]]
    assert lab.read("labels.jsonl") == lab.records(OK)


def test_resume_labels_no_answer_and_unasked_frames_together(lab):
    """(4): frames still to label = no record yet or no_answer, batched in frames.jsonl order,
    seeds by (2); the bar's total counts both kinds."""
    lab.fake.fail_on = 3
    assert assert_aborted(*lab.run()) == [label_bar(6, 7)]
    lab.fake.fail_on = None
    code, err = lab.run()
    assert code == 0 and screen(err) == [label_bar(2, 2)]
    assert_progress(err)
    assert lab.asked(first=3) == [([7, 8], [SEED + 7, SEED + 8])]
    assert lab.read("labels.jsonl") == lab.records(OK, asks=2)


@pytest.mark.parametrize("frames", [[], [("failed", True), ("failed", False)]])
def test_abort_without_ok_frame(lab, frames):
    lab.write_frames(lab.root / "frames", frames)
    assert_aborted(*lab.run())
    assert lab.fake.calls == []


MISSING = ["frames/frames.jsonl", "frames/run.json", "yuv", "gui-labels.yaml"]
MISSING += ["gui-screens.yaml", "vlm.yaml"]


@pytest.mark.parametrize("missing", MISSING)
def test_abort_on_unreadable_file(lab, missing):
    """The spec leaves open when a .yuv is read: a bar drawn before the abort ends at the frames
    written (here the last ok frame's .yuv is missing)."""
    yuv = f"frames/frames/{lab.ids[OK[-1]]}.yuv"
    (lab.root / (yuv if missing == "yuv" else missing)).unlink()
    above = assert_aborted(*lab.run())
    out = lab.root / "out" / "labels.jsonl"
    written = len(jsonl(out.read_text())) if out.exists() else 0
    assert above in ([], [label_bar(written, 7)])


def test_runs_as_module(lab):
    """`python -m mcpref.gui.labels <config path>`, here on frames without an ok one."""
    lab.write_frames(lab.root / "frames", [("failed", True)])
    argv = [sys.executable, "-m", "mcpref.gui.labels", lab.root / "gui-labels.yaml"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=8, check=False)
    assert_aborted(proc.returncode, proc.stderr)
