"""`main` of src/mcpref/gui/labels.py (with its file helpers in label_files.py) against
specs/gui-labels.md: (2)-(5), output files, reruns and aborts; fake ask_many, fake frames."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from mcpref.gui.labels import main
from mcpref.vlm.client import load_config as load_vlm_config

PATHS = {"frames_dir": "frames", "screens": "gui-screens.yaml", "vlm": "vlm.yaml"}
PATHS["out_dir"] = "out"
TEXTS = {"instruction": "Screen?", "question": "Which?\n{classes}"}
TEXTS.update(class_line="{name}: {description}", other_description="something else")
QUESTION = "Which?\ninventory: Alpha screen.\ninventory_book: Alpha with the book."
QUESTION += "\nchest3: Beta screen.\nother: something else"
FRAMES = [  # status, line.isGuiInventory, (answer, its label by (3)) of successive asks
    ("failed", True, []),
    ("ok", True, [("inventory", "inventory")]),
    ("ok", False, [(" `inventory_book` \n", "inventory_book")]),
    ("failed", False, []),
    ("ok", True, [("chest3", "chest3")]),
    ("ok", False, [("other", "other")]),
    ("ok", True, [("Chest3", "invalid")]),
    ("ok", False, [(None, "no_answer"), ("'chest3'", "chest3")]),
    ("ok", True, [('"other"', "other")]),
]
OK = [1, 2, 4, 5, 6, 7, 8]
SEED = 5
BATCHES = [[1, 2, 4], [5, 6, 7], [8]]  # batch = 3
VPT = [f"{y}_flag_{g}" for y in ("inventory", "other") for g in ("true", "false")]
HASHED = {"config": "gui-labels.yaml", "vlm": "vlm.yaml", "screens": "gui-screens.yaml"}


def summary(labels: list, *flags: int) -> dict:
    """summary.json: the count of each label listed; the four (5) counts in the order of VPT."""
    return {"labels": dict(Counter(labels)), "vpt_check": dict(zip(VPT, flags))}


@dataclass
class Lab:
    """One fake run directory; `run` calls main and returns (exit code, stderr)."""

    root: Path
    ids: list
    fake: object
    capsys: object
    write_frames: object

    def run(self) -> tuple[int, str]:
        code = main([str(self.root / "gui-labels.yaml")])
        return code, self.capsys.readouterr().err

    def read(self, name: str):
        text = (self.root / "out" / name).read_text()
        return jsonl(text) if name.endswith(".jsonl") else json.loads(text)

    def outputs(self) -> dict:
        out = self.root / "out"
        return {p.name: p.read_bytes() for p in out.iterdir()} if out.exists() else {}

    def records(self, ps: list, asks: int = 1) -> list:
        """labels.jsonl of positions ps, each frame asked at most `asks` times; seeds by (2)."""
        recs = []
        for p in ps:
            answer, label = FRAMES[p][2][min(asks, len(FRAMES[p][2])) - 1]
            recs.append({"id": self.ids[p], "seed": SEED + p, "answer": answer})
            recs[-1]["label"] = label
        return recs

    def asked(self, first: int = 0) -> list:
        """Per ask_many call from `first` on: (frame positions of its prompts, its seeds)."""
        calls = self.fake.calls[first:]
        return [([self.fake.position(q) for q in c.prompts], c.seeds) for c in calls]


def jsonl(text: str) -> list:
    return [json.loads(line) for line in text.splitlines()]


def failed(code: int, err: str) -> None:
    """Exit 1 with a one-line message on stderr (Interface)."""
    assert code == 1 and len(err.strip().splitlines()) == 1 and "Traceback" not in err


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
    (an unfaked ask_many fails); config paths relative to the repository root; cwd elsewhere."""
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


def test_finished_rerun_asks_nothing(lab):
    assert lab.run()[0] == 0 and lab.run()[0] == 0
    before = lab.outputs()
    assert lab.run()[0] == 0
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
    failed(*lab.run())
    assert len(lab.fake.calls) == calls and lab.outputs() == before


def test_abort_when_ask_many_raises_then_resume(lab):
    lab.fake.fail_on = 2
    failed(*lab.run())
    assert lab.read("labels.jsonl") == lab.records([1, 2, 4])
    lab.fake.fail_on = None
    assert lab.run()[0] == 0
    assert [ps for ps, _ in lab.asked(first=2)] == [[5, 6, 7], [8]]
    assert lab.read("labels.jsonl") == lab.records(OK)


@pytest.mark.parametrize("frames", [[], [("failed", True), ("failed", False)]])
def test_abort_without_ok_frame(lab, frames):
    lab.write_frames(lab.root / "frames", frames)
    failed(*lab.run())
    assert lab.fake.calls == []


MISSING = ["frames/frames.jsonl", "frames/run.json", "yuv", "gui-labels.yaml"]
MISSING += ["gui-screens.yaml", "vlm.yaml"]


@pytest.mark.parametrize("missing", MISSING)
def test_abort_on_unreadable_file(lab, missing):
    yuv = f"frames/frames/{lab.ids[OK[-1]]}.yuv"
    (lab.root / (yuv if missing == "yuv" else missing)).unlink()
    failed(*lab.run())


def test_runs_as_module(lab):
    """`python -m mcpref.gui.labels <config path>`, here on frames without an ok one."""
    lab.write_frames(lab.root / "frames", [("failed", True)])
    argv = [sys.executable, "-m", "mcpref.gui.labels", lab.root / "gui-labels.yaml"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=8, check=False)
    failed(proc.returncode, proc.stderr)
