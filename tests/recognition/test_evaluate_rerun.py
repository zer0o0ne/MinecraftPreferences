"""`main` of specs/gui-masks.md over several runs (Behaviour, Datasets): `run.json` before and
after a run, each stored set interrupted after some shards (a fake raising inside its bar) and
continued, a finished run rerun, `thresholds.json` written once and read by a rerun; the bars
of each run (totals, `initial`, lines)."""

import hashlib
import json
import os
from fractions import Fraction

import masks_world as mw
import numpy as np
import pytest
import yaml
from masks_files import write
from masks_oracle import predict
from masks_run import (
    SETS,
    UNITS,
    assert_bars,
    bars,
    config_of,
    expected_of,
    finished,
    frames_of,
    initial,
    interrupting,
    npz,
    rerun,
    shard_files,
    tree,
)
from progress_bars import screen


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def outputs(out) -> dict:
    """Every output file but run.json (the config's paths differ between worlds)."""
    return {k: v for k, v in tree(out).items() if k != "run.json"}


def test_run_json_before_and_after(tmp_path, repo):
    """(R): written after the checks with the config mapping, the sha256 of the config and
    catalogue files, `frames` (`failed` 0, absent from {frames_dir}/run.json without failed
    frames) and the `labels` hash; `"complete": true` added only at the end of a run that
    exits 0."""
    path = write(
        tmp_path, repo, [f for f in mw.frames() if f.status == "ok"], mw.CONFIG
    )
    counts = json.loads((tmp_path / "frames/run.json").read_text())["frames"]
    assert counts == {"ok": 54}
    font = tmp_path / "tex/font/ascii.png"
    os.rename(font, tmp_path / "font.png")
    assert rerun(tmp_path).failed()  # aborts in (T)
    shards = shard_files(tmp_path / "labels/labels", "labels.json")
    files = {"config": sha(path.read_bytes())}
    files["screens"] = sha((tmp_path / "screens.yaml").read_bytes())
    want = {"config": yaml.safe_load(path.read_text()), "sha256": files}
    want["frames"] = {"shards": 6, "ok": 54, "failed": 0}
    want["labels"] = sha(b"".join(p.read_bytes() for p in shards))
    assert json.loads((tmp_path / "out/run.json").read_text()) == want
    os.rename(tmp_path / "font.png", font)
    assert rerun(tmp_path).code == 0
    got = json.loads((tmp_path / "out/run.json").read_text())
    assert got == {**want, "complete": True}


@pytest.mark.parametrize("name", ["base", "unqualified", "no_test"])
def test_finished_run_rerun(tmp_path, repo, name):
    """A rerun after a complete run: every set `already done`, also when its shards are
    empty (the test part of `no_test`), but `nothing to do` for a set without any shard
    (`templates b` of `unqualified`); `thresholds: already done`; `model` and `report`
    redone (`report: nothing to do` without test frames); every output byte-identical."""
    exp = expected_of(name)
    write(tmp_path, repo, frames_of(name), config_of(name))
    assert rerun(tmp_path).code == 0
    before = tree(tmp_path / "out")
    run = rerun(tmp_path)
    assert run.code == 0, run.err
    assert screen(run.err) == bars(exp, finished(exp), thresholds=True)
    assert tree(run.out) == before


CASES = [  # bar, the count at which the fake raises, complete shards of its set then
    ("index", 18, 4),  # advanced by a shard once written: after shard 3
    ("templates catalogue", 2, 2),  # after each class's shard
    ("templates b", 6, 1),  # inside beta's row blocks
    ("templates c", 3, 1),  # inside build fold0
    ("features train", 10, 4),  # train frames per shard 1, 0, 0, 7, 7, 9
    ("score validation", 5, 4),  # 1, 0, 0, 3, 4, 3
    ("thresholds", 4, 0),  # no file before (6) ends
    ("score test", 7, 4),  # 0, 0, 0, 5, 5, 6
    ("report", 3, 0),
]


@pytest.mark.parametrize("desc, at, h", CASES, ids=[c[0] for c in CASES])
def test_interrupted_and_continued(tmp_path, repo, run_of, desc, at, h):
    """Datasets: a run failing inside a bar keeps the complete shards; the rerun continues
    after them, its bars start at the items of the complete shards (`initial`), and every
    output equals an uninterrupted run's byte for byte (`numpy.savez` is deterministic)."""
    exp = expected_of("base")
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    assert rerun(tmp_path, interrupting(desc, at)).failed()
    order = list(UNITS)  # run order
    done = {d: len(exp["items"][d]) for d in SETS if order.index(d) < order.index(desc)}
    if desc in SETS:
        done[desc] = h
    after = order.index(desc) > order.index("thresholds")
    run = rerun(tmp_path)
    assert run.code == 0, run.err
    assert screen(run.err) == bars(exp, done, thresholds=after)
    assert_bars(run.err, initial(exp, done))
    assert outputs(run.out) == outputs(run_of("base").out)


def test_thresholds_written_once_and_read_by_a_rerun(tmp_path, repo):
    """(S): a rerun after `thresholds.json` was written skips (6) (`thresholds: already
    done`), leaves the file as it is and predicts the test frames with the thresholds it
    holds (here replaced by tau 4, theta 0.5, mu 0.25 for every variant and choice, which
    change some predictions)."""
    exp = expected_of("base")
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    assert rerun(tmp_path, interrupting("score test", 1)).failed()
    point = {"tau": 4, "theta": "0.5", "mu": "0.25"}
    path = tmp_path / "out/thresholds.json"
    path.write_text(
        json.dumps({v: dict.fromkeys(mw.CRITERIA, point) for v in "abcdef"})
    )
    stat = path.stat()
    run = rerun(tmp_path)
    assert run.code == 0, run.err
    assert "thresholds: already done" in screen(run.err)
    assert (stat.st_ino, stat.st_mtime_ns) == (
        path.stat().st_ino,
        path.stat().st_mtime_ns,
    )
    floats = {"tau": 4, "theta": 0.5, "mu": 0.25}
    assert all(
        run.report[v][c]["thresholds"] == floats for v in "abcdef" for c in mw.CRITERIA
    )
    test = [npz(p) for p in shard_files(run.out / "features/test", "features.npz")]
    got = np.concatenate([z["pred"] for z in test]).tolist()
    before = np.concatenate([z["pred"] for z in exp["features"]["test"]]).tolist()
    assert got == predicted(exp, 4, Fraction("0.5"), Fraction("0.25")) != before


def predicted(exp: dict, tau: int, theta: Fraction, mu: Fraction) -> list:
    """`pred` of the test frames, (5) at (tau, theta, mu) for every variant and choice."""
    out = []
    for f in exp["frames"]["test"]:
        best = [exp["best"][v][f.id, tau] for v in mw.VARIANTS]
        scores = [{k: Fraction(*p) for k, p in b.items()} for b in best]
        out.append([[predict(s, mw.CLASSES, theta, mu)] * 2 for s in scores])
    return out
