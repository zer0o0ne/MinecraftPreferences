"""The index shards of `main` of specs/gui-masks.md (I): their frames, parts and generator
states against the oracle of masks_expected.py, and a run interrupted at every shard boundary
then continued."""

import json
import os

import masks_world as mw
import numpy as np
import pytest
from masks_files import plan_record, plan_shards, write, write_json
from masks_run import (
    assert_bars,
    expected_of,
    initial,
    interrupting,
    rerun,
    shard_files,
    tree,
)
from progress_bars import Bar, screen

N_S = len(plan_shards(mw.frames()))


def index(out) -> list:
    return [json.loads(p.read_text()) for p in shard_files(out / "index", "index.json")]


def test_worlds_exercise_the_shards():
    """Plan shards: one without frames, one with only a failed frame (both give empty index
    shards), episodes spanning shards; features shards of every part empty in some index
    shard."""
    exp = expected_of("base")
    shards = plan_shards(mw.frames())
    assert [len(s) for s in shards[1:3]] == [0, 1] and shards[2][0].status == "failed"
    assert exp["items"]["index"][1:3] == [0, 0]
    seen = [{f.relpath for f in s} for s in shards]
    assert all(seen[i] & seen[i + 1] for i in (3, 4))
    assert all(0 in exp["items"][d] for d in ("features train", "score test"))


def test_index_shards(world):
    """One index shard per plan shard, empty ones included: the `ok` frames in position
    order with id, episode, label (invalid, no_answer included), g, part; `rng` the split
    generator's state after the shard."""
    _, run, exp = world
    assert index(run.out) == exp["index"]


def parts(out) -> dict:
    """episode -> the parts of its frames over all index shards."""
    out_parts = {}
    for f in (f for s in index(out) for f in s["frames"]):
        out_parts.setdefault(f["episode"], set()).add(f["part"])
    return out_parts


def test_episodes_keep_their_part(run_of):
    """Split: an episode spanning shards has one part; relabelling (episode 0 all no_answer,
    episode 4 invalid) changes no part."""
    relabelled = parts(run_of("relabelled").out)
    assert all(len(p) == 1 for p in relabelled.values())
    assert relabelled == parts(run_of("base").out)


@pytest.mark.parametrize("s", range(N_S))
def test_index_interrupted_at_every_shard_boundary(tmp_path, repo, run_of, s):
    """(I): a run failing on plan shard s (unreadable) keeps index shards 0 .. s-1; the rerun
    continues from them (parts rebuilt, the generator set to the state of shard s-1), its
    `index` bar starting at their frames: every index shard byte-identical to an
    uninterrupted run's. The font is missing so that both runs stop after (I)."""
    exp = expected_of("base")
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    font = tmp_path / "tex/font/ascii.png"
    os.rename(font, tmp_path / "font.png")
    plan = tmp_path / f"frames/plan/{s:06d}/shard.json"
    os.rename(plan, tmp_path / "shard.json")
    assert rerun(tmp_path).failed()
    assert len(index(tmp_path / "out")) == s
    os.rename(tmp_path / "shard.json", plan)
    run = rerun(tmp_path)
    assert run.code == 1 and screen(run.err)[0] == Bar("index", 54, 54, "frame")
    assert_bars(run.err, initial(exp, {"index": s}))
    assert index(run.out) == exp["index"]
    want = {k: v for k, v in tree(run_of("base").out).items() if k.startswith("index/")}
    assert {k: v for k, v in tree(run.out).items() if k.startswith("index/")} == want


def trailing_shards(root) -> None:
    """Plan shards 6 (a failed frame) and 7 (no frame), their empty label shards, the counts
    of both run.json files."""
    late = mw.Frame("7.0_late_1", "data/7.0/late.mp4", "failed", False, None, None)
    rng = np.random.Generator(np.random.PCG64(6)).bit_generator.state
    for k, frames in ((6, [plan_record(late, 1)]), (7, [])):
        plan = {"episodes": [], "frames": frames, "rng": rng}
        write_json(root / f"frames/plan/{k:06d}/shard.json", plan)
        labels = {"labels": [], "end": 57}
        write_json(root / f"labels/labels/{k:06d}/labels.json", labels)
    for path, change in (
        ("frames/run.json", {"ok": 54, "failed": 3}),
        ("labels/run.json", {"shards": 8, "ok": 54, "failed": 3}),
    ):
        data = json.loads((root / path).read_text())
        data["frames"] = change
        write_json(root / path, data)


def test_index_nothing_to_do_then_already_done(tmp_path, repo):
    """Interface, progress bars: once the remaining plan shards hold no `ok` frame, `index:
    nothing to do`, their empty index shards still written (`rng` unchanged); every index
    shard complete before the run: `index: already done`."""
    exp = expected_of("base")
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    trailing_shards(tmp_path)
    assert rerun(tmp_path, interrupting("index", 54)).failed()
    assert len(index(tmp_path / "out")) == N_S
    lines = [screen(rerun(tmp_path).err)[0] for _ in range(2)]
    assert lines == ["index: nothing to do", "index: already done"]
    last = {"frames": [], "rng": exp["index"][-1]["rng"]}
    assert index(tmp_path / "out") == exp["index"] + [last, last]
