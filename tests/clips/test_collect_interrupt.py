"""(6)-(8) of specs/perception-data.md: a rerun after an interruption (an abort by gamma,
a rename failing inside a commit, a kill at each step of a commit) replays the committed
episodes without fetching, decoding or classifying them, removes what no record lists,
empties staging and ends as the uninterrupted run."""

import json
import shutil

import pytest

from .collect_site import assert_outputs

STAGES = ("options", "one clip", "all clips", "record")  # steps of a commit, (6)


def stem(record: dict) -> str:
    return record["relpath"].split("/")[2].removesuffix(".mp4")


def test_rerun_after_an_abort_replays_and_continues(site):
    """gamma raises at its first call for the third episode that calls it."""
    site.aborted(site.failing_gamma())
    records = site.read_jsonl("episodes.jsonl")
    assert not site.completed() and len(records) >= 3
    assert records == site.expected().records[: len(records)]
    asked, decoded = len(site.server.requests), len(site.decodes())
    gamma = site.run()
    assert_outputs(site, site.expected())
    again = [r.path for r in site.server.requests[asked:]]
    for s in map(stem, records):
        assert not [p for p in again if f"/{s}." in p or f"/{s}-" in p], s
        assert s not in {d["key"] for d in site.decodes()[decoded:]}
        assert s not in gamma.by_stem()


def keep_only(site, records: list, ids: list, options: set) -> None:
    """Keeps the lines of `records` (byte for byte), the clips `ids` and the options files
    of the stems `options`; run.json loses its counts."""
    for name, n in (("episodes.jsonl", len(records)), ("clips.jsonl", len(ids))):
        text = (site.out / name).read_text()
        (site.out / name).write_text("".join(text.splitlines(keepends=True)[:n]))
    for path in (site.out / "clips").iterdir():
        if path.name not in ids:
            shutil.rmtree(path)
    for path in (site.out / "options").iterdir():
        if path.name.split("_", 1)[1].removesuffix("-options.json") not in options:
            path.unlink()
    run = site.run_json()
    del run["episodes"], run["clips"]
    (site.out / "run.json").write_text(json.dumps(run))


def killed_in_commit(site, ex, stage: str) -> int:
    """Turns the outputs of a complete run into those of a run killed inside the commit
    of the first episode with two clips, after `stage`: its clips not yet renamed and a
    later video lie in staging. Returns the episode's draw position k."""
    k = next(i for i, r in enumerate(ex.records) if r["clips"] and len(r["clips"]) == 2)
    renamed = {"options": 0, "one clip": 1}.get(stage, 2)
    clips = ex.records[k]["clips"]
    ids = [i for r in ex.records[:k] for i in r["clips"] or []] + clips[:renamed]
    records = ex.records[: k + (stage == "record")]
    keep_only(site, records, ids, {stem(r) for r in ex.records[: k + 1]})
    staging = site.out / "staging"
    for clip in clips[renamed:]:
        (staging / str(k) / f"{clip}.part").mkdir(parents=True)
        (staging / str(k) / f"{clip}.part" / "frames.yuv").write_bytes(b"x")
    staging.mkdir(exist_ok=True)
    (staging / f"{k + 1}.mp4").write_bytes(b"FAKEMP4 later\n")
    return k


@pytest.mark.parametrize("stage", STAGES)
def test_rerun_after_a_kill_inside_a_commit(site, stage):
    """The committed episodes are not decoded or classified again; the interrupted one is,
    unless its record was written."""
    site.run()
    ex = site.expected()
    k = killed_in_commit(site, ex, stage)
    decoded = len(site.decodes())
    gamma = site.run()
    assert_outputs(site, ex)
    again = {d["key"] for d in site.decodes()[decoded:]} | set(gamma.by_stem())
    committed = {stem(r) for r in ex.records[: k + (stage == "record")]}
    assert not again & committed
    assert (stem(ex.records[k]) in again) == (stage != "record")


def test_rerun_after_a_rename_fails_in_a_commit(site):
    """clips/ read-only: the first commit with clips writes the options file, fails to
    rename its first clip from staging/<r>/<id>.part and writes no record."""
    ex = site.expected()
    k = next(i for i, r in enumerate(ex.records) if r["clips"])
    version = ex.records[k]["relpath"].split("/")[1]
    (site.out / "clips").mkdir(parents=True)
    (site.out / "clips").chmod(0o555)
    try:
        site.aborted()
    finally:
        (site.out / "clips").chmod(0o755)
    assert site.read_jsonl("episodes.jsonl") == ex.records[:k] and not site.completed()
    assert f"options/{version}_{stem(ex.records[k])}-options.json" in site.tree()
    assert site.tree("clips") == {}
    staged = {
        f"staging/{k}/{c}.part/{name}": ex.files[f"clips/{c}/{name}"]
        for c in ex.records[k]["clips"]
        for name in ("frames.yuv", "lines.jsonl")
    }
    assert staged.items() <= site.tree("staging").items()
    site.run()
    assert_outputs(site, ex)
