"""Outputs of `collect` (specs/perception-data.md Interface, Produces; (5), (6)) against the
transcription of collect_expected.py: episodes.jsonl, clips.jsonl, clips/, options/,
run.json, an empty staging."""

import collections
import json
from pathlib import Path

import pytest
import yaml

from .collect_site import assert_outputs
from .collect_world import Ep, jsonl


@pytest.mark.parametrize("seed", [0, 5])
def test_outputs_of_a_run_through_every_pool(site, seed):
    site.run(seed=seed)
    assert_outputs(site, site.expected(seed=seed))


def test_clip_files_are_the_state_frames_and_the_span_lines(site):
    """frames.yuv: the L raw frames of the states in state order; lines.jsonl: the lines of
    the span byte for byte (compact JSON, raw UTF-8), each with its newline."""
    site.run()
    ex = site.expected()
    eps = {e.stem: e for e in site.world.eps.values()}
    for row in ex.clips:
        e, st = eps[row["relpath"].split("/")[2][:-4]], row["states"]
        frames = (site.out / "clips" / row["id"] / "frames.yuv").read_bytes()
        assert frames == b"".join(e.frames()[t] for t in st)
        span = jsonl(e.script, e.session).split(b"\n")[st[0] : st[-1] + 1]
        lines = (site.out / "clips" / row["id"] / "lines.jsonl").read_bytes()
        assert lines == b"".join(x + b"\n" for x in span) and "é".encode() in lines


def test_options_files_are_byte_identical_for_reasons_after_empty(site):
    site.run()
    eps, written = {e.relpath: e for e in site.world.eps.values()}, {}
    for r in site.expected().records:
        e = eps[r["relpath"]]
        if r["reason"] in ("malformed", "options", "video", "no_clip", None):
            written[f"options/{e.version}_{e.stem}-options.json"] = e.files()["-options.json"]  # fmt: skip
    assert site.tree("options") == written and len(written) == 10


def test_run_json_holds_config_etags_hashes_classifier_and_counts(site):
    site.etags["8xx"] = None
    site.run(gid="gamma v2")
    ex = site.expected()
    reasons = collections.Counter(r["reason"] or "null" for r in ex.records)
    config = yaml.safe_load(site.paths["perception-data"].read_text())
    names = ("actions", "clips", "screens", "gui_actions", "overlay", "overlay_screens")
    assert site.run_json() == {
        "config": config,
        "index_etags": site.etags,
        "sha256": {n: site.sha256(n) for n in names},
        "classifier": "gamma v2",
        "episodes": dict(reasons),
        "clips": len(ex.clips),
    }
    assert len({site.sha256(n) for n in names}) == 6


def test_staging_holds_nothing_after_a_run(site):
    """(5): the video is deleted in every case; (6): each committed episode's staging."""
    site.run()
    assert site.tree("staging") == {}
    assert site.decodes()  # videos were fetched and decoded


def test_clip_ids_are_version_stem_and_first_state_line(site):
    site.run()
    for row in site.read_jsonl("clips.jsonl"):
        stem = row["relpath"].split("/")[2].removesuffix(".mp4")
        assert row["id"] == f"{row['version']}_{stem}_{row['states'][0]}"
        assert row["relpath"].startswith(f"data/{row['version']}/")


def test_records_and_clips_are_one_json_object_per_line(site):
    site.run()
    for name in ("episodes.jsonl", "clips.jsonl"):
        text = (site.out / name).read_text()
        assert text.endswith("\n")
        assert all(isinstance(json.loads(x), dict) for x in text.splitlines())


def test_paths_resolve_against_the_repository_root(make_site):
    """The fixture's cwd sends every relative config path elsewhere if resolved against
    it (no config found, outputs under the cwd)."""
    site = make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", "w" * 6)]})
    site.run()
    assert site.read_jsonl("episodes.jsonl")[0]["reason"] is None
    assert [p.name for p in Path.cwd().iterdir()] == []
