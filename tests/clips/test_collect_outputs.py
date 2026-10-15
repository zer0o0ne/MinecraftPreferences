"""Outputs of `collect` (specs/perception-data.md Interface, Datasets; (5), (6), (8))
against the transcription of collect_expected.py: the shards (records.json, state.json,
options/, clips/), the indices, run.json with its counts, no staging."""

import dataclasses
import json
from pathlib import Path

import pytest

from mcpref.clips.collect import PerceptionDataConfig

from .collect_expected import JSON_FILES
from .collect_out import assert_outputs
from .collect_site import HASHED
from .collect_world import Ep, jsonl

SAVED = ("malformed", "options", "video", "no_clip", None)  # (6): options files written
RECORD = ("subset", "relpath", "reason", "options", "starts_session", "overlay_start",
          "valid_starts", "clips")  # fmt: skip
CLIP = ("id", "subset", "relpath", "version", "states", "actions", "n", "gui", "classes",
        "pitch")  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1])
def test_outputs_of_a_run_through_every_pool(site, seed):
    """Seed 1: draws of several popped episodes, one ending with an empty pool and no
    candidate, others ended by candidates with the reasons `video` and `no_clip`."""
    site.run(seed=seed)
    ex = site.expected(seed=seed)
    assert_outputs(site, ex)
    assert {r["reason"] for r in ex.records} == {*SAVED, "version", "missing", "empty"}
    assert max(len(s.records) for s in ex.shards) >= 2


def test_json_files_are_indented_dumps_ending_with_a_newline(site):
    """json.dumps(value, indent=2) + newline, ensure_ascii: `café` of the options escaped."""
    site.run()
    files = site.raw()
    named = [p for p in files if p.rsplit("/", 1)[-1] in JSON_FILES] + ["run.json"]
    assert {p.rsplit("/", 1)[-1] for p in named} == {*JSON_FILES, "run.json"}
    for path in named:
        text = files[path].decode("ascii")
        assert text == json.dumps(json.loads(text), indent=2) + "\n", path
    assert all(b"caf\\u00e9" in x for p, x in files.items() if p.endswith("records.json") and b"caf" in x)  # fmt: skip


def test_json_keys_in_the_order_of_datasets(site):
    """records.json, clip.json and state.json as Datasets lists their fields."""
    site.run()
    for path, value in site.tree("shards").items():
        if path.endswith("records.json"):
            assert all(tuple(r) == RECORD for r in value), path
        elif path.endswith("clip.json"):
            assert tuple(value) == CLIP, path
        elif path.endswith("state.json"):
            assert tuple(value) == ("generator", "clips"), path


def test_run_json_holds_config_hashes_classifier_and_counts(site):
    site.run(gid="gamma v2")
    ex = site.expected()
    run = site.run_json()
    assert run == site.expected_run(ex) and run["classifier"] == "gamma v2"
    assert run["episodes"] == ex.counts() and run["clips"] == 8
    assert len(set(run["sha256"].values())) == 6 and tuple(run["sha256"]) == HASHED
    names = [f.name for f in dataclasses.fields(PerceptionDataConfig)]
    assert list(run["config"]) == names  # keys in the order of PerceptionDataConfig
    assert run["config"]["out_dir"] == site.fields["out_dir"]  # as written, unresolved


def test_clip_files_are_the_state_frames_and_the_span_lines(site):
    """frames.yuv: the L raw frames of the states in state order; lines.jsonl: the lines of
    the span byte for byte (compact JSON, raw UTF-8), each with its newline."""
    site.run()
    eps = {e.relpath: e for e in site.world.eps.values()}
    rows, shards, draw = site.rows(), site.raw("shards"), site.expected().draw
    assert len(rows) == 8
    for row in rows:
        e, st, k = eps[row["relpath"]], row["states"], draw[row["relpath"]]
        clip = f"shards/{k:06d}/clips/{row['id']}"
        assert shards[f"{clip}/frames.yuv"] == b"".join(e.frames()[t] for t in st)
        span = jsonl(e.script, e.session).split(b"\n")[st[0] : st[-1] + 1]
        lines = shards[f"{clip}/lines.jsonl"]
        assert lines == b"".join(x + b"\n" for x in span) and "é".encode() in lines


def test_options_files_are_byte_identical_for_reasons_after_empty(site):
    site.run()
    eps, written = {e.relpath: e for e in site.world.eps.values()}, {}
    for k, s in enumerate(site.expected().shards):
        for r in s.records:
            e = eps[r["relpath"]]
            if r["reason"] in SAVED:
                path = f"shards/{k:06d}/options/{e.version}_{e.stem}.json"
                written[path] = e.files()["-options.json"]
    produced = {p: x for p, x in site.raw("shards").items() if "/options/" in p}
    assert produced == written and len(written) == 10


@pytest.mark.parametrize("reason", SAVED[1:])
def test_record_fields_are_null_unless_computed(site, reason):
    filled = {"options": {"options"}, "video": {"options", "starts_session"},
              "no_clip": {"options", "starts_session", "overlay_start", "valid_starts"},
              None: set(RECORD[3:])}[reason]  # fmt: skip
    site.run()
    records = [r for r in site.records() if r["reason"] == reason]
    assert records
    for r in records:
        assert {k for k in RECORD[3:] if r[k] is not None} == filled, r


@pytest.mark.parametrize("reason", ["version", "missing", "empty", "malformed"])
def test_records_of_early_reasons_hold_only_subset_relpath_reason(site, reason):
    site.run()
    records = [r for r in site.records() if r["reason"] == reason]
    assert records and all(r[k] is None for r in records for k in RECORD[3:])


def test_options_reason_records_the_options_object(site):
    site.run()
    (rec,) = [r for r in site.records() if r["reason"] == "options"]
    assert rec["options"] == {"guiScale": 2, "autoJump": True,
                              "key_key.forward": "key.keyboard.w", "fov": 0.5,
                              "lastServer": "café"}  # fmt: skip


def test_clip_ids_are_version_stem_and_first_state_line(site):
    site.run()
    for row in site.rows():
        stem = row["relpath"].split("/")[2].removesuffix(".mp4")
        assert row["id"] == f"{row['version']}_{stem}_{row['states'][0]}"
        assert row["relpath"].startswith(f"data/{row['version']}/")


def test_staging_is_removed_after_a_run(site):
    """(5): the video is deleted in every case; (6): staging/ removed after the last shard."""
    site.run()
    assert not (site.out / "staging").exists()
    assert site.decodes()  # videos were fetched and decoded


def test_paths_resolve_against_the_repository_root(make_site):
    """The fixture's cwd sends every relative config path elsewhere if resolved against
    it (no config found, outputs under the cwd)."""
    site = make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", "w" * 6)]})
    site.run()
    assert site.records()[0]["reason"] is None
    assert [p.name for p in Path.cwd().iterdir()] == []
