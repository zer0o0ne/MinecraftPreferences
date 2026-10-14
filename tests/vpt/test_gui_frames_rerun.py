"""Behaviour (d) and the timing of run.json in specs/vpt-gui-frames.md: reruns and aborts."""

import pytest
from progress_bars import assert_aborted
from vpt_checks import key

from mcpref.vpt.gui_frames import main


def snapshot(site) -> dict:
    return {p: p.read_bytes() for p in site.out.rglob("*") if p.is_file()}


def test_rerun_same_config_skips_existing_frames(site):
    path = site.config()
    assert main(["run", path]) == 0
    before = snapshot(site)
    site.log.unlink()
    assert main(["run", path]) == 0
    assert site.calls() == []
    assert snapshot(site) == before


def test_rerun_tries_failed_frames_again(site):
    rows, _ = site.expected()
    failing = key(rows[3])
    site.behaviours = {failing: ["exit1"]}
    assert main(["run", site.config(http_retries=0)]) == 0
    assert site.run_json()["frames"] == {"ok": 5, "failed": 1}
    site.behaviours = {}
    site.log.unlink()
    assert main(["run", site.config(http_retries=0)]) == 0
    assert [c["key"] for c in site.calls()] == [failing]
    assert site.read_jsonl("frames.jsonl") == rows
    assert site.run_json()["frames"] == {"ok": 6}


def test_rerun_extracts_frame_whose_yuv_is_missing(site):
    rows, _ = site.expected()
    assert main(["run", site.config()]) == 0
    (site.out / "frames" / f"{rows[4]['id']}.yuv").unlink()
    site.log.unlink()
    assert main(["run", site.config()]) == 0
    assert [c["key"] for c in site.calls()] == [key(rows[4])]
    assert site.read_jsonl("frames.jsonl") == rows


@pytest.mark.parametrize(
    "change",
    [{"seed": 1}, {"n_frames": 7}, {"workers": 2}, {"fetch_streams": 5},
     {"ffmpeg_timeout": 4.0}],
)  # fmt: skip
def test_rerun_with_other_config_aborts_untouched(site, capfd, change):
    assert main(["run", site.config()]) == 0
    before = snapshot(site)
    site.log.unlink()
    capfd.readouterr()
    assert_aborted(main(["run", site.config(**change)]), capfd.readouterr().err)
    assert site.calls() == [] and snapshot(site) == before


@pytest.mark.parametrize(
    "etag", ["0x8DA5AB90A1B2FFF", None], ids=["changed", "dropped"]
)
def test_rerun_with_other_index_etag_aborts_untouched(site, capfd, etag):
    assert main(["run", site.config()]) == 0
    before = snapshot(site)
    site.etags["7xx"] = etag
    site.log.unlink()
    capfd.readouterr()
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.calls() == [] and snapshot(site) == before


def test_rerun_after_etag_appears_aborts(site, capfd):
    site.etags["10xx"] = None
    assert main(["run", site.config()]) == 0
    site.etags["10xx"] = "0x8DA5AB90A1B2F10"
    capfd.readouterr()
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)


@pytest.mark.parametrize("suffix", ["-options.json", ".jsonl"])
def test_run_json_written_before_planning(site, capfd, suffix):
    """An abort while planning leaves run.json with the config and the index ETags only."""
    site.faults[site.head(site.first_fetched(suffix))] = [503] * 10
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.run_json() == {"config": site.fields, "index_etags": site.etags}
    assert site.files() == {} and site.calls() == []


def test_interrupted_run_protects_its_config(site, capfd):
    site.faults[site.head(site.first_fetched(".jsonl"))] = [503] * 10
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    site.faults.clear()
    assert_aborted(main(["run", site.config(seed=3)]), capfd.readouterr().err)
    assert site.calls() == []
    assert main(["run", site.config()]) == 0
    assert site.read_jsonl("frames.jsonl") == site.expected()[0]
