"""Behaviour (a), (d) and the timing of run.json in specs/vpt-gui-frames.md: stored indices,
reruns, continuation after an interruption, aborts."""

import collections
import shutil

import pytest
from progress_bars import assert_aborted, screen
from vpt_checks import key, stem
from vpt_world import REQUIRED, frame, joined

from mcpref.vpt.gui_frames import main


def counts(records: list[dict], frames: dict) -> dict:
    """run.json counts of a complete run."""
    reasons = collections.Counter(r["reason"] or "null" for r in records)
    return {"episodes": dict(reasons), "frames": frames}


@pytest.mark.parametrize(
    "n, k, boundary", [(6, 2, False), (6, 3, True), (100, 3, False), (100, 4, True)],
    ids=["N", "N on boundary", "pools", "pools on boundary"],
)  # fmt: skip
def test_rerun_of_finished_run_requests_nothing(site, capfd, n, k, boundary):
    """(3) ends in its third draw (N = 6) or its eighth (N = 100, every pool empty); a
    rerun makes no HTTP request, runs no ffmpeg and changes nothing."""
    path = site.config(n_frames=n, plan_shard_draws=k)
    assert main(["run", path]) == 0
    assert (len(site.draws(n_frames=n)) % k == 0) is boundary
    before = site.snapshot()
    site.log.unlink()
    site.server.requests.clear()
    capfd.readouterr()
    assert main(["run", path]) == 0
    assert site.server.requests == [] and site.calls() == []
    assert site.snapshot() == before
    assert "plan: already done" in screen(capfd.readouterr().err)


def test_rerun_tries_failed_frames_again(site):
    rows, records = site.expected()
    failing = key(rows[3])
    site.behaviours = {failing: ["exit1"]}
    assert main(["run", site.config(http_retries=0)]) == 0
    assert site.run_json()["frames"] == {"ok": 5, "failed": 1}
    site.behaviours = {}
    site.log.unlink()
    assert main(["run", site.config(http_retries=0)]) == 0
    assert [c["key"] for c in site.calls()] == [failing]
    assert site.files()[f"{rows[3]['id']}.yuv"] == frame(
        stem(rows[3]), rows[3]["frame"]
    )
    assert site.run_json() == {"config": site.fields | {"http_retries": 0}} | counts(
        records, {"ok": 6}
    )


def test_rerun_extracts_frame_whose_yuv_is_missing(site):
    rows, _ = site.expected()
    assert main(["run", site.config()]) == 0
    (site.out / "frames" / f"{rows[4]['id']}.yuv").unlink()
    site.log.unlink()
    assert main(["run", site.config()]) == 0
    assert [c["key"] for c in site.calls()] == [key(rows[4])]
    assert len(site.files()) == len(rows)


OUTPUT = {  # (d): output fields; a change aborts
    "index_urls": lambda f: {**f["index_urls"], "7xx": f["index_urls"]["7xx"] + "?v=2"},
    "index_urls reordered": lambda f: dict(reversed(f["index_urls"].items())),
    "min_version": lambda f: "6.13",
    "required_options": lambda f: REQUIRED | {"guiScale": 3},
    "required_options reordered": lambda f: dict(reversed(REQUIRED.items())),
    "required_options as JSON text": lambda f: REQUIRED | {"guiScale": 2.0},
    "width": lambda f: 2, "height": lambda f: 4, "fps": lambda f: 10,
    "n_frames": lambda f: 7, "per_episode": lambda f: 1, "seed": lambda f: 1,
}  # fmt: skip


@pytest.mark.parametrize("change", OUTPUT)
def test_rerun_with_other_output_field_aborts_untouched(site, capfd, change):
    """Compared as JSON text: reordered mappings and 2.0 for 2 differ."""
    assert main(["run", site.config()]) == 0
    before = site.snapshot()
    site.log.unlink()
    site.server.requests.clear()
    capfd.readouterr()
    field = change.split()[0]
    path = site.config(**{field: OUTPUT[change](site.fields)})
    assert_aborted(main(["run", path]), capfd.readouterr().err)
    assert site.calls() == [] and site.server.requests == []
    assert site.snapshot() == before


OTHER = {  # (d): every other field but out_dir; a change continues the run
    "x_ms_version": "2020-04-08", "workers": 1, "fetch_streams": 1, "plan_shard_draws": 3,
    "http_timeout": 3.0, "http_retries": 1, "retry_delay": 0.05, "ffmpeg": None,
    "ffmpeg_rw_timeout": 9, "ffmpeg_timeout": 4.0, "verify_relpath": "data/10.0/e10.mp4",
    "verify_frames": [3],
}  # fmt: skip


@pytest.mark.parametrize("field", OTHER)
def test_rerun_with_other_field_continues_the_plan(site, capfd, field):
    """Stopped in the plan after shard 0 (K = 2), continued with `field` changed (ffmpeg:
    a copy of the fake at another path): the plan as (3') cuts it, every frame, run.json of
    the last run's config."""
    site.stop_in_plan(capfd)
    value = OTHER[field]
    if field == "ffmpeg":
        value = str(site.tmp / "other-ffmpeg")
        shutil.copy(site.ffmpeg, value)
    assert main(["run", site.config(**{field: value})]) == 0
    k = site.fields["plan_shard_draws"] if field != "plan_shard_draws" else value
    assert site.plan() == site.shards(before=((1, 2),), plan_shard_draws=k)
    rows, records = site.expected()
    assert len(site.files()) == len(rows) == len(site.calls())
    program = value if field == "ffmpeg" else site.fields["ffmpeg"]
    assert {c["argv"][0] for c in site.calls()} == {program}
    assert site.run_json() == {"config": site.fields | {field: value}} | counts(
        records, {"ok": 6}
    )


def test_rerun_plans_from_stored_indices_without_request(site, capfd):
    """Stopped in the plan; the server then lists other episodes under other basedirs, with
    other ETags: the rerun downloads no index, plans from the stored ones and gives ffmpeg
    the stored basedirs."""
    expected, basedirs = site.shards(before=((1, 2),)), dict(site.world.basedirs)
    site.stop_in_plan(capfd)
    site.world.episodes["6xx"] = site.world.episodes["6xx"][:1]
    site.world.episodes["10xx"] = []
    site.world.basedirs = {s: f"{site.host}/moved/{s}/" for s in basedirs}
    site.etags = {s: "0x8DA5AB90A1B2FFF" for s in site.etags}
    site.server.requests.clear()
    assert main(["run", site.config()]) == 0
    assert [r.path for r in site.server.requests if r.path.startswith("/idx/")] == []
    assert site.plan() == expected
    urls = {c["argv"][c["argv"].index("-i") + 1] for c in site.calls()}
    assert urls == {
        basedirs[f["subset"]] + f["relpath"] for f in joined(expected, "frames")
    }


@pytest.mark.parametrize("subset", ["6xx", "9xx"])
def test_abort_in_indices_leaves_run_json_and_stored_indices(site, capfd, subset):
    """The index of `subset` fails: run.json holds the config alone, the indices before
    it are stored; a rerun downloads only the others, each once."""
    site.faults[site.get(f"/idx/{subset}.json")] = [404]
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.run_json() == {"config": site.fields}
    order = list(site.world.index_urls)
    stored = order[: order.index(subset)]
    assert sorted(p.name for p in (site.out / "indices").iterdir()) == sorted(
        f"{s}.json" for s in stored
    )
    site.faults.clear()
    site.server.requests.clear()
    assert main(["run", site.config()]) == 0
    asked = [r.path for r in site.server.requests if r.path.startswith("/idx/")]
    assert asked == [f"/idx/{s}.json" for s in order if s not in stored]
    assert site.plan() == site.shards()


@pytest.mark.parametrize("suffix", ["-options.json", ".jsonl"])
def test_abort_while_planning_leaves_run_json_without_counts(site, capfd, suffix):
    site.faults[site.head(site.first_fetched(suffix))] = [503] * 10
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.run_json() == {"config": site.fields}
    assert site.files() == {} and site.calls() == []


def test_abort_in_extraction_leaves_run_json_without_counts(site, capfd):
    """A rename raising (a directory at `<id>.yuv`) aborts (c)."""
    rows, _ = site.expected()
    site.behaviours = {key(rows[4]): ["blocked"]}
    site.sleep = 0.0
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.run_json() == {"config": site.fields}


def test_interrupted_run_continues_with_its_counts(site, capfd):
    """Counts of the run completing an interrupted one: episodes over every shard."""
    site.stop_in_plan(capfd)
    assert_aborted(main(["run", site.config(seed=3)]), capfd.readouterr().err)
    assert site.run_json() == {"config": site.fields | {"http_retries": 0}}
    assert main(["run", site.config()]) == 0
    _, records = site.expected()
    assert site.run_json() == {"config": site.fields} | counts(records, {"ok": 6})
