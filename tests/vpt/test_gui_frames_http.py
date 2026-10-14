"""Behaviour (a) and (b) of specs/vpt-gui-frames.md: index and episode downloads, retries,
aborts, the `x-ms-version` header, no redirects (Library mapping)."""

import itertools
import threading

import pytest
from progress_bars import assert_aborted

from mcpref.vpt.gui_frames import main

TARGETS = ["index", "-options.json", ".jsonl"]


def target(site, which: str) -> str:
    return "/idx/7xx.json" if which == "index" else site.first_fetched(which)


def gaps(site, path: str) -> list[float]:
    times = [r.t for r in site.server.requests if r.path == path]
    return [b - a for a, b in itertools.pairwise(times)]


@pytest.mark.parametrize("version", ["2019-12-12", "2020-04-08"])
def test_every_download_sends_x_ms_version(site, version):
    assert main(["run", site.config(x_ms_version=version)]) == 0
    ours = [r for r in site.server.requests if not r.path.endswith(".mp4")]
    assert {r.headers["x-ms-version"] for r in ours} == {version}
    assert {r.path for r in ours if r.path.startswith("/idx/")} == {
        f"/idx/{s}.json" for s in site.etags
    }


def test_run_downloads_no_video_itself(site):
    """Out of scope: whole-video downloads except `verify`; videos only reach ffmpeg."""
    assert main(["run", site.config(n_frames=100)]) == 0
    videos = [r for r in site.server.requests if r.path.endswith(".mp4")]
    assert {r.headers["User-Agent"].split("/")[0] for r in videos} == {"Python-urllib"}


@pytest.mark.parametrize("which", TARGETS)
def test_timeout_per_phase_without_total_deadline(site, which):
    """Library mapping: 0.7 s before the headers and 0.7 s before the body, 1 s per
    phase: the download succeeds although it takes longer than http_timeout."""
    site.stalls[target(site, which)] = 0.7
    assert main(["run", site.config(http_timeout=1.0, http_retries=0)]) == 0
    assert site.read_jsonl("frames.jsonl") == site.expected()[0]


@pytest.mark.parametrize("fault", [500, 503, "drop", "hold"])
@pytest.mark.parametrize("which", TARGETS)
def test_transient_failure_is_retried_and_plan_unchanged(site, which, fault):
    path = target(site, which)
    site.faults[path] = [fault]
    assert main(["run", site.config(http_timeout=0.5, retry_delay=0.2)]) == 0
    rows, records = site.expected()
    assert site.read_jsonl("frames.jsonl") == rows
    assert site.read_jsonl("episodes.jsonl") == records
    assert len(gaps(site, path)) == 1 and gaps(site, path)[0] >= 0.2
    held = next(r for r in site.server.requests if r.path == path)
    assert fault != "hold" or held.cancelled()


@pytest.mark.parametrize("fault", [503, "drop", "hold"])
@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("which", TARGETS)
def test_retries_exhausted_abort_the_run(site, capfd, which, retries, fault):
    path = target(site, which)
    site.faults[path] = [fault] * 10
    cfg = site.config(http_timeout=0.5, http_retries=retries, retry_delay=0.1)
    assert_aborted(main(["run", cfg]), capfd.readouterr().err)
    assert len(gaps(site, path)) == retries
    assert all(gap >= 0.1 for gap in gaps(site, path))
    assert site.calls() == [] and not (site.out / "frames.jsonl").exists()


@pytest.mark.parametrize("status", [404, 403])
def test_index_http_failure_aborts(site, capfd, status):
    site.faults["/idx/9xx.json"] = [status] * 10
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.calls() == [] and not (site.out / "frames.jsonl").exists()


def test_refused_connection_is_retried(make_site):
    site = make_site(listen=False)
    timer = threading.Timer(0.3, site.server.listen)
    cfg = site.config(http_retries=100, retry_delay=0.05)
    timer.start()
    code = main(["run", cfg])
    timer.join()
    assert code == 0
    assert site.read_jsonl("frames.jsonl") == site.expected()[0]


@pytest.mark.parametrize("which", TARGETS)
def test_redirect_is_not_followed(site, http, which):
    elsewhere = http(lambda req: (200, b"{}"))
    path = target(site, which)
    site.faults[path] = [302] * 10
    site.redirects[path] = f"http://127.0.0.1:{elsewhere.port}/moved"
    main(["run", site.config(http_retries=0, retry_delay=0.05)])
    assert elsewhere.requests == []
