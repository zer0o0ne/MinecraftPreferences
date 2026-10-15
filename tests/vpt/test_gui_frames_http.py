"""Behaviour (a), (b), the download of (e) and Retries of specs/vpt-gui-frames.md through
`main`: one GET per index and per `verify` video, the ranged fetch (f) of episode files,
retries of each request, aborts, the `x-ms-version` header, no redirects (Library mapping).
The server sends ETags; nothing depends on them."""

import itertools
import threading

import pytest
from progress_bars import assert_aborted
from vpt_checks import part_ranges
from vpt_world import FIELDS

from mcpref.vpt.gui_frames import main

TARGETS = [
    "index",
    "-options.json HEAD",
    "-options.json part",
    ".jsonl HEAD",
    ".jsonl part",
]
FAULTS = [500, 503, "drop", "hold", "cut"]  # cut: a body cut short, so not for a HEAD
TRANSIENT = [(w, f) for w in TARGETS for f in FAULTS if f != "cut" or "HEAD" not in w]
VIDEO = "/b4/" + FIELDS["verify_relpath"]  # listed by 10xx only, the fifth index


def target(site, which: str) -> tuple:
    """Request key: the 7xx index GET, or the HEAD or part 1 of the first fetched file."""
    if which == "index":
        return site.get("/idx/7xx.json")
    suffix, kind = which.split()
    path = site.first_fetched(suffix)
    return site.head(path) if kind == "HEAD" else site.part(path, 1)


def ours(site, key: tuple) -> list:
    """Requests of `key` sent by `main` itself (the fake ffmpeg sends Python-urllib's)."""
    asked = zip(site.server.requests, site.asked())
    urllib = "Python-urllib/"
    return [
        r
        for r, k in asked
        if k == key and not r.headers["User-Agent"].startswith(urllib)
    ]


def gaps(site, key: tuple) -> list[float]:
    return [b.t - a.t for a, b in itertools.pairwise(ours(site, key))]


@pytest.mark.parametrize("version", ["2019-12-12", "2020-04-08"])
def test_every_download_sends_x_ms_version(site, version):
    assert main(["run", site.config(x_ms_version=version)]) == 0
    sent = [r for r in site.server.requests if not r.path.endswith(".mp4")]
    assert {r.headers["x-ms-version"] for r in sent} == {version}
    assert {r.method for r in sent} == {"GET", "HEAD"} and any(
        k[2] for k in site.asked()
    )
    assert {r.path for r in sent if r.path.startswith("/idx/")} == {
        f"/idx/{s}.json" for s in site.etags
    }


@pytest.mark.parametrize("q", [1, 4, 20])
def test_indices_by_one_get_and_episode_files_by_head_then_parts(site, q):
    """(a): one GET per index; (b): options and .jsonl by (f), a HEAD then the exact parts
    of p = ceil(L / q) (none for L = 0); a `missing` episode may skip its other file."""
    assert main(["run", site.config(n_frames=100, fetch_streams=q)]) == 0
    for url in site.world.index_urls.values():
        path = url.removeprefix(site.host)
        assert site.asked(path) == [site.get(path)]
    for record in site.expected(n_frames=100)[1]:
        for suffix in ("-options.json", ".jsonl"):
            path = site.path(record["subset"], record["relpath"], suffix)
            asked, data = site.asked(path), site.data(path)
            if record["reason"] == "version" or (
                record["reason"] == "missing" and not asked
            ):
                assert asked == []
                continue
            ranges = part_ranges(len(data), q) if data else []
            assert asked[0] == site.head(path)
            assert sorted(asked[1:]) == sorted(("GET", path, r) for r in ranges)


def test_run_downloads_no_video_itself(site):
    """Out of scope: whole-video downloads except `verify`; videos only reach ffmpeg."""
    assert main(["run", site.config(n_frames=100)]) == 0
    videos = [r for r in site.server.requests if r.path.endswith(".mp4")]
    assert {r.headers["User-Agent"].split("/")[0] for r in videos} == {"Python-urllib"}


def test_verify_downloads_its_video_with_one_get(site):
    """(e): one GET without Range, no HEAD; (f) is out of scope for the `verify` video."""
    assert main(["verify", site.config()]) == 0
    assert [(r.method, r.headers["Range"]) for r in ours(site, site.get(VIDEO))] == [
        ("GET", None)
    ]
    assert [
        r for r in site.server.requests if r.path == VIDEO and r.method != "GET"
    ] == []


@pytest.mark.parametrize("which", TARGETS)
def test_timeout_per_phase_without_total_deadline(site, which):
    """Library mapping: 0.7 s before the headers and 0.7 s before the body, 1 s per
    phase: the download succeeds although it takes longer than http_timeout."""
    site.stalls[target(site, which)] = 0.7
    assert main(["run", site.config(http_timeout=1.0, http_retries=0)]) == 0
    assert site.plan() == site.shards()


@pytest.mark.parametrize("which, fault", TRANSIENT)
def test_transient_failure_is_retried_and_plan_unchanged(site, which, fault):
    key = target(site, which)
    site.faults[key] = [fault]
    assert main(["run", site.config(http_timeout=0.5, retry_delay=0.2)]) == 0
    assert site.plan() == site.shards()
    assert len(gaps(site, key)) == 1 and gaps(site, key)[0] >= 0.2
    assert fault != "hold" or ours(site, key)[0].cancelled()


@pytest.mark.parametrize("fault", [503, "drop", "hold"])
@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("which", TARGETS)
def test_retries_exhausted_abort_the_run(site, capfd, which, retries, fault):
    key = target(site, which)
    site.faults[key] = [fault] * 10
    cfg = site.config(http_timeout=0.5, http_retries=retries, retry_delay=0.1)
    assert_aborted(main(["run", cfg]), capfd.readouterr().err)
    assert len(gaps(site, key)) == retries
    assert all(gap >= 0.1 for gap in gaps(site, key))
    assert site.calls() == [] and site.files() == {}


@pytest.mark.parametrize("fault", [503, "drop", "cut"])
def test_verify_video_download_is_retried(site, fault):
    site.faults[site.get(VIDEO)] = [fault]
    assert main(["verify", site.config(retry_delay=0.2)]) == 0
    assert (
        len(gaps(site, site.get(VIDEO))) == 1 and gaps(site, site.get(VIDEO))[0] >= 0.2
    )


@pytest.mark.parametrize("retries", [0, 2])
def test_verify_video_retries_exhausted_abort(site, capfd, retries):
    site.faults[site.get(VIDEO)] = [503] * 10
    cfg = site.config(http_retries=retries, retry_delay=0.1)
    assert_aborted(main(["verify", cfg]), capfd.readouterr().err)
    assert len(ours(site, site.get(VIDEO))) == retries + 1 and site.calls() == []


@pytest.mark.parametrize("status", [404, 403])
def test_index_http_failure_aborts(site, capfd, status):
    site.faults[site.get("/idx/9xx.json")] = [status] * 10
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    assert site.calls() == [] and site.files() == {}


def test_refused_connection_is_retried(make_site):
    site = make_site(listen=False)
    timer = threading.Timer(0.3, site.server.listen)
    cfg = site.config(http_retries=100, retry_delay=0.05)
    timer.start()
    code = main(["run", cfg])
    timer.join()
    assert code == 0
    assert site.plan() == site.shards()


@pytest.mark.parametrize("which", TARGETS)
def test_redirect_is_not_followed(site, http, capfd, which):
    """A 302 is no success: an index, a HEAD or a part answering it aborts the run."""
    elsewhere = http(lambda req: (200, b"{}"))
    key = target(site, which)
    site.faults[key] = [302] * 10
    site.redirects[key[1]] = f"http://127.0.0.1:{elsewhere.port}/moved"
    assert_aborted(main(["run", site.config(retry_delay=0.05)]), capfd.readouterr().err)
    assert elsewhere.requests == [] and len(ours(site, key)) == 1
