"""Video-specific ranged fetch-to-path: exact slices, streamed retries, cleanup and
wait-all/lowest-index failures; the generic in-memory fetch is covered separately."""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from vpt_checks import part_ranges

from mcpref.vpt.downloads import fetch_to_path
from mcpref.vpt.gui_frames import GuiFramesConfig

PATH = "/b1/data/7.1/video.mp4"


def fetch_file(site, data: bytes | None, q: int, target, **changes) -> bool:
    site.world.files[site.host + PATH] = data
    fields = site.fields | {"fetch_streams": q} | changes
    cfg = GuiFramesConfig(**fields)
    with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}, follow_redirects=False,
                      timeout=httpx.Timeout(cfg.http_timeout)) as client, ThreadPoolExecutor(q) as threads:  # fmt: skip
        return fetch_to_path(client, cfg, site.host + PATH, target, threads)


def body(size: int) -> bytes:
    return random.Random(size).randbytes(size)


def range_key(data: bytes, j: int) -> tuple:
    return "GET", PATH, part_ranges(len(data), 4)[j]


def test_ranges_write_exact_source_slices_and_complete_file(site, tmp_path):
    data, target = body(1003), tmp_path / "nested" / "video.mp4"
    assert fetch_file(site, data, 4, target)
    written = target.read_bytes()
    for byte_range in part_ranges(len(data), 4):
        first, last = map(int, byte_range.removeprefix("bytes=").split("-"))
        assert written[first : last + 1] == data[first : last + 1]
    assert written == data
    assert not tmp_path.joinpath("nested/video.mp4.part").exists()
    assert site.asked()[0] == site.head(PATH)
    assert sorted(site.asked()[1:]) == sorted(
        ("GET", PATH, r) for r in part_ranges(len(data), 4)
    )


def test_empty_and_missing_files_leave_no_partial(site, tmp_path):
    target = tmp_path / "nested" / "video.mp4"
    assert fetch_file(site, b"", 4, target)
    assert target.read_bytes() == b""
    target.unlink()
    assert not fetch_file(site, None, 4, target)
    assert not target.exists() and not target.with_name("video.mp4.part").exists()


def test_mid_body_transport_failure_is_retried_with_its_range(site, tmp_path):
    data, target = body(1003), tmp_path / "video.mp4"
    failed = range_key(data, 1)
    site.faults[failed] = ["cut"]
    assert fetch_file(site, data, 4, target)
    assert target.read_bytes() == data
    assert site.asked(PATH).count(failed) == 2
    assert not target.with_name("video.mp4.part").exists()


def test_mid_body_failure_exhaustion_cleans_partial(site, tmp_path):
    data, target = body(1003), tmp_path / "video.mp4"
    failed = range_key(data, 1)
    site.faults[failed] = ["cut"] * 3
    with pytest.raises(httpx.TransportError):
        fetch_file(site, data, 4, target)
    assert site.asked(PATH).count(failed) == 3
    assert not target.exists() and not target.with_name("video.mp4.part").exists()


def test_failure_waits_for_all_and_reports_lowest_index(site, tmp_path):
    data, target = body(1003), tmp_path / "video.mp4"
    low, high = range_key(data, 1), range_key(data, 3)
    site.faults, site.stalls = {low: [416], high: [404]}, {high: 0.3}
    started = time.monotonic()
    with pytest.raises(RuntimeError) as info:
        fetch_file(site, data, 4, target)
    assert time.monotonic() - started >= 0.3
    assert low[2] in str(info.value) and "HTTP 416" in str(info.value)
    assert site.asked(PATH).count(low) == site.asked(PATH).count(high) == 1
    assert not target.exists() and not target.with_name("video.mp4.part").exists()


def test_shared_executor_bounds_ranges_across_files(site, tmp_path, monkeypatch):
    from mcpref.vpt import downloads

    data = body(1003)
    site.world.files[site.host + PATH] = data
    cfg = GuiFramesConfig(**(site.fields | {"fetch_streams": 4}))
    original, lock, active = downloads.part_to_path, threading.Lock(), [0, 0]

    def measured(*args):
        with lock:
            active[0] += 1
            active[1] = max(active)
        try:
            time.sleep(0.03)
            return original(*args)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(downloads, "part_to_path", measured)
    with ThreadPoolExecutor(cfg.fetch_streams) as threads:

        def fetch(target):
            with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}) as client:
                return fetch_to_path(client, cfg, site.host + PATH, target, threads)

        with ThreadPoolExecutor(2) as videos:
            targets = [tmp_path / f"video-{j}.mp4" for j in range(2)]
            assert all(
                job.result() for job in [videos.submit(fetch, p) for p in targets]
            )
    assert active[1] == cfg.fetch_streams
    assert [path.read_bytes() for path in targets] == [data, data]
