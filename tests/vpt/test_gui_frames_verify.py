"""Behaviour (e) of specs/vpt-gui-frames.md: `main verify` with a fake ffmpeg (its bars
`indices`, `decode`, `verify` and ffmpeg's stderr included), and once with the real ffmpeg
on a tiny generated H.264 video."""

import subprocess
import tempfile
from pathlib import Path

import pytest
from progress_bars import Bar, assert_aborted, assert_progress, screen, states
from vpt_checks import EPISODES, command_head
from vpt_world import FIELDS

from mcpref.vpt.gui_frames import main

VERIFY = FIELDS["verify_relpath"]
REAL_FFMPEG = Path("/home/dev/.local/bin/ffmpeg")
FAILURES = ["exit1", "short", "long", "slow"]  # slow: longer than ffmpeg_timeout
INDICES = Bar("indices", 5, 5, "file")  # verify_relpath is listed by 10xx only
DECODE = Bar("decode", 25, None, "frame")  # its video: 24 lines, 25 frames
VERIFY_BAR = Bar("verify", 5, 5, "frame")


def vkey(i: int) -> str:
    return f"verify-me_{i}"


def extractions(site, i: int | None = None) -> list[dict]:
    calls = [c for c in site.calls() if c["key"] != "decode"]
    return [c for c in calls if i is None or c["key"] == vkey(i)]


@pytest.fixture
def temp_dir(tmp_path, monkeypatch) -> Path:
    """The temporary directory of `tempfile` and of child processes."""
    path = tmp_path / "tmp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    monkeypatch.setenv("TMPDIR", str(path))
    return path


def test_verify_extracts_each_frame_and_decodes_the_download(site):
    assert main(["verify", site.config()]) == 0
    url = site.world.basedirs["10xx"] + VERIFY
    calls = extractions(site)
    assert sorted(c["key"] for c in calls) == sorted(map(vkey, FIELDS["verify_frames"]))
    for c in calls:
        i = int(c["key"].rsplit("_", 1)[1])
        assert c["argv"][:-1] == command_head(site.fields, url, i)
        assert c["argv"][-1].endswith(".part")
    decodes = [c for c in site.calls() if c["key"] == "decode"]
    assert len(decodes) == 1 and decodes[0]["argv"][0] == site.fields["ffmpeg"]
    assert url not in decodes[0]["argv"]  # a local copy, decoded sequentially


def test_verify_bars_totals_and_final_counts(site, capfd):
    assert main(["verify", site.config()]) == 0
    out, err = capfd.readouterr()
    assert out == "" and screen(err) == [INDICES, DECODE, VERIFY_BAR]
    assert_progress(err)


def test_verify_without_frames_prints_nothing_to_do(site, capfd):
    assert main(["verify", site.config(verify_frames=[])]) == 0
    err = capfd.readouterr().err
    assert screen(err).count("verify: nothing to do") == 1
    assert states(err, "verify") == [] and extractions(site) == []


def test_verify_url_from_first_index_listing_it(make_site, capfd):
    """7xx precedes 10xx in config order and lists the same relpath: the `indices` bar
    stops after the second download."""
    episodes = EPISODES | {"7xx": [*EPISODES["7xx"], EPISODES["10xx"][1]]}
    site = make_site(episodes=episodes)
    assert main(["verify", site.config()]) == 0
    assert screen(capfd.readouterr().err)[0] == Bar("indices", 2, 5, "file")
    base = site.world.basedirs["7xx"]
    assert {c["argv"][c["argv"].index("-i") + 1] for c in extractions(site)} == {
        base + VERIFY
    }
    videos = {r.path for r in site.server.requests if r.path.endswith(".mp4")}
    assert videos == {base.removeprefix(site.host) + VERIFY}


def test_verify_aborts_when_no_index_lists_relpath(site, capfd):
    cfg = site.config(verify_relpath="data/10.0/nowhere.mp4")
    code = main(["verify", cfg])
    err = capfd.readouterr().err
    assert_aborted(code, err)
    assert site.calls() == [] and screen(err)[:-1] == [INDICES]


def assert_verify_stopped(code: int, err: str) -> None:
    """Aborted with the `verify` bar closed below its total: a frame never equalled."""
    assert_aborted(code, err)
    *done, verify = screen(err)[:-1]
    assert done == [INDICES, DECODE]
    assert (verify.desc, verify.total, verify.unit, verify.postfix) == (
        "verify",
        5,
        "frame",
        None,
    )
    assert verify.n <= 4


@pytest.mark.parametrize("i", [0, 20, 23])
def test_verify_aborts_on_a_frame_unequal_to_the_decode(site, capfd, i):
    site.behaviours = {vkey(i): ["wrong"]}
    assert_verify_stopped(main(["verify", site.config()]), capfd.readouterr().err)


@pytest.mark.parametrize("failure", FAILURES)
def test_verify_retries_a_failed_extraction(site, failure):
    site.behaviours = {vkey(19): [failure, "ok"]}
    assert main(["verify", site.config(ffmpeg_timeout=1.0, retry_delay=0.2)]) == 0
    starts = [c["t"] for c in extractions(site, 19)]
    assert len(starts) == 2 and starts[1] - starts[0] >= 0.2


@pytest.mark.parametrize("failure", FAILURES)
def test_verify_aborts_when_a_frame_still_fails(site, capfd, failure):
    site.behaviours = {vkey(1): [failure] * 2}
    cfg = site.config(ffmpeg_timeout=1.0, http_retries=1, retry_delay=0.05)
    assert_verify_stopped(main(["verify", cfg]), capfd.readouterr().err)
    assert len(extractions(site, 1)) == 2


def test_verify_prints_ffmpeg_stderr_above_the_running_bar(site, capfd):
    """The decode's stderr above `decode`, an extraction's above `verify`."""
    site.noise = {"decode": [b"decode note\n"], vkey(19): [b"seek note\n"]}
    assert main(["verify", site.config()]) == 0
    shown = [INDICES, "decode note", DECODE, "seek note", VERIFY_BAR]
    err = capfd.readouterr().err
    assert screen(err) == shown
    assert_progress(err)


def test_verify_decode_never_blocked_by_its_stderr(site, capfd):
    """256 KiB of stderr, beyond a pipe's buffer, before the decoded video on stdout."""
    big = "e" * 2**18
    site.noise = {"decode": [big.encode()]}
    assert main(["verify", site.config()]) == 0
    assert screen(capfd.readouterr().err) == [INDICES, big, DECODE, VERIFY_BAR]


@pytest.mark.parametrize("behaviour, code", [("ok", 0), ("wrong", 1)])
def test_verify_deletes_its_temporary_files(site, temp_dir, behaviour, code):
    site.behaviours = {vkey(20): [behaviour]}
    assert main(["verify", site.config()]) == code
    assert list(temp_dir.iterdir()) == []
    assert not [p for p in site.out.rglob("*") if p.is_file()]


def test_verify_downloads_indices_itself_and_writes_nothing(site):
    """After a run has stored the indices: `verify` downloads them again, in config order up
    to 10xx (the first listing verify_relpath), and leaves out_dir as it is."""
    assert main(["run", site.config()]) == 0
    before = site.snapshot()
    site.server.requests.clear()
    assert main(["verify", site.config()]) == 0
    asked = [r.path for r in site.server.requests if r.path.startswith("/idx/")]
    assert asked == [f"/idx/{s}.json" for s in site.world.index_urls]
    assert site.snapshot() == before


def h264(path: Path, frames: int) -> bytes:
    """32 x 16 yuv420p H.264 at 20 fps, a keyframe every 10 frames, B-frames, noise."""
    src = "testsrc2=size=32x16:rate=20,noise=alls=40:allf=t"
    cmd = [str(REAL_FFMPEG), "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", src,
           "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10",
           "-bf", "2", str(path)]  # fmt: skip
    subprocess.run(cmd, check=True, timeout=30)
    return path.read_bytes()


@pytest.mark.skipif(not REAL_FFMPEG.exists(), reason="needs a real ffmpeg")
def test_verify_with_real_ffmpeg_over_byte_ranges(site, tmp_path, temp_dir):
    """[R]: -ss with the exact decimal of i / 20 returns frame i of the sequential decode."""
    video = h264(tmp_path / "v.mp4", 25)
    site.world.files[site.world.basedirs["10xx"] + VERIFY] = video
    frames = [0, 1, 9, 10, 11, 19, 20, 24]
    cfg = site.config(
        ffmpeg=str(REAL_FFMPEG), width=32, height=16, verify_frames=frames
    )
    assert main(["verify", cfg]) == 0
    ranged = [r for r in site.server.requests if r.headers["Range"]]
    assert ranged and {r.headers["x-ms-version"] for r in ranged} == {"2019-12-12"}
    assert list(temp_dir.iterdir()) == []


@pytest.mark.skipif(not REAL_FFMPEG.exists(), reason="needs a real ffmpeg")
def test_verify_with_real_ffmpeg_rejects_wrong_frame_size(site, tmp_path, capfd):
    """Not width x height x 3 / 2 bytes: the extraction fails; a frame still failing aborts."""
    site.world.files[site.world.basedirs["10xx"] + VERIFY] = h264(
        tmp_path / "v.mp4", 25
    )
    cfg = site.config(ffmpeg=str(REAL_FFMPEG), width=16, height=16, verify_frames=[3],
                      http_retries=0)  # fmt: skip
    assert_aborted(main(["verify", cfg]), capfd.readouterr().err)
