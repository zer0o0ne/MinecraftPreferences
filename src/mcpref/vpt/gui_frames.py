"""Interface of specs/vpt-gui-frames.md: random VPT frames with an open GUI."""

import argparse
import dataclasses
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.http.retry import get
from mcpref.vpt.config import GuiFramesConfig, load_config
from mcpref.vpt.episodes import (
    EpisodeRecord,
    PlannedFrame,
    eligible_frames,
    index,
    options_ok,
    plan,
    version_ok,
)
from mcpref.vpt.ffmpeg import decode, extract, ffmpeg_command, frame_time
from mcpref.vpt.yuv import yuv420p_to_rgb

__all__ = [
    "EpisodeRecord",
    "GuiFramesConfig",
    "PlannedFrame",
    "eligible_frames",
    "ffmpeg_command",
    "frame_time",
    "load_config",
    "main",
    "options_ok",
    "plan",
    "version_ok",
    "yuv420p_to_rgb",
]


def download(client: httpx.Client, cfg: GuiFramesConfig, url: str) -> httpx.Response:
    """(a), (e): a GET that must succeed after the retries."""
    response = get(client, url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)
    return response.raise_for_status()


def fetch(client: httpx.Client, cfg: GuiFramesConfig, url: str) -> bytes | None:
    """(b): the body, or None for HTTP 404; other failures after the retries abort."""
    response = get(client, url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)
    return None if response.status_code == 404 else response.raise_for_status().content


def extract_all(
    cfg: GuiFramesConfig, frames: list[PlannedFrame], urls: list[str], out: Path
) -> list[str]:
    """(c), (d): statuses in plan order, `workers` threads; existing <id>.yuv skipped."""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{f.id}.yuv" for f in frames]
    statuses = ["ok"] * len(frames)
    todo = [k for k, path in enumerate(paths) if not path.exists()]
    with ThreadPoolExecutor(cfg.workers) as threads:
        jobs = {
            threads.submit(extract, cfg, urls[k], frames[k].frame, paths[k]): k
            for k in todo
        }
        for done, job in enumerate(as_completed(jobs), 1):
            statuses[jobs[job]] = "ok" if job.result() else "failed"
            failed = statuses.count("failed")
            print(f"extracted {done}/{len(todo)}, {failed} failed", file=sys.stderr)
    return statuses


def check_rerun(path: Path, header: dict) -> None:
    """(d): an existing run.json must hold the same config and index ETags."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    same = json.dumps(previous["config"]) == json.dumps(header["config"])
    if not same or previous["index_etags"] != header["index_etags"]:
        raise RuntimeError(f"{path} holds another config or other index ETags")


def write_run(out: Path, run: dict) -> None:
    """Behaviour: run.json, the header before planning, completed with the counts."""
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def frame_record(frame: PlannedFrame, status: str) -> dict:
    """Interface: one record of frames.jsonl."""
    record = frame._asdict()
    prev_line, line = record.pop("lines")
    return record | {"prev_line": prev_line, "line": line, "status": status}


def write_records(
    out: Path,
    header: dict,
    frames: list[PlannedFrame],
    episodes: list[EpisodeRecord],
    statuses: list[str],
) -> None:
    """Interface Produces: frames.jsonl (plan order), episodes.jsonl (draw order), run.json."""
    records = map(frame_record, frames, statuses)
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    lines = (json.dumps(e._asdict()) + "\n" for e in episodes)
    (out / "episodes.jsonl").write_text("".join(lines))
    reasons, counts = Counter(e.reason for e in episodes), Counter(statuses)
    write_run(out, header | {"episodes": reasons, "frames": counts})


def run(cfg: GuiFramesConfig, client: httpx.Client) -> None:
    """Behaviour (a)-(d)."""
    out = REPO / cfg.out_dir
    indices = {s: download(client, cfg, url) for s, url in cfg.index_urls.items()}
    etags = {s: response.headers.get("ETag") for s, response in indices.items()}
    header = {"config": dataclasses.asdict(cfg), "index_etags": etags}
    check_rerun(out / "run.json", header)
    out.mkdir(parents=True, exist_ok=True)
    write_run(out, header)
    bodies = {cfg.index_urls[s]: response.content for s, response in indices.items()}
    frames, episodes = plan(
        cfg, lambda url: bodies[url] if url in bodies else fetch(client, cfg, url)
    )
    basedirs = {s: index(response.content)[0] for s, response in indices.items()}
    urls = [basedirs[f.subset] + f.relpath for f in frames]
    statuses = extract_all(cfg, frames, urls, out / "frames")
    write_records(out, header, frames, episodes, statuses)


def video_url(client: httpx.Client, cfg: GuiFramesConfig) -> str:
    """(e): verify_relpath under the basedir of the index that lists it."""
    for url in cfg.index_urls.values():
        basedir, relpaths = index(download(client, cfg, url).content)
        if cfg.verify_relpath in relpaths:
            return basedir + cfg.verify_relpath
    raise RuntimeError(f"no index lists {cfg.verify_relpath}")


def verify(cfg: GuiFramesConfig, client: httpx.Client) -> None:
    """(e): every verify frame by ffmpeg_command equals the sequential decode."""
    url = video_url(client, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "video.mp4"
        video.write_bytes(download(client, cfg, url).content)
        decoded = decode(cfg, video)
        for i in cfg.verify_frames:
            path = Path(tmp) / f"{i}.yuv"
            if not extract(cfg, url, i, path) or path.read_bytes() != decoded.get(i):
                raise RuntimeError(f"frame {i} of {url} is not the sequential decode")
            print(f"verified frame {i}", file=sys.stderr)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.vpt.gui_frames {verify|run} <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog="python -m mcpref.vpt.gui_frames")
    parser.add_argument("command", choices=["verify", "run"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}) as client:
            (verify if args.command == "verify" else run)(cfg, client)
    except FAILURES as error:
        return failed(f"mcpref.vpt.gui_frames {args.command}", error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
