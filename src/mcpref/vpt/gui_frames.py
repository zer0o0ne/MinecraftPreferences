"""Interface of specs/vpt-gui-frames.md: random VPT frames with an open GUI."""

import argparse
import dataclasses
import json
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.vpt.config import GuiFramesConfig, check_rerun, load_config, write_run
from mcpref.vpt.downloads import download_indices, fetch
from mcpref.vpt.episodes import (
    EpisodeRecord,
    PlannedFrame,
    eligible_frames,
    index,
    options_ok,
    plan,
    version_ok,
)
from mcpref.vpt.ffmpeg import extract, ffmpeg_command, frame_time
from mcpref.vpt.verify import verify
from mcpref.vpt.yuv import yuv420p_to_rgb

__all__ = [
    "EpisodeRecord",
    "GuiFramesConfig",
    "PlannedFrame",
    "eligible_frames",
    "fetch",
    "ffmpeg_command",
    "frame_time",
    "load_config",
    "main",
    "options_ok",
    "plan",
    "version_ok",
    "yuv420p_to_rgb",
]


def extract_all(
    cfg: GuiFramesConfig, frames: list[PlannedFrame], urls: list[str], out: Path
) -> list[str]:
    """(c), (d): statuses in plan order, `workers` threads; existing <id>.yuv skipped;
    bar `extract`, enclosing the pool, whose queued frames an error cancels."""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{f.id}.yuv" for f in frames]
    statuses = ["ok"] * len(frames)
    todo = [k for k, path in enumerate(paths) if not path.exists()]
    if not todo:
        done = "already done" if frames else "nothing to do"
        print(f"extract: {done}", file=sys.stderr)
        return statuses
    with tqdm(total=len(todo), desc="extract", unit="frame") as bar:
        bar.set_postfix_str(f"{statuses.count('failed')} failed")
        threads = ThreadPoolExecutor(cfg.workers)
        try:
            jobs = {
                threads.submit(extract, cfg, urls[k], frames[k].frame, paths[k]): k
                for k in todo
            }
            for job in as_completed(jobs):
                statuses[jobs[job]] = "ok" if job.result() else "failed"
                bar.update(1)
                bar.set_postfix_str(f"{statuses.count('failed')} failed")
        finally:
            threads.shutdown(cancel_futures=True)
    return statuses


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
    """Behaviour (a)-(d), the files of (b) by the ranged fetch (f)."""
    out = REPO / cfg.out_dir
    indices = dict(download_indices(client, cfg))
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
