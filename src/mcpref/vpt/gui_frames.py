"""Interface of specs/vpt-gui-frames.md: random VPT frames with an open GUI."""

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.shards.store import complete_shards, make_dirs
from mcpref.vpt.config import GuiFramesConfig, check_outputs, load_config, store_run
from mcpref.vpt.downloads import fetch, index_path, store_indices
from mcpref.vpt.episodes import (
    Fetch,
    eligible_frames,
    index,
    options_ok,
    version_ok,
)
from mcpref.vpt.ffmpeg import extract, ffmpeg_command, frame_time
from mcpref.vpt.plan import plan, read_shard
from mcpref.vpt.verify import verify
from mcpref.vpt.yuv import yuv420p_to_rgb

__all__ = [
    "GuiFramesConfig",
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


def submit_shard(
    cfg: GuiFramesConfig,
    threads: ThreadPoolExecutor,
    shard: dict,
    basedirs: dict[str, str],
    out: Path,
) -> list[Future]:
    """(c): the extraction of each frame of a plan shard without <id>.yuv; its video URL
    = the basedir of its subset + its relpath."""
    jobs = []
    for f in shard["frames"]:
        path, url = out / f"{f['id']}.yuv", basedirs[f["subset"]] + f["relpath"]
        if not path.exists():
            jobs.append(threads.submit(extract, cfg, url, f["frame"], path))
    return jobs


def extract_all(
    cfg: GuiFramesConfig, plan_dir: Path, basedirs: dict[str, str], out: Path, n: int
) -> int:
    """(c), (d): the n planned frames without <id>.yuv, shard by shard, on `workers`
    threads, an error cancelling the queued ones; bar `extract`; the failed count."""
    make_dirs(out)
    total, n_failed = n - sum(1 for _ in out.glob("*.yuv")), 0
    if total == 0:
        print(f"extract: {'already done' if n else 'nothing to do'}", file=sys.stderr)
        return n_failed
    with tqdm(total=total, desc="extract", unit="frame") as bar:
        bar.set_postfix_str(f"{n_failed} failed")
        threads = ThreadPoolExecutor(cfg.workers)
        try:
            for k in range(complete_shards(plan_dir)):
                shard = read_shard(plan_dir, k)
                jobs = submit_shard(cfg, threads, shard, basedirs, out)
                for job in as_completed(jobs):
                    n_failed += not job.result()
                    bar.update(1)
                    bar.set_postfix_str(f"{n_failed} failed")
        finally:
            threads.shutdown(cancel_futures=True)
    return n_failed


def stored_fetch(client: httpx.Client, cfg: GuiFramesConfig, indices: Path) -> Fetch:
    """(b): the index_urls answered with the files of (a), every other URL by (f)."""
    files = {url: index_path(indices, s) for s, url in cfg.index_urls.items()}
    return lambda url: (
        files[url].read_bytes() if url in files else fetch(client, cfg, url)
    )


def index_basedirs(cfg: GuiFramesConfig, indices: Path) -> dict[str, str]:
    """(c): the basedir of each indices/<subset>.json, read one at a time."""
    return {s: index(index_path(indices, s).read_bytes())[0] for s in cfg.index_urls}


def run(cfg: GuiFramesConfig, client: httpx.Client) -> None:
    """Behaviour (d), run.json, (a)-(c); the files of (b) by the ranged fetch (f)."""
    out, config = REPO / cfg.out_dir, dataclasses.asdict(cfg)
    check_outputs(out / "run.json", config)
    store_run(out / "run.json", {"config": config})
    indices, plan_dir = out / "indices", out / "plan"
    store_indices(client, cfg, indices)
    n, reasons = plan(cfg, stored_fetch(client, cfg, indices), plan_dir)
    basedirs = index_basedirs(cfg, indices)
    n_failed = extract_all(cfg, plan_dir, basedirs, out / "frames", n)
    frames = {k: v for k, v in (("ok", n - n_failed), ("failed", n_failed)) if v}
    counts = {"episodes": reasons, "frames": frames}
    store_run(out / "run.json", {"config": config} | counts)


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
