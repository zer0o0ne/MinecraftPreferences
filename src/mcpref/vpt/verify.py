"""(e) of specs/vpt-gui-frames.md: frames extracted as in (c) against a sequential decode."""

import sys
import tempfile
from contextlib import closing
from pathlib import Path

import httpx
from tqdm import tqdm

from mcpref.vpt.config import GuiFramesConfig
from mcpref.vpt.downloads import download, download_indices
from mcpref.vpt.episodes import index
from mcpref.vpt.ffmpeg import decode, extract


def video_url(client: httpx.Client, cfg: GuiFramesConfig) -> str:
    """(e): verify_relpath under the basedir of the first index that lists it."""
    with closing(download_indices(client, cfg)) as responses:
        for _, response in responses:
            basedir, relpaths = index(response.content)
            if cfg.verify_relpath in relpaths:
                return basedir + cfg.verify_relpath
    raise RuntimeError(f"no index lists {cfg.verify_relpath}")


def check_frames(
    cfg: GuiFramesConfig, url: str, tmp: Path, decoded: dict[int, bytes]
) -> None:
    """(e): each verify frame, extracted as in (c), equals the decode; bar `verify`."""
    if not cfg.verify_frames:
        print("verify: nothing to do", file=sys.stderr)
        return
    with tqdm(total=len(cfg.verify_frames), desc="verify", unit="frame") as bar:
        for i in cfg.verify_frames:
            path = tmp / f"{i}.yuv"
            if not extract(cfg, url, i, path) or path.read_bytes() != decoded.get(i):
                raise RuntimeError(f"frame {i} of {url} is not the sequential decode")
            bar.update(1)


def verify(cfg: GuiFramesConfig, client: httpx.Client) -> None:
    """(e): every verify frame by ffmpeg_command equals the sequential decode."""
    url = video_url(client, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "video.mp4"
        video.write_bytes(download(client, cfg, url).content)
        check_frames(cfg, url, Path(tmp), decode(cfg, video))
