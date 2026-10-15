"""Interface of specs/gui-screens.md: GUI textures from the official client jar [J]."""

import argparse
import hashlib
import io
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.screens import ScreenClass, fields, keys, load_screens, read_catalogue
from mcpref.http.retry import get
from mcpref.shards.store import make_dirs, write_file

PREFIX = "assets/minecraft/textures/"
FONT = "font/ascii.png"


@dataclass(frozen=True)
class TexturesConfig:
    """Interface Data: section `textures` of configs/gui-screens.yaml."""

    url: str
    sha1: str
    out_dir: Path
    http_timeout: float
    http_retries: int
    retry_delay: float


def load_textures_config(path: Path) -> TexturesConfig:
    """Interface: the `textures` keys of the catalogue; `out_dir` resolved against REPO."""
    textures = fields(read_catalogue(path)["textures"], keys(TexturesConfig))
    return TexturesConfig(**{**textures, "out_dir": REPO / textures["out_dir"]})


def needed(classes: dict[str, ScreenClass]) -> set[str]:
    """Notation T: every texture named by a blit of any member, and font/ascii.png."""
    blits = (b for c in classes.values() for m in c.members.values() for b in m.blits)
    return {b.texture for b in blits} | {FONT}


def make_parents(out_dir: Path, T: set[str]) -> None:
    """Interface `main`: make_dirs of the parent of every path of T under out_dir."""
    for parent in sorted({(out_dir / t).parent for t in T}):
        make_dirs(parent)


def missing(out_dir: Path, T: set[str]) -> list[str]:
    """Interface `main`: the paths of T without a file out_dir/<path>, sorted."""
    return sorted(t for t in T if not (out_dir / t).is_file())


def download(cfg: TexturesConfig) -> bytes:
    """The body of `url`; at most `http_retries` retries after `retry_delay` each."""
    with httpx.Client() as client:
        response = get(
            client, cfg.url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay
        )
    return response.raise_for_status().content


def check_sha1(cfg: TexturesConfig, body: bytes) -> None:
    """Interface `main`: the sha1 of the jar equals `sha1`."""
    if hashlib.sha1(body).hexdigest() != cfg.sha1:
        raise RuntimeError(f"sha1 of {cfg.url} is not {cfg.sha1}")


def write_members(jar: zipfile.ZipFile, paths: list[str], out_dir: Path) -> None:
    """Every member assets/minecraft/textures/<path> looked up by exact name (first
    absent: KeyError), then each written to out_dir/<path> by specs/shards.md (4)."""
    infos = {path: jar.getinfo(PREFIX + path) for path in paths}
    for path, info in infos.items():
        write_file(out_dir / path, jar.read(info))


def fetch_textures(cfg: TexturesConfig, classes: dict[str, ScreenClass]) -> None:
    """Interface `main`: the missing paths of T from the verified jar, held in memory."""
    T = needed(classes)
    make_parents(cfg.out_dir, T)
    paths = missing(cfg.out_dir, T)
    if not paths:
        return
    body = download(cfg)
    check_sha1(cfg, body)
    with zipfile.ZipFile(io.BytesIO(body), "r") as jar:
        write_members(jar, paths, cfg.out_dir)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.gui.textures <catalogue path>`; exit code."""
    parser = argparse.ArgumentParser(prog="python -m mcpref.gui.textures")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        fetch_textures(load_textures_config(args.config), load_screens(args.config))
    except FAILURES as error:
        return failed("mcpref.gui.textures", error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
