"""Interface of specs/gui-screens.md: GUI textures from the official client jar [J]."""

import argparse
import hashlib
import sys
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from mcpref import REPO
from mcpref.gui.screens import ScreenClass, fields, keys, load_screens, read_catalogue

PREFIX = "assets/minecraft/textures/"
FONT = "font/ascii.png"

Outcome = httpx.Response | httpx.TransportError
FAILURES = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    yaml.YAMLError,
    httpx.HTTPError,
    zipfile.BadZipFile,
)


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
    """Interface `main`: every texture named by a blit, and font/ascii.png."""
    blits = (b for c in classes.values() for m in c.members.values() for b in m.blits)
    return {b.texture for b in blits} | {FONT}


def attempt(client: httpx.Client, cfg: TexturesConfig) -> Outcome:
    """One GET of `url`: `http_timeout` per phase, no redirects; returns a transport error."""
    timeout = httpx.Timeout(cfg.http_timeout)
    try:
        return client.get(cfg.url, timeout=timeout, follow_redirects=False)
    except httpx.TransportError as error:
        return error


def retryable(outcome: Outcome) -> bool:
    """Interface `main`: any `httpx.TransportError` or HTTP 5xx is retried."""
    return isinstance(outcome, httpx.TransportError) or outcome.is_server_error


def download(cfg: TexturesConfig) -> bytes:
    """The body of `url`; at most `http_retries` retries after `retry_delay` each."""
    with httpx.Client() as client:
        outcome = attempt(client, cfg)
        for _ in range(cfg.http_retries):
            if not retryable(outcome):
                break
            time.sleep(cfg.retry_delay)
            outcome = attempt(client, cfg)
    if isinstance(outcome, httpx.TransportError):
        raise outcome
    return outcome.raise_for_status().content


def wanted(zf: zipfile.ZipFile) -> list[str]:
    """File members under assets/minecraft/textures/gui/, and font/ascii.png."""
    gui = [i for i in zf.infolist() if i.filename.startswith(PREFIX + "gui/")]
    return [i.filename for i in gui if not i.is_dir()] + [PREFIX + FONT]


def extract(jar: Path, out_dir: Path) -> None:
    """Each wanted member to out_dir/<path relative to assets/minecraft/textures/>."""
    with zipfile.ZipFile(jar) as zf:
        for name in wanted(zf):
            target = out_dir / name[len(PREFIX) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as member:
                target.write_bytes(member.read())


def fetch_textures(cfg: TexturesConfig, classes: dict[str, ScreenClass]) -> None:
    """Interface `main`: download, sha1 check, extraction; nothing if all are present."""
    if all((cfg.out_dir / t).exists() for t in needed(classes)):
        return
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    jar = cfg.out_dir / "client.jar.part"
    body = download(cfg)
    jar.write_bytes(body)
    if hashlib.sha1(body).hexdigest() != cfg.sha1:
        jar.unlink()
        raise RuntimeError(f"sha1 of {cfg.url} is not {cfg.sha1}")
    extract(jar, cfg.out_dir)
    jar.unlink()


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.gui.textures <catalogue path>`; exit code."""
    parser = argparse.ArgumentParser(prog="python -m mcpref.gui.textures")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        fetch_textures(load_textures_config(args.config), load_screens(args.config))
    except FAILURES as error:
        cause = " ".join(f"{type(error).__name__}: {error}".split())
        print(f"mcpref.gui.textures: {cause}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
