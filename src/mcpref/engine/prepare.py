"""`prepare` (0)-(6) of specs/engine.md: the engine jar J with LWJGL 3.2.3 arm64."""

import argparse
import hashlib
import shutil
import sys
import zipfile
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import httpx
from tqdm import tqdm

from mcpref.cli.failure import failed
from mcpref.engine.config import JAR, EngineConfig, M, engine_jar, load_config
from mcpref.engine.patch import patch
from mcpref.http.retry import Outcome, download, last_response, retried, retryable
from mcpref.shards.store import make_dirs, part_path, publish

PROG = "mcpref.engine.prepare"


def lwjgl_jars(cfg: EngineConfig) -> list[tuple[str, str, bool]]:
    """Notation M: (url, expected sha256, is natives jar) of the 2|M| jars in order."""
    v, jars = cfg.lwjgl_version, []
    for m in M:
        base, sha = f"{cfg.lwjgl_url}/{m}/{v}/{m}-{v}", cfg.lwjgl_sha256[m]
        jars.append((f"{base}.jar", sha["classes"], False))
        jars.append((f"{base}-natives-linux-arm64.jar", sha["natives"], True))
    return jars


def verify(url: str, actual: str, expected: str) -> None:
    """(2), (3): a sha256 mismatch raises."""
    if actual != expected:
        raise ValueError(f"{url}: sha256 {actual} != {expected}")


def fetch_lwjgl(cfg: EngineConfig) -> list[tuple[bytes, bool]]:
    """(2): each LWJGL jar downloaded into memory and verified; nothing written."""
    jars, listed = [], lwjgl_jars(cfg)
    with tqdm(total=len(listed), desc="lwjgl", unit="file", file=sys.stderr) as bar:
        for url, expected, natives in listed:
            data = download(url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)
            verify(url, hashlib.sha256(data).hexdigest(), expected)
            jars.append((data, natives))
            bar.update()
    return jars


def attempt(
    client: httpx.Client, cfg: EngineConfig, Z: Path, bar: tqdm, hashes: list
) -> Outcome:
    """(3): one attempt; a transport error or a 5xx response is returned, any other
    non-200 raises; on 200 the body is written to Z, hashed and counted."""
    bar.reset()
    sha = hashlib.sha256()
    hashes.append(sha)
    timeout = httpx.Timeout(cfg.http_timeout)
    try:
        with client.stream("GET", cfg.engine_url, timeout=timeout) as response:
            if response.is_server_error:
                return response
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"{cfg.engine_url}: HTTP {response.status_code}")
            with Z.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
                    sha.update(chunk)
                    bar.update(len(chunk))
            return response
    except httpx.TransportError as error:
        return error


def fetch_engine(cfg: EngineConfig, Z: Path) -> None:
    """(3): Z fetched with retries; a last transport error or 5xx raises; the last
    attempt's sha256 verified."""
    headers = {"Accept-Encoding": "identity"}
    hashes: list = []
    bar = tqdm(
        total=cfg.engine_size,
        desc="engine.zip",
        unit="B",
        unit_scale=True,
        file=sys.stderr,
    )
    with bar, httpx.Client(follow_redirects=True, headers=headers) as client:
        call = partial(attempt, client, cfg, Z, bar, hashes)
        outcome = retried(call, retryable, cfg.http_retries, cfg.retry_delay)
    last_response(outcome).raise_for_status()
    verify(cfg.engine_url, hashes[-1].hexdigest(), cfg.engine_sha256)


def prepare(cfg: EngineConfig) -> None:
    """(0)-(6): D cleared; J built unless it exists; D removed."""
    D, J = cfg.minestudio_dir / "download", engine_jar(cfg)
    Z, S = D / "engine.zip", D / JAR
    if D.exists():
        shutil.rmtree(D)
    if J.exists():
        print("engine: already done", file=sys.stderr)
        return
    jars = fetch_lwjgl(cfg)
    make_dirs(D)
    fetch_engine(cfg, Z)
    with zipfile.ZipFile(Z) as z:
        z.extract(JAR, D)
    make_dirs(J.parent)
    patch(S, jars, part_path(J))
    publish(part_path(J), J)
    shutil.rmtree(D)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.engine.prepare <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        prepare(load_config(args.config))
    except Exception as error:  # noqa: BLE001 (specs/engine.md: any Exception)
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
