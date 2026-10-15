"""(a), (e), (f) of specs/vpt-gui-frames.md: indices, the verify video, ranged fetch;
also of specs/perception-data.md (access fields `Access`)."""

import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import httpx
from tqdm import tqdm

from mcpref.http.retry import Method, request
from mcpref.shards.store import make_dirs, write_file
from mcpref.vpt.config import Access, GuiFramesConfig


def send(method: Method, cfg: Access, url: str) -> httpx.Response:
    """Retries: one request of (a), (e) or (f) with its own retries."""
    return request(method, url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)


def download(client: httpx.Client, cfg: Access, url: str) -> httpx.Response:
    """(a), (e): one GET that must succeed after the retries."""
    return send(client.get, cfg, url).raise_for_status()


def download_indices(
    client: httpx.Client, cfg: Access
) -> Iterator[tuple[str, httpx.Response]]:
    """(e): each subset's index downloaded, in config order; bar `indices`."""
    with tqdm(total=len(cfg.index_urls), desc="indices", unit="file") as bar:
        for s, url in cfg.index_urls.items():
            response = download(client, cfg, url)
            bar.update(1)
            yield s, response


def index_path(directory: Path, subset: str) -> Path:
    """Interface Produces: indices/<subset>.json."""
    return directory / f"{subset}.json"


def store_indices(client: httpx.Client, cfg: GuiFramesConfig, directory: Path) -> None:
    """(a): each index without a stored file downloaded with one GET, in config order,
    and written atomically (specs/shards.md (4)); bar `indices`."""
    make_dirs(directory)
    missing = [s for s in cfg.index_urls if not index_path(directory, s).exists()]
    if not missing:
        print("indices: already done", file=sys.stderr)
        return
    with tqdm(total=len(missing), desc="indices", unit="file") as bar:
        for s in missing:
            body = download(client, cfg, cfg.index_urls[s]).content
            write_file(index_path(directory, s), body)
            bar.update(1)


def size(client: httpx.Client, cfg: Access, url: str) -> int | None:
    """(f): L, the Content-Length of a HEAD answering 200; None for 404; else abort."""
    response = send(client.head, cfg, url)
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"HEAD {url}: HTTP {response.status_code}")
    if "Content-Length" not in response.headers:
        raise RuntimeError(f"HEAD {url}: HTTP 200 without Content-Length")
    return int(response.headers["Content-Length"])


def parts(L: int, q: int) -> list[tuple[int, int]]:
    """(f): part j = bytes [jp, min((j+1)p, L) - 1], j < ceil(L/p), p = ceil(L/q); L > 0."""
    p = -(-L // q)
    return [(first, min(first + p, L) - 1) for first in range(0, L, p)]


def part(client: httpx.Client, cfg: Access, url: str, first: int, last: int) -> bytes:
    """(f): one range GET, which must answer 206 with exactly last - first + 1 bytes."""
    ranged = partial(client.get, headers={"Range": f"bytes={first}-{last}"})
    response = send(ranged, cfg, url)
    status, n = response.status_code, len(response.content)
    if status != 206 or n != last - first + 1:
        raise RuntimeError(f"GET {url} bytes={first}-{last}: HTTP {status}, {n} bytes")
    return response.content


def fetch(client: httpx.Client, cfg: Access, url: str) -> bytes | None:
    """(f): the file, its parts started at once on q threads and concatenated; None if
    missing; failing parts abort once all have finished, with the lowest failing j."""
    L = size(client, cfg, url)
    if L is None:
        return None
    if L == 0:
        return b""
    q = cfg.fetch_streams
    with ThreadPoolExecutor(q) as threads:
        jobs = [threads.submit(part, client, cfg, url, *ab) for ab in parts(L, q)]
    return b"".join(job.result() for job in jobs)
