"""(a), (e), (f) of specs/vpt-gui-frames.md: indices, the verify video, ranged fetch;
also of specs/perception-data.md (access fields `Access`)."""

import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path

import httpx
from tqdm import tqdm

from mcpref.http.retry import Method, request, retried
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


def pwrite(fd: int, data: bytes, offset: int) -> int:
    """Write all of data at offset without moving the shared file position."""
    view = memoryview(data)
    while view:
        n = os.pwrite(fd, view, offset)
        view, offset = view[n:], offset + n
    return offset


def stream_part(
    client: httpx.Client, cfg: Access, url: str, fd: int, first: int, last: int
) -> tuple[httpx.Response, int] | httpx.TransportError:
    """One complete streaming attempt, including response-body iteration."""
    try:
        with client.stream(
            "GET",
            url,
            headers={"Range": f"bytes={first}-{last}"},
            timeout=httpx.Timeout(cfg.http_timeout),
            follow_redirects=False,
        ) as response:
            n, offset = 0, first
            for chunk in response.iter_raw(chunk_size=last - first + 1):
                n += len(chunk)
                if response.status_code == 206:
                    offset = pwrite(fd, chunk, offset)
            return response, n
    except httpx.TransportError as error:
        return error


def part_to_path(
    client: httpx.Client, cfg: Access, url: str, fd: int, first: int, last: int
) -> None:
    """Write one range, retrying transport errors and HTTP 5xx across whole attempts."""
    call = partial(stream_part, client, cfg, url, fd, first, last)
    again = lambda outcome: (
        isinstance(outcome, httpx.TransportError) or outcome[0].is_server_error
    )
    outcome = retried(call, again, cfg.http_retries, cfg.retry_delay)
    if isinstance(outcome, httpx.TransportError):
        raise outcome
    response, n = outcome
    if response.status_code != 206 or n != last - first + 1:
        status = response.status_code
        raise RuntimeError(f"GET {url} bytes={first}-{last}: HTTP {status}, {n} bytes")


def range_executor(cfg: Access) -> ThreadPoolExecutor:
    """The bounded executor shared by a pipeline's video fetches."""
    return ThreadPoolExecutor(cfg.fetch_streams)


def fetch_to_path(
    client: httpx.Client,
    cfg: Access,
    url: str,
    path: Path,
    threads: ThreadPoolExecutor,
) -> bool:
    """Video-specific (f): stream parts on the caller-owned executor; False for a 404."""
    L = size(client, cfg, url)
    if L is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".part")
    try:
        with staged.open("wb") as output:
            if L:
                jobs = [
                    threads.submit(part_to_path, client, cfg, url, output.fileno(), *ab)
                    for ab in parts(L, cfg.fetch_streams)
                ]
                wait(jobs)
                for job in jobs:
                    job.result()
        staged.replace(path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return True
