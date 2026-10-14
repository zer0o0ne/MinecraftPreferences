"""(a), (b), (e) of specs/vpt-gui-frames.md: indices, episode files and the verify video."""

from collections.abc import Iterator

import httpx
from tqdm import tqdm

from mcpref.http.retry import get
from mcpref.vpt.config import GuiFramesConfig


def download(client: httpx.Client, cfg: GuiFramesConfig, url: str) -> httpx.Response:
    """(a), (e): a GET that must succeed after the retries."""
    response = get(client, url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)
    return response.raise_for_status()


def download_indices(
    client: httpx.Client, cfg: GuiFramesConfig
) -> Iterator[tuple[str, httpx.Response]]:
    """(a), (e): each subset's index downloaded, in config order; bar `indices`."""
    with tqdm(total=len(cfg.index_urls), desc="indices", unit="file") as bar:
        for s, url in cfg.index_urls.items():
            response = download(client, cfg, url)
            bar.update(1)
            yield s, response


def fetch(client: httpx.Client, cfg: GuiFramesConfig, url: str) -> bytes | None:
    """(b): the body, or None for HTTP 404; other failures after the retries abort."""
    response = get(client, url, cfg.http_timeout, cfg.http_retries, cfg.retry_delay)
    return None if response.status_code == 404 else response.raise_for_status().content
