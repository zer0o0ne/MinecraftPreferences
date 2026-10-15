"""Fixtures of the tests of specs/perception-data.md: a Site (collect_site.py) per test."""

import os
from pathlib import Path

import pytest
from progress_bars import tqdm_defaults

from .collect_site import Site
from .collect_world import EPISODES

PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
              "all_proxy", "NO_PROXY", "no_proxy")  # fmt: skip


@pytest.fixture
def make_site(http, tmp_path, repo, monkeypatch):
    """Site factory, listening unless `listen=False`. The cwd is chosen so that paths
    resolved against it land elsewhere; proxy and TQDM_* variables are unset."""
    rel = Path(os.path.relpath(tmp_path / "out", repo))
    cwd = tmp_path.joinpath(*["cwd"] * (rel.parts.count("..") + 1))
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    tqdm_defaults(monkeypatch)

    def make(listen: bool = True, episodes: dict = EPISODES) -> Site:
        site = Site(http, tmp_path, repo, episodes)
        if listen:
            site.server.listen()
        return site

    return make


@pytest.fixture
def site(make_site) -> Site:
    return make_site()
