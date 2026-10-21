"""Fixtures of the tests of specs/preference-pool.md: a PoolSite (pool_site.py) per test,
under tmp/repo, the repository root of the test (`mcpref.REPO` patched there); the working
directory tmp/cwd, where the relative config paths lead nowhere; proxy and TQDM_*
variables unset."""

import pytest
from clips.conftest import PROXY_VARS
from pretrain.index_run import repo_root
from progress_bars import tqdm_defaults

from .pool_site import PoolSite
from .pool_world import EPISODES


@pytest.fixture
def root(tmp_path, monkeypatch):
    """tmp/repo as the repository root; the working directory tmp/cwd."""
    (tmp_path / "repo").mkdir()
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    with repo_root(tmp_path / "repo"):
        yield tmp_path / "repo"


@pytest.fixture
def make_site(http, root, monkeypatch):
    """PoolSite factory, listening unless `listen=False`."""
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    tqdm_defaults(monkeypatch)

    def make(listen: bool = True, episodes: dict = EPISODES) -> PoolSite:
        site = PoolSite(http, root, episodes)
        if listen:
            site.server.listen()
        return site

    return make


@pytest.fixture
def site(make_site) -> PoolSite:
    return make_site()
