"""Fixtures of the tests of specs/engine.md: tmp/repo as the repository root (`mcpref.REPO`
patched as in tests/pretrain/index_run.py), the working directory tmp/cwd, where the
relative config paths lead nowhere; the simulator's variables of (7) unset and restored;
proxy and TQDM_* variables unset."""

import pytest
from clips.conftest import PROXY_VARS
from pretrain.index_run import repo_root
from progress_bars import tqdm_defaults

from .engine_world import VARS
from .prepare_site import Site


@pytest.fixture
def root(tmp_path, monkeypatch):
    """tmp/repo as the repository root; the working directory tmp/cwd."""
    (tmp_path / "repo").mkdir()
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    with repo_root(tmp_path / "repo"):
        yield tmp_path / "repo"


@pytest.fixture
def clean_env(monkeypatch) -> None:
    """MINESTUDIO_DIR and MINESTUDIO_GPU_RENDER unset; after the test as before it, also
    when the code under test sets them."""
    for var in VARS:
        monkeypatch.setenv(var, "set by the test")
        monkeypatch.delenv(var)
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    tqdm_defaults(monkeypatch)


@pytest.fixture
def site(http, root, clean_env, tmp_path, capsys) -> Site:
    """A fake Maven and engine host and the config of prepare_site.py, listening."""
    return Site(http, root, tmp_path, capsys)
