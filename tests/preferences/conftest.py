"""Fixtures of the tests of specs/vlm-labels.md: a Site (labels_run.py) per test."""

import pytest

from .config_files import TINY
from .labels_run import Site
from .labels_world import HAND


@pytest.fixture
def make_site(tmp_path):
    """Site factory: the shards of Q_c, the tiny config with `changes`, written unless
    `write=False`; `run`: {Q_c}/run.json (default: complete, its counts)."""

    def make(shards: list = HAND, write: bool = True, run=None, **changes) -> Site:
        site = Site(tmp_path, shards, {**TINY, **changes})
        if write:
            site.write(run)
        return site

    return make
