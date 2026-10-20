"""Fixtures of the tests of specs/train-index.md: a Site (index_run.py) per test."""

import pytest

from .index_run import Site
from .index_world import CONFIG, HAND, World


@pytest.fixture
def make_site(tmp_path):
    """Site factory: the world, the config with `changes`, written unless `write=False`;
    c_run, q_run: run.json of C and Q_c (Site.write)."""

    def make(
        world: World = HAND, write: bool = True, c_run=None, q_run=None, **changes
    ) -> Site:
        site = Site(tmp_path, world, {**CONFIG, **changes})
        if write:
            site.write(c_run, q_run)
        return site

    return make
