"""The runs of `main` of specs/gui-masks.md on the scenarios of masks_run.py, shared by the
modules checking its outputs."""

import functools

import pytest
from masks_run import SCENARIOS, config_of, expected_of, frames_of, run_main


@pytest.fixture(scope="session")
def run_of(tmp_path_factory, repo):
    """name -> the run of main on scenario `name`, made once per session; read-only."""

    @functools.cache
    def run(name: str):
        root = tmp_path_factory.mktemp(name)
        return run_main(root, repo, config_of(name), frames_of(name))

    return run


@pytest.fixture(scope="session", params=list(SCENARIOS))
def world(request, run_of):
    """(name, the run of main, the oracle's run); read-only."""
    return request.param, run_of(request.param), expected_of(request.param)
