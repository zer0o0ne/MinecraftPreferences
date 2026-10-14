"""The runs of `main` of specs/gui-masks.md on the scenarios of masks_run.py, shared by the
modules checking its outputs."""

import pytest
from masks_run import SCENARIOS, config_of, expected_of, frames_of, run_main


@pytest.fixture(scope="session", params=list(SCENARIOS))
def world(request, tmp_path_factory, repo):
    """(name, the run of main, the oracle's run); read-only."""
    name = request.param
    run = run_main(
        tmp_path_factory.mktemp(name), repo, config_of(name), frames_of(name)
    )
    return name, run, expected_of(name)
