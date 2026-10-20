"""The runs of `main` of specs/gui-masks.md on the scenarios of masks_run.py, shared by the
modules checking its outputs; the working directory of the tests of specs/gui-classifier.md."""

import functools

import pytest
from classifier_world import elsewhere
from masks_run import SCENARIOS, config_of, expected_of, frames_of, run_main
from progress_bars import tqdm_defaults


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


@pytest.fixture
def away(tmp_path, repo, monkeypatch) -> None:
    """A working directory from which the repository-relative paths of files under tmp_path
    lead nowhere (specs/gui-classifier.md resolves them against the repository root);
    tqdm with its defaults."""
    monkeypatch.chdir(elsewhere(tmp_path, repo))
    tqdm_defaults(monkeypatch)
