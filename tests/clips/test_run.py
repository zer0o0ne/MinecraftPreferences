"""`main` of src/mcpref/clips/run.py against specs/perception-data.md (Interface, revision of
2026-10-20; (9); Progress bars) and specs/gui-classifier.md (Interface): with a fake gui-masks
output (classifier_world.py) and a fake `collect`, gamma and gamma_id of the classifier
passed to `collect`; exit 1 with the one-line message `mcpref.clips.run: <type>: <message>`
as the last line of stderr on any exception; the arguments parsed by argparse."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from classifier_world import elsewhere, gamma_id
from progress_bars import Bar, assert_aborted, screen, tqdm_defaults

from mcpref.clips import run
from mcpref.clips.collect import load_config

from .run_world import MARK, PREFIX, FakeCollect, H, W, World, patch_collect


@pytest.fixture
def world(tmp_path, repo, monkeypatch):
    """A World; the working directory elsewhere; tqdm with its defaults."""
    monkeypatch.chdir(elsewhere(tmp_path, repo))
    tqdm_defaults(monkeypatch)
    return World(tmp_path, repo)


def main(monkeypatch, capfd, world: World, error=None, argv=None):
    """main on the world's config with a fake collect: (exit code, stdout, stderr, fake)."""
    fake = FakeCollect(world.frames, error)
    patch_collect(monkeypatch, fake)
    code = run.main([world.path] if argv is None else argv)
    out, err = capfd.readouterr()
    return code, out, err, fake


def test_gamma_and_its_id_passed_to_collect(world, monkeypatch, capfd):
    """load_config, load_classifier of the classifier config, then collect(cfg, gamma,
    gamma_id); exit 0; the bar `classifier` drawn first, closed before collect."""
    code, out, err, fake = main(monkeypatch, capfd, world)
    assert code == 0 and out == "", err
    assert len(fake.calls) == 1
    cfg, labels, gid = fake.calls[0]
    assert cfg == load_config(Path(world.path))
    assert labels == ["inv", "chest", "other"]
    assert gid == gamma_id(world.masks, "macro_recall")
    assert screen(err) == [Bar("classifier", 3, 3, "shard"), MARK]


def no_run(world: World) -> str:
    (world.masks.out / "run.json").unlink()
    return f"{world.masks.masks_dir} is not complete"


def other_catalogue(world: World) -> str:
    world.masks.catalogue.write_text(world.masks.catalogue.read_text() + "# edited\n")
    return f"{world.masks.masks_dir} was built from another catalogue than {world.masks.screens}"  # fmt: skip


def empty_mask(world: World) -> str:
    """chest (shard 1): the bar has advanced by one shard."""
    t = world.masks.classes["chest"]["main"][0]
    np.savez(
        world.masks.out / "templates/catalogue/000001/pairs.npz",
        **{"d/main/T": t, "d/main/M": np.zeros((H, W), bool)},
    )
    return f"{world.masks.masks_dir}: the mask of d/main of chest is empty"


@pytest.mark.parametrize("breaking", [no_run, other_catalogue, empty_mask])
def test_failed_check_of_the_classifier(world, monkeypatch, capfd, breaking):
    """A failed check of gui-classifier (L): exit 1, its message (paths as written in the
    configs) the last line of stderr, below the bar drawn before; collect never called."""
    message = breaking(world)
    code, out, err, fake = main(monkeypatch, capfd, world)
    above = assert_aborted(code, err)
    assert out == "" and fake.calls == []
    assert err.splitlines()[-1] == f"{PREFIX}RuntimeError: {message}"
    bars = [Bar("classifier", 1, 3, "shard")] if breaking is empty_mask else []
    assert above == bars


@pytest.mark.parametrize(
    "error, line",
    [
        (
            ZeroDivisionError("fake collect failed"),
            "ZeroDivisionError: fake collect failed",
        ),
        (RuntimeError("fake abort"), "RuntimeError: fake abort"),
    ],
)
def test_exception_of_collect(world, monkeypatch, capfd, error, line):
    """Any Exception of collect, inside FAILURES or not: exit 1, the message last."""
    code, _, err, fake = main(monkeypatch, capfd, world, error)
    above = assert_aborted(code, err)
    assert err.splitlines()[-1] == PREFIX + line and len(fake.calls) == 1
    assert above == [Bar("classifier", 3, 3, "shard"), MARK]


def test_other_base_exceptions_propagate(world, monkeypatch, capfd):
    """Only an Exception gives exit 1: a KeyboardInterrupt goes through."""
    with pytest.raises(KeyboardInterrupt):
        main(monkeypatch, capfd, world, KeyboardInterrupt())


@pytest.mark.parametrize(
    "change, start",
    [
        ({"classifier_config": 1}, "ValueError: "),
        ({"classifier_config": None}, "ValueError: "),
        ({"classifier_config": "configs/no-such-classifier.yaml"}, ""),
    ],
)
def test_config_errors(world, monkeypatch, capfd, change, start):
    """The config field (ValueError on a non-string) and a missing classifier config: exit
    1 with the message, before any bar; collect never called."""
    world.path = world.config(**change)
    code, _, err, fake = main(monkeypatch, capfd, world)
    assert assert_aborted(code, err) == [] and fake.calls == []
    assert err.startswith(PREFIX + start) and err.count("\n") == 1


def test_classifier_config_error(world, monkeypatch, capfd):
    """The classifier's own config rejected: its ValueError is the message."""
    doc = {"masks_dir": world.masks.masks_dir, "criterion": "f1"}
    world.classifier.write_text(yaml.safe_dump(doc))
    code, _, err, fake = main(monkeypatch, capfd, world)
    assert assert_aborted(code, err) == [] and fake.calls == []
    want = "ValueError: criterion 'f1' is not one of ['agreement', 'macro_recall']"
    assert err == PREFIX + want + "\n"


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1", "a.yaml"]])
def test_wrong_argument_list_exits_2(world, monkeypatch, capfd, argv):
    """argparse: one positional config path; otherwise exit 2 with the usage."""
    with pytest.raises(SystemExit) as info:
        main(monkeypatch, capfd, world, argv=argv)
    assert info.value.code == 2
    out, err = capfd.readouterr()
    assert out == "" and err.startswith("usage: ") and "error: " in err


def test_run_as_module(tmp_path, repo):
    """`python -m mcpref.clips.run <config path>`: exit 1 with the message on a missing
    config; exit 2 with the usage without one."""
    cwd = elsewhere(tmp_path, repo)
    argv = [sys.executable, "-m", "mcpref.clips.run"]
    missing = subprocess.run(
        [*argv, str(tmp_path / "missing.yaml")], cwd=cwd, capture_output=True, text=True,
        timeout=60, check=False,
    )  # fmt: skip
    assert missing.returncode == 1 and missing.stdout == ""
    assert missing.stderr.startswith(PREFIX) and missing.stderr.count("\n") == 1
    bare = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60, check=False)  # fmt: skip
    assert bare.returncode == 2 and bare.stderr.startswith("usage: ")
