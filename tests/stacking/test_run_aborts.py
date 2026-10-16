"""`main` of src/mcpref/stacking/run.py against specs/gui-stack.md (Behaviour (R): the checks,
their messages and order; Interface: exit 1 with a one-line message as the last line of
stderr)."""

import json
import shutil

import pytest
import yaml
from progress_bars import assert_aborted

from mcpref.stacking.run import main

from .stack_files import EMPTY, Shard, World, run, run_main, write_masks
from .stack_world import CONFIG, LABELS, NAN_WORLD, with_pred

TEST = [with_pred(["te0"], ["beta"], [True], [[]], 4)]
C_ZERO = World([], LABELS, {"train": [Shard(["tr0"], ["alpha"], [True], [[]])],
                            "validation": [Shard(["va0"], ["beta"], [False], [[]])],
                            "test": TEST})  # fmt: skip
NO_S = {"train": [EMPTY], "validation": [EMPTY]}
S_EMPTY = World(["a/alpha"], LABELS,
                {**NO_S, "test": [with_pred(["te0"], ["beta"], [True], [[0.5]], 4)]})  # fmt: skip
BOTH = World([], LABELS, {**NO_S, "test": TEST})


def last_line(r) -> str:
    assert_aborted(r.code, r.err)
    return r.err.splitlines()[-1]


def written(root, key: str) -> str:
    """A path of the config as written in it."""
    return yaml.safe_load((root / "gui-stack.yaml").read_text())[key]


def failed(message: str) -> str:
    return f"mcpref.stacking.run: RuntimeError: {message}"


@pytest.mark.parametrize("state", ["missing", "complete false", "without complete"])
def test_masks_not_complete(tmp_path, repo, state):
    """{masks_dir}/run.json missing or without "complete": true: abort before writing
    {out_dir}/run.json."""
    write_masks(tmp_path / "masks", NAN_WORLD)
    path = tmp_path / "masks/run.json"
    if state == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps({"complete": False} if "false" in state else {}))
    r = run(tmp_path, repo, CONFIG)
    assert last_line(r) == failed(f"{written(tmp_path, 'masks_dir')} is not complete")
    assert not (tmp_path / "out/run.json").exists()


CHANGES = {
    "config": ["config"],
    "masks": ["masks"],
    "config and masks": ["config", "masks"],
}


@pytest.mark.parametrize("changed", CHANGES.values(), ids=CHANGES)
def test_changed_since_the_run_started(tmp_path, repo, changed):
    """A run stopped after (R) (its first features read failing); the config file or
    {masks_dir}/run.json then changed: the rerun aborts, naming the first in the order
    config, masks."""
    first = run_main(tmp_path, repo, NAN_WORLD, CONFIG, fail=lambda log: True)
    assert_aborted(first.code, first.err)
    assert (tmp_path / "out/run.json").exists()
    config = {**CONFIG, "epochs": 3} if "config" in changed else CONFIG
    if "masks" in changed:
        path = tmp_path / "masks/run.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "note": 1}))
    r = run(tmp_path, repo, config)
    out = written(tmp_path, "out_dir")
    assert last_line(r) == failed(
        f"{changed[0]} changed since the run started; delete {out}"
    )


CASES = {"C = 0": (C_ZERO, "no column"), "S empty": (S_EMPTY, "no training frame"),
         "C = 0 and S empty": (BOTH, "no column")}  # fmt: skip


@pytest.mark.parametrize("world, message", CASES.values(), ids=CASES)
def test_no_column_before_no_training_frame(tmp_path, repo, world, message):
    r = run_main(tmp_path, repo, world, CONFIG)
    assert last_line(r) == failed(message)


def break_masks(masks, how: str) -> None:
    if how == "header.json missing":
        (masks / "features/header.json").unlink()
    elif how == "features.npz missing":
        (masks / "features/validation/000001/features.npz").unlink()
    elif how == "foreign entry":
        (masks / "features/test/notes.txt").write_text("")
    else:
        shutil.rmtree(masks / "features/train/000001")


BROKEN = [
    "header.json missing",
    "features.npz missing",
    "foreign entry",
    "shard missing",
]


@pytest.mark.parametrize("how", BROKEN)
def test_unreadable_inputs(tmp_path, repo, how):
    """An unreadable file (OSError) or `complete_shards` raising: exit 1, one line."""
    write_masks(tmp_path / "masks", NAN_WORLD)
    break_masks(tmp_path / "masks", how)
    r = run(tmp_path, repo, CONFIG)
    last_line(r)


def test_config_violating_a_constraint(tmp_path, repo):
    r = run_main(tmp_path, repo, NAN_WORLD, {**CONFIG, "batch": 0})
    assert last_line(r).startswith("mcpref.stacking.run: ValueError: ")


def test_missing_config(tmp_path, capsys):
    code = main([str(tmp_path / "absent.yaml")])
    assert_aborted(code, capsys.readouterr().err)
