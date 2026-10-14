"""`main` of src/mcpref/stacking/run.py against specs/gui-stack.md (Interface): exit 1 with a
one-line message as the last line of stderr."""

import pytest
from progress_bars import assert_aborted

from mcpref.stacking.run import main

from .stack_world import (
    CONFIG,
    LABELS,
    NAN_WORLD,
    PRED_KEYS,
    Part,
    World,
    preds_for,
    run_main,
)

TEST = Part(["te0"], ["beta"], [True], [[0.5]])
C_ZERO = World([], LABELS, Part(["tr0"], ["alpha"], [True], [[]]),
               Part(["va0"], ["beta"], [False], [[]]), Part(["te0"], ["beta"], [True], [[]]),
               preds_for(TEST, LABELS, 2))  # fmt: skip
S_EMPTY = World(["a/alpha"], LABELS, Part([], [], [], []), Part([], [], [], []), TEST,
                preds_for(TEST, LABELS, 2))  # fmt: skip
READ_KEYS = ["columns", "labels", "train_x", "train_y", "validation_x", "validation_y",
             "test_ids", "test_y", "test_g", "test_x", *PRED_KEYS]  # fmt: skip
CASES = {
    "C = 0": (C_ZERO, CONFIG),
    "S empty": (S_EMPTY, CONFIG),
    "features file missing": (None, CONFIG),
    "config violating a constraint": (NAN_WORLD, {**CONFIG, "hidden": 0}),
}


@pytest.mark.parametrize("world, config", CASES.values(), ids=CASES)
def test_failures(tmp_path, repo, world, config):
    run = run_main(tmp_path, repo, world, config)
    assert_aborted(run.code, run.err)


@pytest.mark.parametrize("key", READ_KEYS)
def test_features_without_a_key(tmp_path, repo, key):
    """Every key this block reads, the twelve test_pred_* included."""
    run = run_main(tmp_path, repo, NAN_WORLD, drop=[key])
    assert_aborted(run.code, run.err)


def test_missing_config(tmp_path, capsys):
    code = main([str(tmp_path / "absent.yaml")])
    assert_aborted(code, capsys.readouterr().err)
