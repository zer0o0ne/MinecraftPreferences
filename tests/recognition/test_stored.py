"""`read_thresholds` and `complete_run` of src/mcpref/recognition/stored.py against
specs/gui-masks.md (Interface, revisions of 2026-10-20; Produces: thresholds.json, run.json):
per variant and threshold choice (tau, theta, mu), theta and mu as the Fractions of their
strings; the parsed run.json, or RuntimeError naming the path as written when it does not
hold "complete": true."""

import json
from fractions import Fraction

import pytest

from mcpref.recognition.stored import complete_run, read_thresholds

WRITTEN = "data/gui_masks_as_written"


def test_read_thresholds(tmp_path):
    doc = {
        "a": {"agreement": {"tau": 12, "theta": "0.9", "mu": "0"},
              "macro_recall": {"tau": 4, "theta": "0.75", "mu": "0.10"}},
        "d": {"agreement": {"tau": 0, "theta": "0.5", "mu": "0.25"},
              "macro_recall": {"tau": 40, "theta": "1", "mu": "0.333"}},
    }  # fmt: skip
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    got = read_thresholds(path)
    assert got == {
        "a": {"agreement": (12, Fraction(9, 10), Fraction(0)),
              "macro_recall": (4, Fraction(3, 4), Fraction(1, 10))},
        "d": {"agreement": (0, Fraction(1, 2), Fraction(1, 4)),
              "macro_recall": (40, Fraction(1), Fraction(333, 1000))},
    }  # fmt: skip
    for variant in got.values():
        for tau, theta, mu in variant.values():
            assert type(tau) is int
            assert type(theta) is Fraction and type(mu) is Fraction


def test_complete_run_returns_parsed_run_json(tmp_path):
    """complete_run: the parsed {masks_dir}/run.json when it holds "complete": true."""
    doc = {
        "config": {"out_dir": "data/gui_masks", "tau_grid": [0, 4]},
        "sha256": {"config": "0a" * 32, "screens": "1b" * 32},
        "complete": True,
    }
    (tmp_path / "run.json").write_text(json.dumps(doc, indent=2) + "\n")
    assert complete_run(tmp_path, WRITTEN) == doc


@pytest.mark.parametrize(
    "run", [None, {"sha256": {"screens": "1b" * 32}}, {"complete": False}]
)
def test_complete_run_rejects(tmp_path, run):
    """complete_run: run.json absent, as written at the start of a gui-masks run (no
    `complete`), or `"complete": false` -> RuntimeError(f"{written} is not complete")."""
    if run is not None:
        (tmp_path / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    with pytest.raises(RuntimeError) as error:
        complete_run(tmp_path, WRITTEN)
    assert str(error.value) == f"{WRITTEN} is not complete"
