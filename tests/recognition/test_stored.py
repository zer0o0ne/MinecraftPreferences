"""`read_thresholds` of src/mcpref/recognition/stored.py against specs/gui-masks.md
(Interface, revisions of 2026-10-20; Produces: thresholds.json): per variant and threshold
choice (tau, theta, mu), theta and mu as the Fractions of their strings."""

import json
from fractions import Fraction

from mcpref.recognition.stored import read_thresholds


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
