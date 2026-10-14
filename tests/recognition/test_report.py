"""`write_report` of src/mcpref/recognition/report.py against specs/gui-masks.md (Interface):
report.json with fractions as JSON floats, report.md headed by the title."""

import json
from fractions import Fraction as Fr

from mcpref.recognition.report import write_report

REPORT = {  # the shapes of the gui-masks and gui-stack reports: fractions, ints, nulls, matrices
    "split": {
        "train": {"episodes": 3, "frames": 7},
        "folds": [{"episodes": 1, "frames": 2}],
    },
    "qualified": ["alpha", "beta"],
    "test": {
        "agreement": Fr(1, 3),
        "macro_recall": None,
        "per_label": {"alpha": {"n_true": 2, "precision": Fr(1, 2), "errors": ["f1"]}},
        "confusion": {"alpha": {"alpha": 1, "other": 1}},
    },
    "agreement_matrices": {
        "agreement": {
            "alpha": {
                "n": 3,
                "labellers": ["VLM", "a"],
                "matrix": [[1, Fr(2, 3)], [Fr(2, 3), 1]],
            }
        }
    },
}
AS_JSON = {  # REPORT with every fraction a float
    **REPORT,
    "test": {
        **REPORT["test"],
        "agreement": 1 / 3,
        "per_label": {"alpha": {"n_true": 2, "precision": 0.5, "errors": ["f1"]}},
    },
    "agreement_matrices": {
        "agreement": {
            "alpha": {
                "n": 3,
                "labellers": ["VLM", "a"],
                "matrix": [[1, 2 / 3], [2 / 3, 1]],
            }
        }
    },
}


def test_report_json_fractions_as_floats(tmp_path):
    """report.json: the report itself, nesting kept, fractions as JSON floats, None as null."""
    write_report(tmp_path, REPORT, "Some title")
    text = (tmp_path / "report.json").read_text()
    assert json.loads(text) == AS_JSON
    assert json.loads(text)["test"]["agreement"] == float(Fr(1, 3))


def test_report_md_headed_by_title(tmp_path):
    """report.md: first line `# <title>`, then the report as tables."""
    write_report(tmp_path, REPORT, "GUI masks report")
    lines = (tmp_path / "report.md").read_text().splitlines()
    assert lines[0] == "# GUI masks report"
    assert any("|" in line for line in lines[1:])
