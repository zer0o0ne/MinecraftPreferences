"""`check_fingerprints` (4) of src/mcpref/runs/files.py against specs/runs.md: the order
given, `sha256` before the top level, the KeyErrors, Python != on the parsed values, no file."""

import contextlib
import json
import os
import re

import pytest

from mcpref.runs.files import check_fingerprints

OUT = "./data/out_as_written/"  # out_dir as written in a config: kept verbatim
RUN = {
    "config": {"seed": 1},
    "sha256": {"config": "c1", "screens": "s-sha"},
    "screens": "s-top",
    "frames": {"ok": 2, "failed": 1},
    "labels": "l1",
}
SAME = {"config": "c1", "screens": "s-sha", "frames": {"failed": 1, "ok": 2}}
SAME["labels"] = "l1"


def check(tmp_path, fingerprints: dict, run=RUN) -> None:
    """check_fingerprints on {tmp_path}/run.json holding `run`."""
    (tmp_path / "run.json").write_text(json.dumps(run))
    check_fingerprints(tmp_path / "run.json", fingerprints, OUT)


def changed(name: str):
    """RuntimeError of (4) naming `name`, out_dir as given."""
    message = f"{name} changed since the run started; delete {OUT}"
    return pytest.raises(RuntimeError, match=f"^{re.escape(message)}$")


def test_no_run_file_checks_nothing(tmp_path):
    """(4): no file -> nothing, whatever the fingerprints; nothing written."""
    assert (
        check_fingerprints(tmp_path / "absent/run.json", {"config": "zz"}, OUT) is None
    )
    assert os.listdir(tmp_path) == []


def test_equal_fingerprints_pass(tmp_path):
    """(4): sha256 for config and screens, the top level for frames and labels."""
    check(tmp_path, SAME)
    assert json.loads((tmp_path / "run.json").read_text()) == RUN


ORDERS = [["config", "frames", "labels"], ["labels", "config", "frames"]]
ORDERS += [["frames", "labels", "config"]]


@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_first_difference_in_the_order_given(tmp_path, order):
    """(4): every fingerprint differs; the first in the given order is named."""
    with changed(order[0]):
        check(tmp_path, {name: "other" for name in order})


def test_equal_fingerprints_before_the_first_difference(tmp_path):
    with changed("labels"):
        check(tmp_path, {"config": "c1", "frames": SAME["frames"], "labels": "l2"})


@pytest.mark.parametrize("value, fails", [("s-sha", False), ("s-top", True)])
def test_sha256_before_the_top_level(tmp_path, value, fails):
    """(4): a name in sha256 and at the top level is compared with the one in sha256."""
    with changed("screens") if fails else contextlib.nullcontext():
        check(tmp_path, {"screens": value})


@pytest.mark.parametrize("fingerprints", [{"labels": "l1"}, {}])
def test_key_error_without_sha256(tmp_path, fingerprints):
    """(4): an existing file holds a sha256 object, else KeyError (also when every name is
    at the top level and equal, and for no name)."""
    with pytest.raises(KeyError):
        check(tmp_path, fingerprints, {k: v for k, v in RUN.items() if k != "sha256"})


ORDERED = {  # fingerprints in order -> error; a name in neither raises once reached
    "neither": ({"config": "c1", "episodes": 3}, KeyError),
    "neither, then a difference": ({"episodes": 3, "labels": "l2"}, KeyError),
    "a difference, then neither": ({"labels": "l2", "episodes": 3}, RuntimeError),
}


@pytest.mark.parametrize("case", ORDERED)
def test_name_in_neither(tmp_path, case):
    fingerprints, error = ORDERED[case]
    with pytest.raises(error):
        check(tmp_path, fingerprints)


NE = {  # (stored, fingerprint, differs) under Python != on the parsed values
    "dict order": ({"ok": 2, "failed": 1}, {"failed": 1, "ok": 2}, False),
    "int and float": (3, 3.0, False),
    "true and 1": (True, 1, False),
    "str and int": ("3", 3, True),
    "list and tuple": ([1, 2], (1, 2), True),
    "nested value": ({"ok": 2}, {"ok": 3}, True),
}


@pytest.mark.parametrize("case", NE)
def test_compared_with_ne_on_parsed_values(tmp_path, case):
    stored, fingerprint, differs = NE[case]
    with changed("frames") if differs else contextlib.nullcontext():
        check(tmp_path, {"frames": fingerprint}, {**RUN, "frames": stored})
