"""(1) and (2) of specs/vpt-gui-frames.md: eligible frames, version and options checks,
episode reasons in order (through `plan` with a fake `fetch`)."""

import pytest
from vpt_checks import plan_config, read_shards
from vpt_world import (
    BAD_JSON,
    BAD_JSONL,
    DIFFERENT,
    OPTIONS,
    REQUIRED,
    Ep,
    joined,
    lines,
    make_world,
    options,
)

from mcpref.vpt.gui_frames import (
    eligible_frames,
    options_ok,
    plan,
    version_ok,
)


@pytest.mark.parametrize(
    "gui, expected",
    [
        ("#", []),  # i = 0 is never eligible
        ("##", [1]),
        (".#", []),  # a container appears one tick late
        ("#.", []),  # the closing tick shows no GUI
        (".##", [2]),
        ("..####.", [3, 4, 5]),
        ("#.#.#.#", []),  # open for single ticks only
        ("i#.#i", [1, 4]),  # inventory then container, container then inventory
        ("ii..##.#", [1, 5]),
        ("##.##.###", [1, 4, 7, 8]),
    ],
)
def test_eligible_frames_eq1(gui, expected):
    """F_e = {i : 1 <= i <= n-1, isGuiOpen in lines i-1 and i}, ascending."""
    assert eligible_frames(lines(gui)) == expected


def test_eligible_frames_ignores_is_gui_inventory():
    """isGuiInventory is an independent check, not part of (1)."""
    flipped = [
        line | {"isGuiInventory": not line["isGuiInventory"]} for line in lines("#.##")
    ]
    assert eligible_frames(flipped) == [3]


@pytest.mark.parametrize(
    "version, min_version, ok",
    [
        ("6.12", "6.12", True),
        ("6.13", "6.12", True),
        ("6.11", "6.12", False),
        ("6.9", "6.12", False),  # (6, 9) < (6, 12); as strings "6.9" > "6.12"
        ("6.10", "6.12", False),
        ("7.0", "6.12", True),
        ("10.0", "6.12", True),  # (10, 0) > (6, 12); as strings "10.0" < "6.12"
        ("9.1", "10.0", False),
        ("10.0", "9.12", True),
        ("6.12", "6.9", True),
    ],
)
def test_version_ok_compares_integer_tuples(version, min_version, ok):
    assert version_ok(version, min_version) is ok


def test_options_ok_accepts_required_values_and_ignores_other_keys():
    assert options_ok(OPTIONS, REQUIRED) is True


@pytest.mark.parametrize("key", sorted(REQUIRED))
def test_options_ok_rejects_any_differing_key(key):
    assert options_ok(OPTIONS | {key: DIFFERENT[key]}, REQUIRED) is False


REASONS = [  # (Ep fields, reason of (2))
    ({"relpath": "data/6.11/x.mp4"}, "version"),
    ({"relpath": "data/6.9/x.mp4", "options": None, "jsonl": None}, "version"),
    ({"options": None}, "missing"),
    ({"jsonl": None}, "missing"),
    ({"options": None, "jsonl": b""}, "missing"),
    ({"options": b"", "jsonl": None}, "missing"),
    ({"options": BAD_JSON, "jsonl": None}, "missing"),
    ({"options": b""}, "empty"),
    ({"jsonl": b""}, "empty"),
    ({"options": b"", "jsonl": BAD_JSONL}, "empty"),
    ({"options": BAD_JSON, "jsonl": b""}, "empty"),
    ({"options": BAD_JSON}, "malformed"),
    ({"jsonl": BAD_JSONL}, "malformed"),
    ({"options": options(lang="en_gb"), "jsonl": BAD_JSONL}, "malformed"),
    ({"options": options(guiScale=3), "gui": "#.#.#."}, "options"),
    *[({"options": options(**{k: v})}, "options") for k, v in DIFFERENT.items()],
    ({"gui": "#.#.#."}, "no_gui"),
    ({"gui": "#"}, "no_gui"),
    ({"gui": ".#.i.#"}, "no_gui"),
    ({"gui": "##"}, None),
    ({"relpath": "data/6.12/x.mp4", "gui": ".##"}, None),
    ({"relpath": "data/10.0/x.mp4", "gui": "i.##"}, None),
]


def records(world, directory) -> list[dict]:
    """Episode records of the plan shards `plan` writes to `directory`."""
    plan(plan_config(world), world.fetch, directory)
    return joined(read_shards(directory), "episodes")


@pytest.mark.parametrize("fields, reason", REASONS)
def test_reason_is_first_failing_check_eq2(tmp_path, fields, reason):
    """Its record: options parsed for `options`, `no_gui` and valid episodes, else null."""
    e = Ep(**({"relpath": "data/7.1/x.mp4"} | fields), reason=reason)
    world = make_world({"8xx": [e]})
    assert records(world, tmp_path / "plan") == [e.record("8xx")]
    if reason == "version":  # nothing downloaded but the indices
        assert set(world.fetched) == set(world.index_urls.values())


@pytest.mark.parametrize(
    "fields",
    [{"options": None}, {"options": b""}, {"options": BAD_JSON},
     {"options": options(lang="en_gb"), "jsonl": BAD_JSONL},
     {"relpath": "data/6.11/x.mp4", "options": options(lang="en_gb")}],
    ids=["missing", "empty", "malformed", "malformed jsonl", "version"],
)  # fmt: skip
def test_record_options_null_for_other_reasons(tmp_path, fields):
    e = Ep(**({"relpath": "data/7.1/x.mp4"} | fields))
    world = make_world({"7xx": [e]})
    assert records(world, tmp_path / "plan")[0]["options"] is None
