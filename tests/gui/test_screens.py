"""src/mcpref/gui/screens.py against specs/gui-screens.md (Interface)."""

import dataclasses
from collections import Counter
from functools import reduce

import pytest

from mcpref.gui.screens import (
    Blit,
    Element,
    Label,
    Member,
    Rect,
    ScreenClass,
    element_at,
    load_screens,
)

BLIT = ("texture", "u", "v", "w", "h", "x", "y")
LEVELS = {
    "top": (),
    "class": ("classes", "alpha"),
    "member": ("classes", "alpha", "members", "first"),
    "blit": ("classes", "alpha", "members", "first", "blits", 0),
    "dynamic": ("classes", "alpha", "members", "first", "dynamic", 0),
    "label": ("classes", "alpha", "members", "first", "labels", 0),
    "element": ("classes", "alpha", "elements", 0),
}
REQUIRED = [
    ("top", "classes"),
    *[("class", k) for k in ("members", "description", "window", "elements")],
    *[("member", k) for k in ("blits", "dynamic", "labels")],
    *[("blit", k) for k in BLIT],
    *[("dynamic", k) for k in ("name", "rect")],
    *[("label", k) for k in ("name", "rect", "color")],
    *[("element", k) for k in ("index", "name", "kind", "hit", "click")],
]


def at(data: dict, level: str) -> dict:
    return reduce(lambda d, k: d[k], LEVELS[level], data)


def canonical(cls: ScreenClass) -> dict:
    """All fields; order kept only where the spec fixes it (blits in draw order)."""
    members = {
        k: (m.blits, Counter(m.dynamic), Counter(m.labels))
        for k, m in cls.members.items()
    }
    fields = {f.name: getattr(cls, f.name) for f in dataclasses.fields(cls)}
    return fields | {"members": members, "elements": Counter(cls.elements)}


def by_index(cls: ScreenClass) -> dict:
    return {e.index: e for e in cls.elements}


def test_yaml_forms_map_to_dataclasses(catalogue, write_config):
    alpha = load_screens(write_config(catalogue))["alpha"]
    first = Member(
        (
            Blit("gui/container/inventory.png", 0, 0, 176, 166, 232, 97),
            Blit("gui/widgets.png", 0, 66, 200, 20, 220, 200),
        ),
        (Rect("arrow", 40, 2, 24, 17),),
        (Label(Rect("Title", 8, 6, 31, 8), 0x404040),),
    )
    assert (alpha.name, alpha.description) == ("alpha", "Alpha screen.")
    assert dict(alpha.members) == {
        "first": first,
        "second": Member((), (), (Label(Rect("T", 8, 6, 5, 9), 0xE0E0E0),)),
    }
    assert by_index(alpha)[37] == Element(
        37, "c", "button", Rect("c", 5, 30, 21, 13), (15.5, 36.5)
    )
    assert len(alpha.elements) == 3


def test_unnamed_rectangles_get_spec_names(catalogue, write_config):
    alpha = load_screens(write_config(catalogue))["alpha"]
    assert alpha.window == Rect("window", 0, 0, 176, 166)
    assert alpha.book_panel == Rect("book_panel", 160, 97, 147, 166)
    assert all(e.hit.name == e.name for e in alpha.elements)


def test_absent_book_panel_is_none(catalogue, write_config):
    assert load_screens(write_config(catalogue))["beta"].book_panel is None


def test_sequences_are_tuples_and_clicks_floats(catalogue, write_config):
    alpha = load_screens(write_config(catalogue))["alpha"]
    member = alpha.members["first"]
    assert all(
        type(s) is tuple
        for s in (alpha.elements, member.blits, member.dynamic, member.labels)
    )
    assert all(
        type(e.click) is tuple and all(type(c) is float for c in e.click)
        for e in alpha.elements
    )


def test_dataclasses_are_frozen_with_spec_fields(catalogue, write_config):
    alpha = load_screens(write_config(catalogue))["alpha"]
    member = alpha.members["first"]
    objects = [
        (alpha.window, ("name", "x", "y", "w", "h")),
        (member.blits[0], BLIT),
        (alpha.elements[0], ("index", "name", "kind", "hit", "click")),
        (member.labels[0], ("rect", "color")),
        (member, ("blits", "dynamic", "labels")),
        (alpha, ("name", "description", "window", "book_panel", "members", "elements")),
    ]
    for obj, names in objects:
        assert tuple(f.name for f in dataclasses.fields(obj)) == names
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, names[0], None)


@pytest.mark.parametrize("level", list(LEVELS))
def test_unknown_key_raises(catalogue, write_config, level):
    at(catalogue, level)["bogus"] = 1
    with pytest.raises(ValueError):
        load_screens(write_config(catalogue))


def test_color_on_dynamic_entry_raises(catalogue, write_config):
    """Dynamic entries are {name, rect}: `color` belongs to label entries only."""
    at(catalogue, "dynamic")["color"] = 0x404040
    with pytest.raises(ValueError):
        load_screens(write_config(catalogue))


@pytest.mark.parametrize("level, key", REQUIRED)
def test_missing_required_key_raises(catalogue, write_config, level, key):
    del at(catalogue, level)[key]
    with pytest.raises(ValueError):
        load_screens(write_config(catalogue))


@pytest.mark.parametrize("level", [k for k in LEVELS if k != "top"])
def test_src_note_cond_are_ignored(catalogue, write_config, level):
    plain = {k: canonical(c) for k, c in load_screens(write_config(catalogue)).items()}
    at(catalogue, level).update(src="G:x.java:1", note="a note", cond="hovered=0")
    assert {
        k: canonical(c) for k, c in load_screens(write_config(catalogue)).items()
    } == plain


# Eq. (4): hit [5, 30, 21, 13] has centre (5 + 21/2, 30 + 13/2) = (15.5, 36.5).
@pytest.mark.parametrize(
    "click", [[15.0, 36.5], [15.5, 36.0], [16.0, 37.0], [15.5, 37.5]]
)
def test_click_off_hit_centre_raises(catalogue, write_config, click):
    catalogue["classes"]["alpha"]["elements"][2]["click"] = click
    with pytest.raises(ValueError):
        load_screens(write_config(catalogue))


# alpha: a = [0, 18) x [0, 18), b = [18, 36) x [0, 18), c = [5, 26) x [30, 43); half-open.
@pytest.mark.parametrize(
    "x, y, index",
    [
        (0, 0, 1),
        (17.5, 17.5, 1),
        (17.999, 5, 1),
        (18, 0, 2),
        (35.5, 17.5, 2),
        (36, 5, None),
        (5, 18, None),
        (-0.5, 5, None),
        (5, 30, 37),
        (25.5, 42.5, 37),
        (26, 35, None),
        (10, 43, None),
        (4.5, 35, None),
    ],
)
def test_element_at_half_open_borders(catalogue, write_config, x, y, index):
    alpha = load_screens(write_config(catalogue))["alpha"]
    assert element_at(alpha, x, y) == by_index(alpha).get(index)


# The approved catalogue.


@pytest.fixture(scope="module")
def screens(repo) -> dict:
    return load_screens(repo / "configs/gui-screens.yaml")


def transcribe(name: str, c: dict) -> ScreenClass:
    """Naive reading of the YAML forms of the Interface."""

    def rects(entries: list) -> tuple:
        return tuple(Rect(d["name"], *d["rect"]) for d in entries)

    def labels(entries: list) -> tuple:
        return tuple(Label(Rect(d["name"], *d["rect"]), d["color"]) for d in entries)

    def element(e: dict) -> Element:
        hit = Rect(e["name"], *e["hit"])
        return Element(e["index"], e["name"], e["kind"], hit, tuple(e["click"]))

    members = {
        k: Member(
            tuple(Blit(*(b[f] for f in BLIT)) for b in m["blits"]),
            rects(m["dynamic"]),
            labels(m["labels"]),
        )
        for k, m in c["members"].items()
    }
    panel = Rect("book_panel", *c["book_panel"]) if "book_panel" in c else None
    window = Rect("window", *c["window"])
    elements = tuple(element(e) for e in c["elements"])
    return ScreenClass(name, c["description"], window, panel, members, elements)


def test_loader_reproduces_catalogue(screens, classes):
    expected = {k: canonical(transcribe(k, c)) for k, c in classes.items()}
    assert {k: canonical(c) for k, c in screens.items()} == expected


def inside(hit: list, px: float, py: float) -> bool:
    x, y, w, h = hit
    return x <= px < x + w and y <= py < y + h


def border_points(x: int, y: int, w: int, h: int) -> list:
    """Corners just inside and just outside a hit rectangle, in steps of 0.5 frame px."""
    return [
        (x, y),
        (x - 0.5, y),
        (x, y - 0.5),
        (x + w - 0.5, y + h - 0.5),
        (x + w, y),
        (x, y + h),
    ]


def test_element_at_matches_naive_scan_on_hit_borders(screens, classes):
    for k, c in classes.items():
        for px, py in (p for e in c["elements"] for p in border_points(*e["hit"])):
            hits = [f["index"] for f in c["elements"] if inside(f["hit"], px, py)]
            found = element_at(screens[k], px, py)
            assert ([found.index] if found else []) == hits, (k, px, py)
