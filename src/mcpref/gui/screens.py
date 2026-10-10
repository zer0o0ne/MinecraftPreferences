"""Interface of specs/gui-screens.md: the screen catalogue of configs/gui-screens.yaml."""

import dataclasses
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

IGNORED = {"src", "note", "cond"}


@dataclass(frozen=True)
class Rect:
    """Notation: rectangle (x, y, w, h) = [x, x+w) x [y, y+h) in frame px."""

    name: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Blit:
    """Notation: texture path, u, v, w, h, destination (x, y)."""

    texture: str
    u: int
    v: int
    w: int
    h: int
    x: int
    y: int


@dataclass(frozen=True)
class Element:
    """(2)-(4): canonical index, hit rectangle, click point."""

    index: int
    name: str
    kind: Literal["slot", "button"]
    hit: Rect
    click: tuple[float, float]


@dataclass(frozen=True)
class Member:
    """(5): drawing data of one member."""

    blits: tuple[Blit, ...]
    dynamic: tuple[Rect, ...]
    labels: tuple[Rect, ...]


@dataclass(frozen=True)
class ScreenClass:
    """(1): a class, its members and elements."""

    name: str
    description: str
    window: Rect
    book_panel: Rect | None
    members: Mapping[str, Member]
    elements: tuple[Element, ...]


def keys(cls: type) -> set[str]:
    """Interface Data: the keys of an entry are the field names of its dataclass."""
    return {f.name for f in dataclasses.fields(cls)}


def fields(
    entry: dict, required: AbstractSet[str], optional: AbstractSet[str] = frozenset()
) -> dict:
    """Interface Data: `entry` without src/note/cond; unknown or missing keys are errors."""
    unknown = set(entry) - required - optional - IGNORED
    missing = required - set(entry)
    if unknown or missing:
        found = f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        raise ValueError(f"{found} among {sorted(entry)}")
    return {k: v for k, v in entry.items() if k in required | optional}


def read_catalogue(path: Path) -> dict:
    """Interface Data: the YAML file with top-level keys `classes`, `textures`."""
    return fields(yaml.safe_load(path.read_text()), {"classes", "textures"})


def named(entry: dict) -> Rect:
    """Interface Data: a dynamic or label entry `{name, rect}`."""
    e = fields(entry, {"name", "rect"})
    return Rect(e["name"], *e["rect"])


def member(entry: dict) -> Member:
    """(5): blits, dynamic and label rectangles of one member."""
    e = fields(entry, keys(Member))
    return Member(
        blits=tuple(Blit(**fields(b, keys(Blit))) for b in e["blits"]),
        dynamic=tuple(named(r) for r in e["dynamic"]),
        labels=tuple(named(r) for r in e["labels"]),
    )


def click_point(hit: Rect) -> tuple[float, float]:
    """(4): centre of the hit rectangle."""
    return (hit.x + hit.w / 2, hit.y + hit.h / 2)


def element(entry: dict) -> Element:
    """(2)-(4): one element; its `click` must equal (4) for its hit rectangle."""
    e = fields(entry, keys(Element))
    hit = Rect(e["name"], *e["hit"])
    click = click_point(hit)
    if tuple(e["click"]) != click:
        raise ValueError(f"click {e['click']} of {e['name']} is not (4): {click}")
    return Element(e["index"], e["name"], e["kind"], hit, click)


def screen_class(name: str, entry: dict) -> ScreenClass:
    """(1)-(5): one class; `book_panel` only on recipe-book classes."""
    e = fields(entry, keys(ScreenClass) - {"name", "book_panel"}, {"book_panel"})
    book_panel = e.get("book_panel")
    return ScreenClass(
        name=name,
        description=e["description"],
        window=Rect("window", *e["window"]),
        book_panel=None if book_panel is None else Rect("book_panel", *book_panel),
        members={m: member(v) for m, v in e["members"].items()},
        elements=tuple(element(x) for x in e["elements"]),
    )


def load_screens(path: Path) -> dict[str, ScreenClass]:
    """Interface: the classes of the catalogue, in file order."""
    classes = read_catalogue(path)["classes"]
    return {name: screen_class(name, entry) for name, entry in classes.items()}


def contains(r: Rect, x: float, y: float) -> bool:
    """Notation: (x, y) in [r.x, r.x + r.w) x [r.y, r.y + r.h)."""
    return r.x <= x < r.x + r.w and r.y <= y < r.y + r.h


def element_at(cls: ScreenClass, x: float, y: float) -> Element | None:
    """Interface: the element whose hit rectangle contains the frame point (x, y)."""
    return next((e for e in cls.elements if contains(e.hit, x, y)), None)
