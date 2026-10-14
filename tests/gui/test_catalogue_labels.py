"""Label entries of configs/gui-screens.yaml (the approved catalogue) against
specs/gui-screens.md (5) and the Interface's data change of 2026-10-14."""

import re
from collections import Counter

import pytest

# Interface, hand-checked label colours: (class, label, colour).
LABEL_COLOURS = [
    ("inventory", "Crafting", 0x404040),
    ("beacon", "Primary Power", 0xE0E0E0),
    ("sleep", "Leave Bed", 0xFFFFFF),
]


def labels(classes: dict, names=None) -> list:
    names = classes if names is None else names
    return [
        x for k in names for m in classes[k]["members"].values() for x in m["labels"]
    ]


@pytest.mark.parametrize("name, text, colour", LABEL_COLOURS)
def test_hand_checked_label_colours(classes, name, text, colour):
    found = [x["color"] for x in labels(classes, [name]) if x["name"] == text]
    assert found == [colour]


def test_label_colours_follow_data_change(classes):
    """98 x 0x404040, 12 x 0xFFFFFF and the death title 0xFFFFFF, 2 x 0xE0E0E0; 113 labels."""
    colours = Counter(x["color"] for x in labels(classes))
    assert colours == {0x404040: 98, 0xFFFFFF: 13, 0xE0E0E0: 2}


def test_colours_are_yaml_hex_literals(repo, classes):
    """Interface: `color` is 0xRRGGBB written as a YAML hex literal, on every label."""
    text = (repo / "configs/gui-screens.yaml").read_text()
    values = re.findall(r"^[ -]*color:[ ]*(.*?)[ ]*$", text, flags=re.MULTILINE)
    assert all(re.fullmatch(r"0x[0-9A-Fa-f]{6}", v) for v in values), set(values)
    assert len(values) == len(labels(classes))


def test_label_notes_carry_no_colour(classes):
    """Data change: the colour is dropped from the notes."""
    notes = [x["note"] for x in labels(classes) if "note" in x]
    assert not [n for n in notes if "colour" in n or "0x" in n]


def test_label_heights_follow_5(classes):
    """(5): 8 without a shadow, 9 with one, 18 only for the death title drawn at 2x."""
    heights = Counter((k, x["rect"][3]) for k in classes for x in labels(classes, [k]))
    assert {h for _, h in heights} <= {8, 9, 18}
    assert {k: n for (k, h), n in heights.items() if h == 18} == {"death": 1}


def test_merchant_title_is_dynamic(classes):
    """(5): the title varies with profession and level. Widest of the 66 en_us titles:
    "Leatherworker - Journeyman", advances 6 per glyph but t 4, k 5, space 4:
    75 + 14 + 60 = 149; x = (640 - 276) / 2 + 49 + 276 / 2 - 149 // 2 = 295,
    y = (360 - 166) / 2 + 6 = 103; height 8 without a shadow."""
    merchant = classes["merchant"]["members"]["merchant"]
    assert [295, 103, 149, 8] in [d["rect"] for d in merchant["dynamic"]]
    assert sorted(x["name"] for x in merchant["labels"]) == ["Inventory", "Trades"]


def test_death_button_texts_are_not_labels(classes):
    """(5): Respawn and Title Screen lie inside their hit rectangles; the title is white."""
    found = [(x["name"], x["color"]) for x in labels(classes, ["death"])]
    assert found == [("You Died!", 0xFFFFFF)]
