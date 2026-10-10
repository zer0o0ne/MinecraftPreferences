"""configs/gui-screens.yaml (the approved catalogue) against specs/gui-screens.md."""

import itertools
from collections import Counter

import pytest

ONE_MEMBER = [
    "inventory",
    "inventory_book",
    "crafting",
    "crafting_book",
    "shulker_box",
    "brewing_stand",
    "enchanting_table",
    "grindstone",
    "stonecutter",
    "loom",
    "cartography_table",
    "merchant",
    "beacon",
    "lectern",
    "death",
    "sleep",
    "pause",
]
TABLE = dict.fromkeys(ONE_MEMBER, 1) | {  # (1): class -> number of members
    "furnace": 3,
    "furnace_book": 3,
    "chest3": 5,
    "chest6": 2,
    "dispenser": 2,
    "hopper": 2,
    "anvil": 2,
    "horse": 7,
    "horse_chest3": 2,
    "horse_chest6": 2,
    "horse_chest9": 2,
    "horse_chest12": 2,
    "horse_chest15": 4,
}
# [G]: screens without the player inventory.
NO_PLAYER = {"lectern", "death", "sleep", "pause"}
# [G] at 640x360, GUI scale 1: guiLeft = (640 - xSize) / 2, guiTop = (360 - ySize) / 2.
HAND = [
    # inventory 176x166 at (232, 97); hotbar slot 0 at (8, 142) -> slot (240, 239).
    ("inventory", 1, "slot", [239, 238, 18, 18], [248, 247]),
    # chest6 176x222 at (232, 69); chest slot 53 (row 5, col 8) at (8 + 144, 18 + 90).
    ("chest6", 90, "slot", [383, 176, 18, 18], [392, 185]),
    # book open: guiLeft = 177 + (640 - 176 - 200) / 2 = 309;
    # recipe button 20x18 at (guiLeft + 104, 360 / 2 - 22).
    ("inventory_book", 47, "button", [413, 158, 20, 18], [423, 167]),
    # beacon 230x219 at (205, 70); tier-2 primary (strength) 22x22
    # at (205 + 76 - 22 / 2, 70 + 22 + 2 * 25).
    ("beacon", 42, "button", [270, 142, 22, 22], [281, 153]),
]


def all_elements(classes: dict) -> list:
    return [e for c in classes.values() for e in c["elements"]]


def all_members(classes: dict, names=None) -> list:
    names = classes if names is None else names
    return [m for k in names for m in classes[k]["members"].values()]


def by_index(c: dict) -> dict:
    return {e["index"]: e for e in c["elements"]}


@pytest.mark.parametrize("name, index, kind, hit, click", HAND)
def test_hand_checked_entries(classes, name, index, kind, hit, click):
    e = by_index(classes[name])[index]
    assert (e["kind"], e["hit"], e["click"]) == (kind, hit, click)


def test_classes_and_member_counts_follow_table_1(classes):
    assert {k: len(c["members"]) for k, c in classes.items()} == TABLE


def test_book_panel_only_on_recipe_book_classes(classes):
    book = {k for k, c in classes.items() if "book_panel" in c}
    assert book == {"inventory_book", "crafting_book", "furnace_book"}


def test_full_screen_classes_carry_labels_and_elements_only(classes):
    members = all_members(classes, ["death", "sleep", "pause"])
    assert all(m["blits"] == [] and m["dynamic"] == [] for m in members)


def test_label_heights_follow_5(classes):
    """(5): 8 without a shadow, 9 with one, 18 only for the death title drawn at 2x."""
    heights = Counter(
        (k, label["rect"][3])
        for k in classes
        for m in all_members(classes, [k])
        for label in m["labels"]
    )
    assert {h for _, h in heights} <= {8, 9, 18}
    assert {k: n for (k, h), n in heights.items() if h == 18} == {"death": 1}


def test_blit_textures_lie_under_gui(classes):
    """main extracts gui/ and font/ascii.png: every blit texture exists after a download."""
    textures = {b["texture"] for m in all_members(classes) for b in m["blits"]}
    assert all(t.startswith("gui/") for t in textures)


def test_slot_hit_is_18_square(classes):
    """(2): a slot at (x_s, y_s) is hit on (x_s - 1, y_s - 1, 18, 18)."""
    slots = [e for e in all_elements(classes) if e["kind"] == "slot"]
    assert all(e["hit"][2:] == [18, 18] for e in slots)


def test_every_click_is_the_hit_centre_inside_it(classes):
    for e in all_elements(classes):
        x, y, w, h = e["hit"]
        assert e["click"] == [x + w / 2, y + h / 2], e
        assert x <= e["click"][0] < x + w and y <= e["click"][1] < y + h, e


def test_no_two_elements_of_a_class_overlap(classes):
    for k, c in classes.items():
        for a, b in itertools.combinations([e["hit"] for e in c["elements"]], 2):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            assert not (
                ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
            ), (k, a, b)


def test_indices_unique_and_consecutive(classes):
    """(3): from 1 on classes showing the player inventory, from 37 on the others."""
    for k, c in classes.items():
        start, n = (37 if k in NO_PLAYER else 1), len(c["elements"])
        assert sorted(e["index"] for e in c["elements"]) == list(
            range(start, start + n)
        ), k


def test_n_max_is_90_on_chest6(classes):
    top = {k: max(by_index(c)) for k, c in classes.items()}
    assert max(top.values()) == top["chest6"] == 90


def player_slot(i: int, x0: int, y0: int) -> list:
    """[G]: hotbar 0-8 in a row at y0; main 9-35 in 3 rows of 9 from 58 px above it."""
    if i <= 9:
        return [x0 + 18 * (i - 1), y0]
    return [x0 + 18 * ((i - 10) % 9), y0 - 58 + 18 * ((i - 10) // 9)]


def test_indices_1_to_36_are_the_player_slots(classes):
    """(3): 1-9 = hotbar slots 0-8, 10-36 = main inventory slots 9-35."""
    for k in classes.keys() - NO_PLAYER:
        el = by_index(classes[k])
        x0, y0 = el[1]["hit"][:2]
        for i in range(1, 37):
            assert el[i]["kind"] == "slot", (k, i)
            assert el[i]["hit"][:2] == player_slot(i, x0, y0), (k, i)


def test_container_slots_precede_buttons(classes):
    """(3): from 37, container slots first, then buttons."""
    for k, c in classes.items():
        kinds = [e["kind"] for i, e in sorted(by_index(c).items()) if i >= 37]
        assert kinds == sorted(kinds, key=lambda kind: kind == "button"), k
