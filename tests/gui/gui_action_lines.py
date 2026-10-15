"""Hand-made `.jsonl` lines (fields of specs/vpt-actions.md [R3]) and a tiny catalogue
(specs/gui-screens.md) for the tests of specs/vpt-gui-actions.md.

A row is one line: `inv(...)` the player inventory, `box(...)` a container, `world(...)`
no GUI; `keys` held (short names), `again` keys auto-repeated (in `newKeys` while held),
`press` the buttons of `newButtons` in order, `hold` buttons held besides, `at` the mouse
position in window px, `ms` the `milli` offset (50 per line by default).
"""

from mcpref.gui.screens import Element, Rect, ScreenClass

KEY = "key.keyboard."
T0 = 1650000000000
MS = 250  # double_click_ms of the tests that do not vary it
GUIS = {"": (False, False), "@": (True, True), "#": (True, False)}


def px(fx: float, fy: float) -> tuple[float, float]:
    """Window px whose frame point (x/2, y/2) is (fx, fy)."""
    return 2.0 * fx, 2.0 * fy


def slot(index: int, x: int, y: int) -> Element:
    hit = Rect(f"s{index}", x, y, 18, 18)
    return Element(index, f"s{index}", "slot", hit, (x + 9.0, y + 9.0))


def button(index: int, x: int, y: int, w: int, h: int) -> Element:
    hit = Rect(f"b{index}", x, y, w, h)
    return Element(index, f"b{index}", "button", hit, (x + w / 2, y + h / 2))


def screen(name: str, *elements: Element) -> ScreenClass:
    window = Rect("window", 90, 40, 200, 120)
    return ScreenClass(name=name, description=f"{name}.", window=window,
                       book_panel=None, members={}, elements=elements)  # fmt: skip


# Frame px: slots 1, 2, 3 at x 100, 118, 136 (y 100), slot 10 below slot 1; `box` lacks
# slot 3 and has container slots 37, 38, 40 (n_max); `lectern`, `death`, `pause` have
# buttons over slot 1.
SCREENS = {
    "inv": screen("inv", slot(1, 100, 100), slot(2, 118, 100), slot(3, 136, 100),
                  slot(10, 100, 118), button(37, 160, 100, 20, 13)),
    "box": screen("box", slot(1, 100, 100), slot(2, 118, 100), slot(10, 100, 118),
                  slot(37, 100, 50), slot(38, 118, 50), slot(40, 136, 50)),
    "lectern": screen("lectern", button(37, 100, 100, 18, 18),
                      button(38, 118, 100, 18, 18)),
    "death": screen("death", button(37, 100, 100, 40, 20)),
    "pause": screen("pause", button(37, 100, 100, 40, 20)),
}  # fmt: skip
S1, S2, S3, S10 = px(109, 109), px(127, 109), px(145, 109), px(109, 127)
B37, C37 = px(170, 106), px(109, 59)  # `inv` button 37, `box` slot 37
OUT = px(20, 20)  # no element of any class


def row(gui: str, keys=(), press=(), hold=(), at=OUT, again=(), ms=None) -> dict:
    return {"gui": gui, "keys": list(keys), "press": list(press), "hold": list(hold),
            "at": at, "again": set(again), "ms": ms}  # fmt: skip


def inv(**kw) -> dict:
    return row("@", **kw)


def box(**kw) -> dict:
    return row("#", **kw)


def world(**kw) -> dict:
    return row("", **kw)


def line(t: int, r: dict, new: list[str]) -> dict:
    is_open, is_inventory = GUIS[r["gui"]]
    (x, y), ms = r["at"], 50 * t if r["ms"] is None else r["ms"]
    held = r["press"] + [b for b in r["hold"] if b not in r["press"]]
    mouse = {"x": x, "y": y, "dx": 0.0, "dy": 0.0, "scaledX": x / 2, "scaledY": y / 2,
             "dwheel": 0.0, "buttons": held, "newButtons": list(r["press"])}  # fmt: skip
    keyboard = {
        "keys": [KEY + k for k in r["keys"]],
        "newKeys": [KEY + k for k in new],
        "chars": "",
    }
    return {
        "mouse": mouse, "keyboard": keyboard, "isGuiOpen": is_open,
        "isGuiInventory": is_inventory, "hotbar": 0, "yaw": 0.0, "pitch": 0.0,
        "xpos": 0.5, "ypos": 64.0, "zpos": 0.5, "tick": t, "milli": T0 + ms,
        "inventory": [], "serverTick": t, "serverTickDurationMs": 50.0, "stats": {},
    }  # fmt: skip


def episode(*rows: dict) -> list[dict]:
    """Lines of the rows; `newKeys` = keys not held on the line before, plus `again`."""
    lines, before = [], set()
    for t, r in enumerate(rows):
        new = [k for k in r["keys"] if k not in before or k in r["again"]]
        lines.append(line(t, r, new))
        before = set(r["keys"])
    return lines
