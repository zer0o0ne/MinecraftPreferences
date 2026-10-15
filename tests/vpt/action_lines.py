"""Hand-made `.jsonl` lines with the fields of [R3] for the tests of specs/vpt-actions.md.

A token describes one line: `[gui:]item+item+...` with gui `#` (a chest: no events
passed) or `@` (the player inventory), items short key names (`w`, `space`, `f3`, ...) or
mouse buttons `M0`, `M1`, `M2`; `-` holds nothing; `#` and `@` alone are empty GUI lines.
A key is in `newKeys` on the line it appears; a trailing `!` adds it again (auto-repeat).
"""

KEY = "key.keyboard."
CUSTOM = "minecraft.custom:minecraft."
GUIS = {"#": (True, False), "@": (True, True)}


def line(held, new, buttons, gui="", k=0, hotbar=0, yaw=0.0, pitch=0.0) -> dict:
    """One recorder line; `held`, `new` short key names, `buttons` mouse button numbers."""
    is_open, is_inventory = GUIS.get(gui, (False, False))
    mouse = {"x": 640.0, "y": 360.0, "dx": 0.0, "dy": 0.0, "scaledX": 320.0,
             "scaledY": 180.0, "dwheel": 0.0, "buttons": list(buttons), "newButtons": []}  # fmt: skip
    keyboard = {
        "keys": [KEY + n for n in held],
        "newKeys": [KEY + n for n in new],
        "chars": "",
    }
    return {
        "mouse": mouse, "keyboard": keyboard, "isGuiOpen": is_open,
        "isGuiInventory": is_inventory, "hotbar": hotbar, "yaw": float(yaw),
        "pitch": float(pitch), "xpos": 0.5, "ypos": 64.0, "zpos": 0.5, "tick": k,
        "milli": 1650000000000 + 50 * k, "inventory": [], "serverTick": k,
        "serverTickDurationMs": 50.0, "stats": {CUSTOM + "play_time": 100 + k},
    }  # fmt: skip


def parse(token: str) -> tuple[str, list[str], set[str], list[int]]:
    """gui, held keys, keys pressed again, buttons of one token."""
    if token in GUIS:
        return token, [], set(), []
    gui, _, items = token.rpartition(":")
    names = [] if items == "-" else items.split("+")
    keys = [n.rstrip("!") for n in names if not n.startswith("M")]
    again = {n.rstrip("!") for n in names if n.endswith("!")}
    return gui, keys, again, [int(n[1:]) for n in names if n.startswith("M")]


def episode(tokens: str, hotbar=None, yaw=None, pitch=None) -> list[dict]:
    """Lines of space-separated tokens; per-line hotbar, yaw, pitch lists (0 by default)."""
    parsed = [parse(t) for t in tokens.split()]
    zeros = [0] * len(parsed)
    hotbar, yaw, pitch = hotbar or zeros, yaw or zeros, pitch or zeros
    lines, before = [], set()
    for k, (gui, keys, again, buttons) in enumerate(parsed):
        new = [n for n in keys if n not in before or n in again]
        lines.append(line(keys, new, buttons, gui, k, hotbar[k], yaw[k], pitch[k]))
        before = set(keys)
    return lines


def bump(lines: list[dict], stat: str, *after: int) -> list[dict]:
    """Makes `stat` grow by 1 between line k and k + 1 for every k in `after`; absent while 0."""
    for i, ln in enumerate(lines):
        count = sum(k < i for k in after)
        if count:
            ln["stats"][stat] = ln["stats"].get(stat, 0) + count
    return lines


def repeat(token: str, n: int) -> str:
    return " ".join([token] * n)
