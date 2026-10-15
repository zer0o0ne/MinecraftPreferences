"""Naive transcription of (1)-(11) of specs/vpt-actions.md, the oracle of the random checks."""

import math

KEY, CUSTOM = "key.keyboard.", "minecraft.custom:minecraft."
NAMES = {(1, 0): "forward", (-1, 0): "back", (0, -1): "left", (0, 1): "right",
         (1, -1): "forward_left", (1, 1): "forward_right", (-1, -1): "back_left",
         (-1, 1): "back_right"}  # fmt: skip
DAMAGE = {CUSTOM + "damage_dealt" + s for s in ("", "_absorbed", "_resisted")}
RIDING = {
    CUSTOM + x + "_one_cm" for x in ("boat", "horse", "pig", "minecart", "strider")
}
C3 = {
    KEY + c for c in ["a", "b", "c", "d", "f", "g", "h", "i", "n", "p", "q", "t", "f4"]
}


def world(ln) -> bool:
    return not ln["isGuiOpen"]


def has(ln, name: str, field: str = "keys") -> bool:
    return KEY + name in ln["keyboard"][field]


def move(ln):
    """(1), move: None or the name of (f, r, j)."""
    f, r, j = has(ln, "w") - has(ln, "s"), has(ln, "d") - has(ln, "a"), has(ln, "space")
    if not world(ln) or (f, r, j) == (0, 0, 0):
        return None
    if (f, r) == (0, 0):
        return "jump"
    return ("jump_" if j else "") + NAMES[f, r]


def grew(lines, k: int, pick) -> bool:
    """(2) and (11): some stat x with pick(x) grows from line k to line k + 1."""
    a, b = lines[k]["stats"], lines[k + 1]["stats"]
    return any(b.get(x, 0) > a.get(x, 0) for x in set(a) | set(b) if pick(x))


def run_end(v: list, k: int, w: list | None = None) -> int:
    """Last line of the run of value v[k] from line k; of world lines only when w is given."""
    q = k
    while q + 1 < len(v) and v[q + 1] == v[k] and (w is None or w[q + 1]):
        q += 1
    return q


def fill_releases(v: list, w: list, r_min: int) -> None:
    """(3a), in place."""
    k = 0
    while k < len(v):
        if not (w[k] and v[k] is None):
            k += 1
            continue
        q = run_end(v, k, w)
        inside = 0 < k and q + 1 < len(v) and w[k - 1] and w[q + 1]
        if (
            q - k + 1 < r_min
            and inside
            and v[k - 1] is not None
            and v[k - 1] == v[q + 1]
        ):
            v[k : q + 1] = [v[k - 1]] * (q - k + 1)
        k = q + 1


def merge_short(v: list, w: list, r_min: int) -> None:
    """(3b), in place, in line order."""
    k = 0
    while k < len(v):
        if v[k] is None:
            k += 1
            continue
        q = run_end(v, k)
        before = k > 0 and w[k - 1] and v[k - 1] is not None
        after = q + 1 < len(v) and w[q + 1] and v[q + 1] is not None
        if q - k + 1 < r_min and not v[k].startswith("jump") and before and after:
            v[k : q + 1] = [v[k - 1]] * (q - k + 1)
        k = q + 1


def pieces(p: int, q: int, effect: list, r_min: int) -> list:
    """(4): the pieces of run [p, q]."""
    ks = [k for k in range(p, q) if effect[k]]
    while ks and ks[0] - p + 1 < r_min:
        ks.pop(0)
    while ks and q - ks[-1] < r_min:
        ks.pop()
    return list(zip([p] + [k + 1 for k in ks], ks + [q]))


def channels(lines, r_min: int) -> list:
    """(1)-(3): (values, effects) of move, sneak, attack, use."""
    n, w = len(lines), [world(ln) for ln in lines]
    v = [move(ln) for ln in lines]
    fill_releases(v, w, r_min)
    merge_short(v, w, r_min)

    def grows(pick) -> list:
        return [k < n - 1 and grew(lines, k, pick) for k in range(n)]

    def held(name: str, test) -> list:
        return [name if w[k] and test(ln) else None for k, ln in enumerate(lines)]

    jumps = grows(lambda x: x == CUSTOM + "jump")
    jump = [
        e and has(ln, "space") for e, ln in zip(jumps, lines)
    ]  # j_k of (1), unsmoothed
    mine = grows(lambda x: x.startswith("minecraft.mine_block:") or x in DAMAGE)
    use = grows(lambda x: x.startswith("minecraft.use_item:"))
    return [(v, jump), (held("sneak", lambda ln: has(ln, "left.shift")), [False] * n),
            (held("attack", lambda ln: 0 in ln["mouse"]["buttons"]), mine),
            (held("use", lambda ln: 1 in ln["mouse"]["buttons"]), use)]  # fmt: skip


def segments(lines, r_min: int) -> list:
    """(1)-(6) in the order of (8): (action, first, last)."""
    out = []
    for c, (v, effect) in enumerate(channels(lines, r_min)):
        k = 0
        while k < len(v):
            q = run_end(v, k)
            if v[k] is not None:
                out += [(p2, c, v[k], q2) for p2, q2 in pieces(k, q, effect, r_min)]
            k = q + 1
    for k in range(len(lines) - 1):
        d = (lines[k + 1]["hotbar"] - lines[k]["hotbar"]) % 9
        if world(lines[k]) and d:
            out += (
                [(k, 4, "hotbar_right", k)] * d
                if d <= 4
                else [(k, 4, "hotbar_left", k)] * (9 - d)
            )
    out += [(k, 5, "open_inventory", k) for k, ln in enumerate(lines)
            if world(ln) and has(ln, "e", "newKeys")]  # fmt: skip
    return [(a, p, q) for p, _, a, q in sorted(out, key=lambda s: s[:2])]


def wrap(x: float) -> float:
    return x - 360 * math.ceil((x - 180) / 360)


def interval(segs, lines, a: int, b: int, cut, rho: float) -> list:
    """(7), (8)."""
    lines_in = set(range(a, b)) - set(cut)
    acts = [s[0] for s in segs if any(s[1] <= k <= s[2] for k in lines_in)]
    dpsi = sum(wrap(lines[k + 1]["yaw"] - lines[k]["yaw"]) for k in range(a, b))
    dth = lines[b]["pitch"] - lines[a]["pitch"]
    acts += ["turn_right" if dpsi > 0 else "turn_left"] * math.floor(
        abs(dpsi) / rho + 0.5
    )
    return acts + ["turn_down" if dth > 0 else "turn_up"] * math.floor(
        abs(dth) / rho + 0.5
    )


def modes(lines, start) -> list:
    """(9): Omega_0..Omega_{T-1}."""
    n = len(lines)
    toggles = {"O": [0] * (n + 2), "H": [0] * (n + 2), "V": [0] * (n + 2)}
    passes = [world(ln) or ln["isGuiInventory"] for ln in lines]
    eta = False
    for k, ln in enumerate(lines):
        if world(ln) and has(ln, "f3") and C3 & set(ln["keyboard"]["newKeys"]):
            eta = True
        if k >= 1 and has(lines[k - 1], "f3") and not has(ln, "f3") and passes[k - 1]:
            toggles["O"][k + 1] += not eta
            eta = False
        if world(ln) and has(ln, "f1", "newKeys"):
            toggles["H"][k + 2] += 1
        if passes[k] and has(ln, "f5", "newKeys"):
            toggles["V"][k + 1] += 1
    (o, h, v), out = start, []
    for t in range(n):
        o, h = o ^ (toggles["O"][t] % 2 == 1), h ^ (toggles["H"][t] % 2 == 1)
        v = (v + toggles["V"][t]) % 3
        out.append(o or h or v != 0)
    return out


def foreign(lines) -> list:
    """(10)."""
    return [world(ln) and not has(ln, "f3") and (has(ln, "q", "newKeys") or has(ln, "f", "newKeys"))
            for ln in lines]  # fmt: skip


def riding(lines) -> list:
    """(11)."""
    return [
        k < len(lines) - 1 and grew(lines, k, RIDING.__contains__)
        for k in range(len(lines))
    ]
