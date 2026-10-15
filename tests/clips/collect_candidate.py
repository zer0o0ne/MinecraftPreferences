"""Expected work (2)-(5) of a candidate of specs/perception-data.md on the episodes of
collect_world.py: its classifier calls, the clips of specs/clips.md composed of the
transcriptions of collect_oracle.py, and its record fields and clip files."""

import itertools
import json
import math

import numpy as np

from . import collect_oracle as o
from .collect_world import Ep, lines


def interval(w: o.Work, ev: dict, a: int, b: int) -> tuple[list, bool]:
    """clips (1): actions of S touching I = a..b-1 minus C, then the pitch turns."""
    span = set(range(a, b)) - ev["cut"]
    acts = [s[0] for s in ev["segs"] if any(s[1] <= t <= s[2] for t in span)]
    pitch = [x["pitch"] for x in ev["lines"]]
    turns = math.floor(abs(pitch[b] - pitch[a]) / w.f["turn_deg"] + 0.5)
    acts += ["turn_down" if pitch[b] > pitch[a] else "turn_up"] * turns
    return acts, bool(acts) and not ev["foreign"] & set(range(a, b))


def evidence(w: o.Work, o0: bool) -> dict:
    cut = o.cut_lines(w)
    kappa = [t for t in range(len(w.ep.script)) if t not in cut]
    gsegs, foreign = o.gui_segments(w, cut)
    segs = sorted(o.world_segments(w.ep.script, kappa) + gsegs, key=lambda s: s[1])
    foreign |= {t for t, c in enumerate(w.ep.script) if c == "q"}
    return {"cut": cut, "kappa": kappa, "segs": segs, "foreign": foreign,
            "omega": o.modes(w.ep.script, o0), "lines": lines(w.ep.script, w.ep.session)}  # fmt: skip


def states_of(w: o.Work, ev: dict, p: int) -> list[int]:
    return [ev["kappa"][p + i * w.f["delta"]] for i in range(w.f["states"])]


def starts(w: o.Work, ev: dict) -> list[int]:
    """clips (3), by trying every p."""
    d, n, found = w.f["delta"], w.f["states"], []
    for p in range(len(ev["kappa"]) - (n - 1) * d):
        st = states_of(w, ev, p)
        ok = all(interval(w, ev, a, b)[1] for a, b in itertools.pairwise(st))
        if ok and not any(ev["omega"][t] for t in st):
            found.append(p)
    return found


def accept(w: o.Work, states: list[int]) -> bool:
    """(5): classify the GUI states not yet classified, then vpt-gui-actions (8)."""
    flagged = [t for t in states if o.gui_flag(w.ep.script, t)]
    w.classify(flagged)
    return not any(w.cls[t] in o.NO_ELEMENT for t in flagged)


def sample(w: o.Work, ev: dict, valid: list[int]) -> list[int]:
    """clips (4) with Generator(PCG64([seed, 1, k])); the starts taken, in order."""
    g = np.random.Generator(np.random.PCG64([w.f["seed"], 1, w.k]))
    rest, taken = list(valid), []
    while len(taken) < w.f["per_episode"] and rest:
        p = rest[g.integers(len(rest))]
        st = states_of(w, ev, p)
        if accept(w, st):
            taken.append(p)
            rest = [q for q in rest if states_of(w, ev, q)[-1] < st[0] or st[-1] < states_of(w, ev, q)[0]]  # fmt: skip
        else:
            rest.remove(p)
    return taken


def clip_row(w: o.Work, ev: dict, subset: str, p: int) -> dict:
    """clips (5) and the fields of clip.json, in the order of Datasets."""
    st = states_of(w, ev, p)
    acts = [interval(w, ev, a, b)[0] for a, b in itertools.pairwise(st)]
    gui = [o.gui_flag(w.ep.script, t) for t in st]
    return {"id": f"{w.ep.version}_{w.ep.stem}_{st[0]}", "subset": subset,
            "relpath": w.ep.relpath, "version": w.ep.version, "states": st,
            "actions": acts, "n": [len(a) for a in acts], "gui": gui,
            "classes": [w.cls[t] if g else None for t, g in zip(st, gui)],
            "pitch": [ev["lines"][t]["pitch"] for t in st]}  # fmt: skip


def clip_files(e: Ep, row: dict) -> dict:
    """frames.yuv: the state frames in order; lines.jsonl: the span's lines as in J_e."""
    st, frames = row["states"], e.frames()
    pieces = e.files()[".jsonl"].split(b"\n")[st[0] : st[-1] + 1]
    return {
        f"clips/{row['id']}/frames.yuv": b"".join(frames[t] for t in st),
        f"clips/{row['id']}/lines.jsonl": b"".join(x + b"\n" for x in pieces),
        f"clips/{row['id']}/clip.json": row,
    }


def process(e: Ep, f: dict, k: int, subset: str, broken) -> tuple[dict, list, list]:
    """(2)-(5) of the candidate of draw k: record fields, its clips in the order taken,
    its classifier calls."""
    fields = {"options": json.loads(e.files()["-options.json"]),
              "starts_session": e.session}  # fmt: skip
    if e.video is None or e.stem in broken:
        return fields | {"reason": "video"}, [], []
    w = o.Work(e, f, k)
    w.classify(o.needed(e.script))
    shown = not e.session and o.overlay_shown(f, e.overlay)
    ev = evidence(w, shown)
    valid = starts(w, ev)
    rows = [clip_row(w, ev, subset, p) for p in sample(w, ev, valid)]
    fields |= {"reason": None if rows else "no_clip", "overlay_start": shown,
               "valid_starts": len(valid)}  # fmt: skip
    return fields, rows, w.calls
