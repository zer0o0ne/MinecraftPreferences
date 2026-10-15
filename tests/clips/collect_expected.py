"""Expected outputs of `collect` (specs/perception-data.md) for a world of collect_world.py:
(1) the plan and the reasons, (2)-(5) per valid episode with collect_oracle.py, (6) the
commits; transcribed, nothing from src/."""

import itertools
import json
import math
from dataclasses import dataclass

import numpy as np

from . import collect_oracle as o
from .collect_world import Ep, lines

CHECKS = ("version", "missing", "empty", "malformed", "options")


@dataclass
class Expected:
    records: list  # episodes.jsonl
    clips: list  # clips.jsonl
    files: dict  # path under out_dir -> bytes (clips/, options/)
    calls: dict  # stem of a committed episode -> frames of its classifier calls
    r: dict  # stem -> draw position


def check(e: Ep, f: dict) -> str | None:
    """vpt-gui-frames (2) without no_gui."""
    version = tuple(map(int, e.version.split(".")))
    if version < tuple(map(int, f["min_version"].split("."))):
        return "version"
    files = e.files()
    opts, jl = files["-options.json"], files[".jsonl"]
    if opts is None or jl is None:
        return "missing"
    if not opts or not jl:
        return "empty"
    try:
        parsed = json.loads(opts)
        [json.loads(x) for x in jl.decode().split("\n")[:-1]]
    except ValueError:
        return "malformed"
    if any(parsed.get(k, object()) != v for k, v in f["required_options"].items()):
        return "options"
    return None


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
    """clips (4) with Generator(PCG64([seed, 1, r])); the starts taken, in order."""
    g = np.random.Generator(np.random.PCG64([w.f["seed"], 1, w.r]))
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
    """clips (5) and the clips.jsonl fields."""
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
    }


def process(e: Ep, f: dict, r: int, subset: str, broken) -> tuple[dict, list, list]:
    """(2)-(5): record fields, the clips in the order taken, the classifier calls."""
    fields = {"options": json.loads(e.files()["-options.json"]),
              "starts_session": e.session}  # fmt: skip
    if e.video is None or e.stem in broken:
        return fields | {"reason": "video"}, [], []
    w = o.Work(e, f, r)
    w.classify(o.needed(e.script))
    shown = not e.session and o.overlay_shown(f, e.overlay)
    ev = evidence(w, shown)
    valid = starts(w, ev)
    rows = [clip_row(w, ev, subset, p) for p in sample(w, ev, valid)]
    fields |= {"reason": None if rows else "no_clip", "overlay_start": shown,
               "valid_starts": len(valid)}  # fmt: skip
    return fields, rows, w.calls


def commit(out: Expected, s: str, e: Ep, f: dict, broken) -> bool:
    """(6) for the episode drawn next; whether it passed the checks."""
    r, reason = len(out.records), check(e, f)
    rec = {k: None for k in ("options", "starts_session", "overlay_start",
                             "valid_starts", "clips")}  # fmt: skip
    rec |= {"subset": s, "relpath": e.relpath, "reason": reason}
    if reason == "options":
        rec["options"] = json.loads(e.files()["-options.json"])
    if reason is None:
        fields, rows, out.calls[e.stem] = process(e, f, r, s, broken)
        rows = rows[: f["n_clips"] - len(out.clips)]
        rec |= fields | ({"clips": [x["id"] for x in rows]} if rows else {})
        out.clips += rows
        out.files |= {k: v for x in rows for k, v in clip_files(e, x).items()}
    if rec["reason"] not in CHECKS[:3]:
        out.files[f"options/{e.version}_{e.stem}-options.json"] = e.files()[
            "-options.json"
        ]
    out.records.append(rec)
    out.r[e.stem] = r
    return reason is None


def expected(world, f: dict, broken=frozenset()) -> Expected:
    """(1): draws until N clips are committed or every pool is empty; `broken`: stems
    whose decode fails on every attempt of pass 1."""
    rng = np.random.Generator(np.random.PCG64(f["seed"]))
    pools = {s: list(dict.fromkeys(e.relpath for e in es))
             for s, es in world.episodes.items()}  # fmt: skip
    out = Expected([], [], {}, {}, {})
    while len(out.clips) < f["n_clips"] and any(pools.values()):
        s = [s for s in pools if pools[s]][rng.integers(sum(map(bool, pools.values())))]
        while pools[s]:
            relpath = pools[s].pop(rng.integers(len(pools[s])))
            if commit(out, s, world.eps[s, relpath], f, broken):
                break
    return out
