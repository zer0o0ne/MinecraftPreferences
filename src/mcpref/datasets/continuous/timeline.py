"""One kept grid and the existing evidence rules across source-file boundaries."""

import bisect
import hashlib
import itertools
import json
from collections import Counter

from mcpref.datasets.clips.episode import classify_frames
from mcpref.datasets.clips.evidence import evidence, kept
from mcpref.datasets.clips.sample import gui_flag
from mcpref.datasets.continuous.frames import decoded
from mcpref.datasets.vpt.actions import turns
from mcpref.gui.actions import excluded_states, gui_actions

META_FIELDS = ("inventory", "hotbar", "yaw", "pitch", "xpos", "ypos", "zpos", "mouse")


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def grid_intervals(ev, kappa, delta: int, unknown: list[int]) -> list[dict]:
    n = max(0, (len(kappa) - 1) // delta)
    hits = [[] for _ in range(n)]
    for segment in ev.segments:
        first = bisect.bisect_left(kappa, segment.first)
        last = bisect.bisect_right(kappa, segment.last) - 1
        if first <= last:
            for j in range(first // delta, min(last // delta + 1, n)):
                hits[j].append(segment.action)
    foreign, count = [], 0
    for t in range(len(ev.lines) + 1):
        foreign.append(count)
        count += t in ev.foreign
    result = []
    for j, actions in enumerate(hits):
        a, b = kappa[j * delta], kappa[(j + 1) * delta]
        actions += turns(ev.lines, a, b, ev.turn_deg)
        reasons = []
        if foreign[b] != foreign[a]:
            reasons.append("unsupported_action")
        if any(a < t <= b for t in unknown):
            reasons.append("unknown_source_gap")
        result.append(
            {
                "grid_index": j,
                "actions": actions,
                "n": len(actions),
                "known": not reasons,
                "valid": bool(actions) and not reasons,
                "reasons": reasons or ([] if actions else ["idle"]),
                "raw_span": [a, b],
            }
        )
    return result


def compile_timeline(pipe, rows, sources: list[dict], fragment: dict) -> dict:
    base = sources[0]["offset"]
    classes = {
        int(t) + s["offset"] - base: c
        for s in sources
        for t, c in s["validation"]["classes"].items()
    }
    gui = gui_actions(rows, classes, pipe.screens, pipe.double_click_ms)
    ev = evidence(
        rows, gui, tuple(sources[0]["validation"]["initial_modes"]), pipe.actions
    )
    kappa = kept(ev)
    unknown = [
        s["offset"] - base
        for prev, s in itertools.pairwise(sources)
        if rows[s["offset"] - base]["serverTick"]
        != rows[prev["offset"] - base + prev["rows"] - 1]["serverTick"] + 1
    ]
    stream_id = identity([fragment["fragment_id"], sources[0]["piece"]["piece_id"]])
    return {
        "id": stream_id,
        "rows": rows,
        "evidence": ev,
        "kappa": kappa,
        "grid": kappa[:: pipe.clips.delta],
        "classes": classes,
        "gui": gui,
        "intervals": grid_intervals(ev, kappa, pipe.clips.delta, unknown),
        "world_group": identity(
            [fragment["subset"], fragment["alias"], fragment["filename_session_id"]]
        ),
        "excluded": [],
        "observations": {},
        "locations": {},
        "origin": None,
        "counts": Counter(),
    }


def classified_batch(pipe, timeline: dict, source: dict, batch: dict, global_base: int):
    rows, classes, gui = timeline["rows"], timeline["classes"], timeline["gui"]
    todo = [t for t in batch if gui_flag(rows, t) and t not in classes]
    classify_frames(pipe, {t: frame for t, (_, frame) in batch.items()}, todo, classes)
    excluded = excluded_states(
        rows, {t: classes[t] for t in batch if t in classes}, gui.cut
    )
    for t, (j, frame) in batch.items():
        if timeline["evidence"].modes[t] or t in excluded:
            reason = "mode" if timeline["evidence"].modes[t] else "gui"
            timeline["excluded"].append(
                {"grid_index": j, "raw_index": t, "reason": reason}
            )
            continue
        row, flag = rows[t], gui_flag(rows, t)
        observation_id = f"{timeline['id']}:{j}"
        observation = {
            "id": observation_id,
            "fragment_id": timeline["id"],
            "world_group": timeline["world_group"],
            "grid_index": j,
            "kept_ordinal": j * pipe.clips.delta,
            "raw_index": t,
            "source": source["piece"]["relpath"],
            "source_row": t + global_base - source["offset"],
            "gui": flag,
            "screen_class": classes[t] if flag else None,
            "meta": {k: row[k] for k in META_FIELDS},
        }
        if timeline["origin"] is None:
            timeline["origin"] = observation["meta"]
        timeline["observations"][j * pipe.clips.delta] = observation_id
        yield observation, frame


def observations(pipe, timeline: dict, sources: list[dict]):
    grid, global_base = timeline["grid"], sources[0]["offset"]
    for source in sources:
        start = source["offset"] - global_base
        lo, hi = (
            bisect.bisect_left(grid, start),
            bisect.bisect_left(grid, start + source["rows"]),
        )
        selected = {grid[j] - start: j for j in range(lo, hi)}
        if not selected:
            continue
        batch = {}
        for i, frame in decoded(
            pipe.cfg, source["target"] / "source.mp4", source["rows"] + 1
        ):
            if i in selected:
                batch[i + start] = (selected[i], frame)
            if len(batch) >= pipe.cfg.classifier_batch:
                yield from classified_batch(pipe, timeline, source, batch, global_base)
                batch.clear()
        yield from classified_batch(pipe, timeline, source, batch, global_base)
