"""Naive transcription of (2)-(5) of specs/perception-data.md, a candidate's work, on the
episodes of collect_world.py. The functions the block composes (specs/vpt-actions.md,
vpt-gui-actions.md, clips.md, f3-overlay.md) are transcribed only for what those episodes
contain: world lines hold W (one move value; no effect, hotbar change, yaw turn, F1, F5 or
F3 combo), presses only over slot 1 of `inv` and `box`, the overlay box of frame 0 of a
uniform luma."""

from dataclasses import dataclass, field

from .collect_world import GUI, LUMA, Ep

NO_ELEMENT = ("other", "pause")  # e_t is none; a span whose first frame is one is cut


@dataclass
class Work:
    """The work of the candidate of draw k; `f`: every config value."""

    ep: Ep
    f: dict
    k: int
    cls: dict = field(default_factory=dict)  # frame -> class
    calls: list = field(default_factory=list)  # frames of each classifier call

    def classify(self, frames) -> None:
        """(3), (5): the frames not yet classified, in order, in chunks."""
        todo, b = [t for t in frames if t not in self.cls], self.f["classifier_batch"]
        for chunk in (todo[i : i + b] for i in range(0, len(todo), b)):
            self.calls.append(chunk)
            self.cls.update({t: self.ep.cls(t) for t in chunk})


def spans(script: str) -> list[tuple[int, int]]:
    edges = [t for t in range(len(script) + 1) if (t < len(script) and script[t] in GUI)
             != (t > 0 and script[t - 1] in GUI)]  # fmt: skip
    return [(k, j - 1) for k, j in zip(edges[::2], edges[1::2])]


def first_frame(k: int, j: int) -> int:
    return k + 1 if j > k else k


def needed(script: str) -> list[int]:
    firsts = {first_frame(k, j) for k, j in spans(script)}
    return sorted(firsts | {t for t, c in enumerate(script) if c == "p"})


def gui_flag(script: str, t: int) -> bool:
    """isGuiOpen, false on the first line t >= 1 of a container span."""
    opens = t >= 1 and script[t - 1] not in GUI and script[t] == "c"
    return script[t] in GUI and not opens


def cut_lines(w: Work) -> set:
    spans_ = [s for s in spans(w.ep.script) if w.cls[first_frame(*s)] in NO_ELEMENT]
    return {t for k, j in spans_ for t in range(k, j + 1)}


def gui_segments(w: Work, cut: set) -> tuple[list, set]:
    """Clicks and double clicks on slot 1 per kept span; a press on no element is foreign."""
    segs, foreign, ms = [], set(), w.f["double_click_ms"]
    for k, j in spans(w.ep.script):
        presses = [t for t in range(k, j + 1) if w.ep.script[t] == "p" and t not in cut]
        on_slot = [t for t in presses if w.cls[t] not in NO_ELEMENT]
        foreign |= set(presses) - set(on_slot)
        i = 0
        while i < len(presses):
            t, nxt = presses[i], presses[i + 1] if i + 1 < len(presses) else None
            pair = t in on_slot and nxt in on_slot and 50 * (nxt - t) < ms
            if t in on_slot:
                segs.append(
                    ("double_click_1", t, nxt) if pair else ("left_click_1", t, t)
                )
            i += 2 if pair else 1
    return segs, foreign


def world_segments(script: str, kappa: list) -> list:
    """One forward segment per run of world lines among the kept lines."""
    segs = []
    for pos, t in enumerate(kappa):
        if script[t] not in GUI:
            if pos and script[kappa[pos - 1]] not in GUI:
                segs[-1] = ("forward", segs[-1][1], t)
            else:
                segs.append(("forward", t, t))
    return segs


def overlay_shown(f: dict, overlay: bool) -> bool:
    """f3-overlay (1)-(3) with P and Q inside the uniform box of luma LUMA[overlay]."""
    y = LUMA[overlay]
    s = [min(abs(y - f["text_luma"][m]) <= f["tolerance"][m], y <= f["box_max"][m])
         for m in ("normal", "dimmed")]  # fmt: skip
    return max(s) >= f["threshold"]


def modes(script: str, o0: bool) -> list[bool]:
    """vpt-actions (9): the overlay toggles at frame k + 1 after the release of F3 at k."""
    toggles = [
        k + 1 for k in range(1, len(script)) if script[k - 1] == "f" != script[k]
    ]
    return [o0 != (sum(x <= t for x in toggles) % 2 == 1) for t in range(len(script))]
