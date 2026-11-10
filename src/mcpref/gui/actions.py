"""Interface of specs/vpt-gui-actions.md: the GUI actions of a VPT episode; (3)-(7) in
gui/kept_span.py."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mcpref.config import positive_int, read_yaml
from mcpref.datasets.vpt.actions import WORLD_ACTIONS, Segment
from mcpref.datasets.vpt.video.channels import runs
from mcpref.gui.kept_span import kept_span, new_digits, pressed
from mcpref.gui.screens import ScreenClass

NO_SCREEN = frozenset({"other", "pause", "death"})
CLICKS = ("left_click", "right_click", "shift_click", "double_click")


@dataclass(frozen=True)
class GuiActionsConfig:
    """Interface: configs/datasets/vpt/gui-actions.yaml."""

    double_click_ms: int


@dataclass(frozen=True)
class GuiActions:
    """Interface: segments (3)-(6) in the order of (9), cut lines (2), foreign lines."""

    segments: list[Segment]
    cut: frozenset[int]
    foreign: frozenset[int]


def load_config(path: Path) -> GuiActionsConfig:
    """Interface: the YAML file loaded into GuiActionsConfig; violated constraints raise."""
    cfg = GuiActionsConfig(**read_yaml(path))
    positive_int("double_click_ms", cfg.double_click_ms)
    return cfg


def gui_action_names(screens: Mapping[str, ScreenClass]) -> tuple[str, ...]:
    """Interface: close, then the four click kinds on elements 1..n_max."""
    n_max = max(e.index for cls in screens.values() for e in cls.elements)
    return ("close", *(f"{c}_{i}" for c in CLICKS for i in range(1, n_max + 1)))


def action_names(screens: Mapping[str, ScreenClass]) -> tuple[str, ...]:
    """Interface: the order of A, the world actions, then the GUI actions."""
    return WORLD_ACTIONS + gui_action_names(screens)


def spans(lines: Sequence[Mapping]) -> list[tuple[int, int]]:
    """Notation: the maximal runs [k, j] of lines with isGuiOpen true, in order."""
    return [(p, q) for p, q, gui in runs([x["isGuiOpen"] for x in lines]) if gui]


def first_frame(k: int, j: int) -> int:
    """Notation: u = k + 1 if j > k, else k."""
    return k + 1 if j > k else k


def opens_container(lines: Sequence[Mapping], k: int) -> bool:
    """First-line rule: GUI line k >= 1 starts a span with isGuiInventory false."""
    return k >= 1 and not lines[k - 1]["isGuiOpen"] and not lines[k]["isGuiInventory"]


def needs_class(line: Mapping) -> bool:
    """(1): N^B_t meets {0, 1} or N_t meets D."""
    return bool(pressed(line) or new_digits(line))


def needed_frames(
    lines: Sequence[Mapping], *, gui_spans: Sequence[tuple[int, int]] | None = None
) -> list[int]:
    """Interface (1): first frames of the spans and their lines with a press or digit."""
    span_list = spans(lines) if gui_spans is None else gui_spans
    frames = set()
    for k, j in span_list:
        frames.add(first_frame(k, j))
        frames.update(t for t in range(k, j + 1) if needs_class(lines[t]))
    return sorted(frames)


def gui_actions(
    lines: Sequence[Mapping],
    classes: Mapping[int, str],
    screens: Mapping[str, ScreenClass],
    double_click_ms: int,
    *,
    gui_spans: Sequence[tuple[int, int]] | None = None,
    frames: Sequence[int] | None = None,
) -> GuiActions:
    """Interface: (2) cut spans; (3)-(7) on the kept ones, in line order (9); the
    catalogue class of c_t, none for other, pause and death (no element)."""
    span_list = spans(lines) if gui_spans is None else gui_spans
    needed = needed_frames(lines, gui_spans=span_list) if frames is None else frames
    c = {t: classes[t] for t in needed}
    cut, segs, foreign = set(), [], set()
    for k, j in span_list:
        if c[first_frame(k, j)] in NO_SCREEN:
            cut.update(range(k, j + 1))
            continue
        g = k + opens_container(lines, k)
        on = (t for t in range(g, j + 1) if needs_class(lines[t]))
        cls = {t: None if c[t] in NO_SCREEN else screens[c[t]] for t in on}
        span_segs, span_foreign = kept_span(lines, cls, double_click_ms, g, j)
        segs += span_segs
        foreign |= span_foreign
    return GuiActions(segs, frozenset(cut), frozenset(foreign))


def excluded_states(
    lines: Sequence[Mapping], classes: Mapping[int, str], cut: Collection[int]
) -> frozenset[int]:
    """Interface (8): GUI states of kept spans with c_t other, pause or death."""
    return frozenset(
        t
        for t, c in classes.items()
        if c in NO_SCREEN
        and lines[t]["isGuiOpen"]
        and t not in cut
        and not opens_container(lines, t)
    )
