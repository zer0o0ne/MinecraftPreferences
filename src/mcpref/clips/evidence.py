"""Notation of specs/clips.md: the evidence of an episode (S, C, F, Omega, rho), kappa."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mcpref.gui.actions import GuiActions
from mcpref.vpt.actions import (
    ActionsConfig,
    Segment,
    foreign_lines,
    mode_frames,
    riding_lines,
    segments,
)


@dataclass(frozen=True)
class Evidence:
    """Notation: lines l_t, segments S, cut lines C, foreign lines F, modes Omega_t,
    turn_deg rho."""

    lines: Sequence[Mapping]
    segments: list[Segment]
    cut: frozenset[int]
    foreign: frozenset[int]
    modes: np.ndarray
    turn_deg: float


def kept_lines(n: int, cut: Collection[int]) -> list[int]:
    """Notation: kappa, the lines 0..n-1 not in `cut`, ascending."""
    return [t for t in range(n) if t not in cut]


def kept(ev: Evidence) -> list[int]:
    """Interface: kappa_0 < ... < kappa_{|K|-1}, K = {0..T-1} minus C."""
    return kept_lines(len(ev.lines), ev.cut)


def world_segments(
    lines: Sequence[Mapping], kappa: Sequence[int], r_min: int
) -> list[Segment]:
    """Notation S: `segments` of the kept lines, first and last position mapped to kappa."""
    segs = segments([lines[k] for k in kappa], r_min)
    return [Segment(s.action, kappa[s.first], kappa[s.last]) for s in segs]


def evidence(
    lines: Sequence[Mapping],
    gui: GuiActions,
    start: tuple[bool, bool, int],
    actions: ActionsConfig,
) -> Evidence:
    """Interface: S (world and GUI segments merged by first line), C, F, Omega, rho."""
    world = world_segments(lines, kept_lines(len(lines), gui.cut), actions.r_min)
    segs = sorted(world + gui.segments, key=lambda s: s.first)
    flagged = np.flatnonzero(foreign_lines(lines) | riding_lines(lines))
    foreign = frozenset(flagged.tolist()) | gui.foreign
    modes = mode_frames(lines, start)
    return Evidence(lines, segs, gui.cut, foreign, modes, actions.turn_deg)
