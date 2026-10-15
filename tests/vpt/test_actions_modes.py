"""Modes (9) of specs/vpt-actions.md, hand cases; T, F mark Omega_t per frame."""

import numpy as np
import pytest
from action_lines import episode

from mcpref.vpt.actions import mode_frames

OFF = (False, False, 0)
C3 = ["a", "b", "c", "d", "f", "g", "h", "i", "n", "p", "q", "t", "f4"]


def omega(tokens: str, start=OFF) -> str:
    found = mode_frames(episode(tokens), start)
    assert found.dtype == np.bool_ and found.shape == (len(tokens.split()),)
    return "".join("T" if x else "F" for x in found)


CASES = [  # (tokens, start, Omega)
    # An F3 release at line k toggles the overlay at frame k + 1 (lines 950-953 of [R3]).
    ("- - f3 f3 f3 f3 - - - -", OFF, "FFFFFFFTTT"),
    ("- f3 - - f3 - - -", OFF, "FFFTTTFF"),
    ("- f3 - - -", (True, False, 0), "TTTFF"),
    ("- - f3 -", OFF, "FFFF"),  # the toggle falls after the last frame
    ("f3 f3 f3", OFF, "FFF"),
    ("f3 - -", OFF, "FFT"),
    # The release counts when line k - 1 passes events: the world or the player inventory.
    ("- @:f3 @:f3 @ - -", OFF, "FFFFTT"),
    ("- #:f3 #:f3 # - -", OFF, "FFFFFF"),
    ("- f3 f3 # # #", OFF, "FFFFTT"),
    ("- #:f3 - - -", OFF, "FFFFF"),
    ("- @:f3 # # #", OFF, "FFFTT"),
    # The combo flag: a C3 key pressed in the world with F3 held; the release clears it.
    ("- f3 f3+a f3 - - -", OFF, "FFFFFFF"),
    ("- f3 f3+a f3 - - f3 - - -", OFF, "FFFFFFFFTT"),
    (
        "- f3 f3+a #:f3 # - f3 - f3 - - -",
        OFF,
        "FFFFFFFFFFTT",
    ),  # survives a chest release
    ("- f3 f3+a @:f3 @ - - -", OFF, "FFFFFFFF"),  # cleared by an inventory release
    ("a a+f3 a+f3 - -", OFF, "FFFFT"),  # a held before F3: not pressed with F3 held
    ("- a+f3 - -", OFF, "FFFF"),  # pressed on the line F3 is pressed
    ("a a+f3 a!+f3 - -", OFF, "FFFFF"),  # an auto-repeat press is a press
    ("- @:f3 @:f3+a @ - -", OFF, "FFFFTT"),  # not set in a GUI
    ("- f3 f3+a f3+b f3 - f3 - -", OFF, "FFFFFFFFT"),  # two combos, one flag
    # F1 in the world toggles the HUD at frame k + 2.
    ("- - f1 - - -", OFF, "FFFFTT"),
    ("- f1 f1 f1 - -", OFF, "FFFTTT"),
    ("- f1 - f1 - - -", OFF, "FFFTTFF"),
    ("- @:f1 - - -", OFF, "FFFFF"),
    ("- #:f1 - - -", OFF, "FFFFF"),
    ("- - f1 -", OFF, "FFFF"),
    ("- f3 f3+f1 f3 - - -", OFF, "FFFFTTT"),  # F1 is no combo key: both modes switch
    ("- f1 - -", (False, True, 0), "TTTF"),
    # F5 on a world line steps the view at frame k + 1, modulo 3; a container screen, the
    # player inventory included, consumes it.
    ("- f5 - f5 - f5 - -", OFF, "FFTTTTFF"),
    ("- @:f5 - -", OFF, "FFFF"),
    ("- #:f5 - -", OFF, "FFFF"),
    ("- @:f5 f5 - -", OFF, "FFFFF"),  # held from the GUI into the world: no new press
    ("- f5 @ @", OFF, "FFTT"),  # the step stands when a GUI opens on the next line
    ("- f3 f3+f5 f3 - - -", OFF, "FFFTTTT"),  # F5 is no combo key: both modes switch
    ("f5 f5 f5", OFF, "FTT"),
    ("- f5 - f5 -", (False, False, 1), "TTTTF"),
    ("- f5 - -", (False, False, 2), "TTFF"),
    ("- - -", (False, True, 0), "TTT"),
    ("- - -", (False, False, 1), "TTT"),
    ("- - -", (True, True, 2), "TTT"),
    # Omega is the union of the three modes.
    ("- f3 f1 - - -", (True, False, 0), "TTTFTT"),
    ("- f5 - f5 - f5 f3 - f1 - - -", OFF, "FFTTTTFFTTTT"),
]


@pytest.mark.parametrize("tokens, start, expected", CASES)
def test_mode_frames(tokens, start, expected):
    assert omega(tokens, start) == expected


@pytest.mark.parametrize("key", C3)
def test_combo_keys(key):
    assert omega(f"- f3 f3+{key} f3 - - -") == "FFFFFFF"


@pytest.mark.parametrize("key", ["w", "s", "e", "space", "k", "left.shift", "f2", "j"])
def test_other_keys_with_f3_held(key):
    assert omega(f"- f3 f3+{key} f3 - - -") == "FFFFFTT"
