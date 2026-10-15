"""Labels (1), hotbar (5), inventory (6), the order of (8), (10), (11), (13) and the config of
specs/vpt-actions.md."""

import dataclasses

import numpy as np
import pytest
import yaml
from action_lines import CUSTOM, bump, episode, repeat

from mcpref.vpt.actions import (
    WORLD_ACTIONS,
    ActionsConfig,
    Segment,
    foreign_lines,
    load_config,
    riding_lines,
    segments,
    starts_session,
)

S = Segment
MOVES = ["forward", "back", "left", "right", "forward_left", "forward_right", "back_left",
         "back_right"]  # fmt: skip
RIDING = [
    "boat_one_cm",
    "horse_one_cm",
    "pig_one_cm",
    "minecart_one_cm",
    "strider_one_cm",
]


def test_world_actions_in_spec_order():
    assert WORLD_ACTIONS == (*MOVES, "jump", *("jump_" + m for m in MOVES), "sneak",
                             "turn_left", "turn_right", "turn_up", "turn_down", "attack", "use",
                             "hotbar_left", "hotbar_right", "open_inventory")  # fmt: skip
    assert len(WORLD_ACTIONS) == 27


def test_segment_is_frozen_with_spec_fields():
    assert [f.name for f in dataclasses.fields(Segment)] == ["action", "first", "last"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        S("use", 0, 0).first = 1


# (1): f = [w] - [s], r = [d] - [a], j = [space]; W+S and A+D cancel; sprint ignored.
LABELS = [("w", "forward"), ("s", "back"), ("a", "left"), ("d", "right"),
          ("w+a", "forward_left"), ("w+d", "forward_right"), ("s+a", "back_left"),
          ("s+d", "back_right"), ("space", "jump"), ("w+space", "jump_forward"),
          ("s+space", "jump_back"), ("a+space", "jump_left"), ("d+space", "jump_right"),
          ("w+a+space", "jump_forward_left"), ("w+d+space", "jump_forward_right"),
          ("s+a+space", "jump_back_left"), ("s+d+space", "jump_back_right"),
          ("w+s+d", "right"), ("w+a+d", "forward"), ("w+s+a+d+space", "jump"),
          ("w+left.control", "forward"), ("s+a+d+space", "jump_back")]  # fmt: skip


@pytest.mark.parametrize("keys, name", LABELS)
def test_move_label(keys, name):
    assert segments(episode(repeat(keys, 3)), r_min=3) == [S(name, 0, 2)]


@pytest.mark.parametrize(
    "keys", ["-", "w+s", "a+d", "w+s+a+d", "left.control", "c", "M2"]
)
def test_no_world_action(keys):
    assert segments(episode(repeat(keys, 3)), r_min=3) == []


def test_sneak_is_its_own_channel():
    lines = episode("w w+left.shift w+left.shift w w - left.shift")
    assert segments(lines, r_min=3) == [
        S("forward", 0, 4),
        S("sneak", 1, 2),
        S("sneak", 6, 6),
    ]
    assert segments(episode("w+s+left.shift a+d+left.shift"), r_min=3) == [
        S("sneak", 0, 1)
    ]


def test_attack_and_use():
    assert segments(episode("M0 M0+M1 M1 M1"), r_min=3) == [
        S("attack", 0, 1),
        S("use", 1, 3),
    ]


@pytest.mark.parametrize("token", ["left.shift", "M0", "M1"])
def test_no_smoothing_outside_move(token):
    lines = episode(f"{token} - {token} {token}")
    assert [(s.first, s.last) for s in segments(lines, r_min=3)] == [(0, 0), (2, 3)]


@pytest.mark.parametrize("gui", ["#", "@"])
def test_gui_line_ends_every_run(gui):
    held = "w+left.shift+M0+M1"
    found = segments(episode(f"{held} {gui}:{held} {held}"), r_min=3)
    assert found == [
        S(a, k, k) for k in (0, 2) for a in ("forward", "sneak", "attack", "use")
    ]


# (5): delta = (h1 - h0) mod 9; delta right if delta <= 4, else 9 - delta left.
@pytest.mark.parametrize("h0, h1, action, n", [(0, 8, "hotbar_left", 1), (0, 4, "hotbar_right", 4),
                         (0, 5, "hotbar_left", 4), (8, 0, "hotbar_right", 1),
                         (3, 1, "hotbar_left", 2), (1, 7, "hotbar_left", 3),
                         (6, 1, "hotbar_right", 4)])  # fmt: skip
def test_hotbar(h0, h1, action, n):
    assert (
        segments(episode("- - -", hotbar=[h0, h0, h1]), r_min=3)
        == [S(action, 1, 1)] * n
    )


@pytest.mark.parametrize("tokens, expected", [("# -", []), ("@ -", []),
                         ("- #", [S("hotbar_right", 0, 0)]), ("- -", [S("hotbar_right", 0, 0)])])  # fmt: skip
def test_hotbar_needs_a_world_line(tokens, expected):
    assert segments(episode(tokens, hotbar=[3, 4]), r_min=3) == expected


@pytest.mark.parametrize("tokens, firsts", [("- e e", [1]), ("- e e!", [1, 2]),
                         ("- @:e #:e", []), ("- - e", [2]), ("e - -", [0])])  # fmt: skip
def test_open_inventory(tokens, firsts):
    assert segments(episode(tokens), r_min=3) == [
        S("open_inventory", k, k) for k in firsts
    ]


def test_order_by_first_line_then_channel():
    lines = episode("M1 M1+w w+M0 e e+w+left.shift+M0+M1 -", hotbar=[0, 0, 0, 0, 0, 8])
    assert segments(lines, r_min=1) == [
        S("use", 0, 1), S("forward", 1, 2), S("attack", 2, 2), S("open_inventory", 3, 3),
        S("forward", 4, 4), S("sneak", 4, 4), S("attack", 4, 4), S("use", 4, 4),
        S("hotbar_left", 4, 4),
    ]  # fmt: skip


def test_foreign_lines():
    lines = episode("q - f #:q @:f f3+q f3+f q q q! - f+e")
    found = foreign_lines(lines)
    assert found.dtype == np.bool_ and found.shape == (12,)
    assert found.tolist() == [True, False, True, False, False, False, False, True, False,
                              True, False, True]  # fmt: skip


@pytest.mark.parametrize("stat", RIDING)
def test_riding_lines(stat):
    found = riding_lines(bump(episode("- - - #"), CUSTOM + stat, 1, 2))
    assert found.dtype == np.bool_ and found.tolist() == [False, True, True, False]


@pytest.mark.parametrize("stat", ["walk_one_cm", "fly_one_cm", "aviate_one_cm", "jump"])
def test_other_stats_are_not_riding(stat):
    assert (
        riding_lines(bump(episode("- - -"), CUSTOM + stat, 0, 1)).tolist()
        == [False] * 3
    )


@pytest.mark.parametrize("empty, expected", [(0, True), (1, False), (None, False)])
def test_starts_session(empty, expected):
    lines = episode("- - -")
    if empty is not None:
        lines[empty]["stats"] = {}
    assert starts_session(lines) is expected


def write(tmp_path, **fields) -> object:
    path = tmp_path / "vpt-actions.yaml"
    path.write_text(yaml.safe_dump({"turn_deg": 10, "r_min": 3} | fields))
    return path


def test_approved_config_has_starting_values(repo):
    cfg = load_config(repo / "configs/vpt-actions.yaml")
    assert isinstance(cfg, ActionsConfig)
    assert [f.name for f in dataclasses.fields(cfg)] == ["turn_deg", "r_min"]
    assert (cfg.turn_deg, cfg.r_min) == (10.0, 3)
    assert type(cfg.turn_deg) is float and type(cfg.r_min) is int
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.r_min = 4


@pytest.mark.parametrize("turn_deg", [2.5, 1, 1e-3])
def test_turn_deg_read_as_float(tmp_path, turn_deg):
    cfg = load_config(write(tmp_path, turn_deg=turn_deg, r_min=1))
    assert type(cfg.turn_deg) is float and cfg.turn_deg == turn_deg and cfg.r_min == 1


@pytest.mark.parametrize("fields", [{"turn_deg": 0}, {"turn_deg": 0.0}, {"turn_deg": -10},
                         {"turn_deg": "10"}, {"turn_deg": True}, {"turn_deg": float("nan")},
                         {"r_min": True}, {"r_min": 0}, {"r_min": -1}, {"r_min": 3.0},
                         {"r_min": "3"}])  # fmt: skip
def test_config_constraints(tmp_path, fields):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, **fields))
