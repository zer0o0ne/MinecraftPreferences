"""Independent old interval oracle and kept-grid boundary cases."""

from types import SimpleNamespace

from datasets.clips.clip_world import ACTIONS, OFF, gui, random_episode
from datasets.vpt.action_lines import episode

from mcpref.datasets.clips.evidence import evidence, kept
from mcpref.datasets.clips.validity import interval
from mcpref.datasets.continuous.timeline import compile_timeline, grid_intervals


def test_grid_actions_match_existing_scalar_oracle():
    for seed in range(141, 149):
        lines, g, start = random_episode(seed, 90)
        ev = evidence(lines, g, start, ACTIONS)
        kappa = kept(ev)
        result = grid_intervals(ev, kappa, 6, [])
        expected = [interval(ev, kappa, j, 6) for j in range(0, len(kappa) - 6, 6)]
        assert [(r["actions"], r["valid"]) for r in result] == expected


def test_unknown_join_masks_only_interval_crossing_it():
    ev = evidence(episode("w w w w w w w w w"), gui(), OFF, ACTIONS)
    result = grid_intervals(ev, kept(ev), 2, [4])
    assert [r["valid"] for r in result] == [True, False, True, True]
    assert result[1]["reasons"] == ["unknown_source_gap"]


def test_modes_and_held_actions_do_not_reset_at_source_boundary():
    lines = episode("w w w+f1 w w w+f1 w w")
    pipe = SimpleNamespace(
        screens={}, double_click_ms=250, actions=ACTIONS, clips=SimpleNamespace(delta=2)
    )
    sources = [
        {
            "offset": 0,
            "rows": 4,
            "piece": {"piece_id": "a"},
            "validation": {"classes": {}, "initial_modes": OFF},
        },
        {
            "offset": 4,
            "rows": 4,
            "piece": {"piece_id": "b"},
            "validation": {"classes": {}, "initial_modes": OFF},
        },
    ]
    timeline = compile_timeline(
        pipe,
        lines,
        sources,
        {"fragment_id": "f", "subset": "s", "alias": "p", "filename_session_id": "r"},
    )
    assert timeline["evidence"].modes.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert [r["actions"] for r in timeline["intervals"]] == [["forward"]] * 3
    assert timeline["grid"] == [0, 2, 4, 6]
