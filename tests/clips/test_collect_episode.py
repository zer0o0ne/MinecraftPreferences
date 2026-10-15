"""(3)-(5) of specs/perception-data.md on one-episode worlds, by hand (delta 2, 3 states,
turn_deg 4, double_click_ms 250): the start of the modes (session start, the overlay on
frame 0), the composed actions in the clips, cut spans and foreign lines."""

import pytest

from .collect_world import Ep, jsonl


def one(make_site, script: str, **kw):
    """A site serving the one episode data/7.0/a.mp4."""
    return make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", script, **kw)]})


@pytest.mark.parametrize(
    "session, overlay, start, valid",
    [
        (True, True, False, 4),  # a session start: all modes off, frame 0 unread
        (
            False,
            True,
            True,
            0,
        ),  # the overlay on frame 0 stays on: every state has a mode
        (False, False, False, 4),
        (True, False, False, 4),
    ],
)
def test_start_of_the_modes(make_site, session, overlay, start, valid):
    """`wwwwwwww`: starts 0..3 without modes."""
    site = one(make_site, "w" * 8, session=session, overlay=overlay)
    site.run()
    (rec,) = site.records()
    assert rec["starts_session"] is session and rec["overlay_start"] is start
    assert rec["valid_starts"] == valid
    assert rec["reason"] == (None if valid else "no_clip")


@pytest.mark.parametrize("overlay, valid", [(True, 2), (False, 0)])
def test_f3_release_toggles_the_overlay_read_from_frame_0(make_site, overlay, valid):
    """`wwfwwwwwww`: F3 released at line 3 toggles the overlay at frame 4; shown on frame 0,
    it is off from frame 4 (starts 4, 5); off on frame 0, it is on from frame 4 (none)."""
    site = one(make_site, "wwfwwwwwww", overlay=overlay)
    site.run()
    (rec,) = site.records()
    assert rec["overlay_start"] is overlay and rec["valid_starts"] == valid
    if valid:
        assert {tuple(r["states"]) for r in site.rows()} <= {
            (4, 6, 8),
            (5, 7, 9),
        }


def test_overlay_read_with_the_overlay_config(make_site):
    """text_luma 100: frame 0 (box luma 200) does not show the overlay."""
    site = one(make_site, "w" * 8, overlay=True)
    site.sub["text_luma"] = {"normal": 100, "dimmed": 60}
    site.run()
    assert site.records()[0]["overlay_start"] is False


@pytest.mark.parametrize(
    "ms, actions",
    [
        (250, [["forward", "double_click_1"], ["double_click_1", "forward"]]),
        (50, [["forward", "left_click_1"], ["left_click_1", "forward"]]),  # 50 ms apart
    ],
)
def test_gui_actions_in_the_intervals(make_site, ms, actions):
    """`wppww`: presses over slot 1 at lines 1, 2; the pair lies in both intervals."""
    site = one(make_site, "wppww")
    site.run(double_click_ms=ms)
    (row,) = site.rows()
    assert row["states"] == [0, 2, 4] and row["actions"] == actions
    assert row["n"] == [2, 2] and row["gui"] == [False, True, False]


def test_click_and_world_actions(make_site):
    """`wpwiw`: interval (0, 2) holds W at 0 and the click at 1; (2, 4) W at 2."""
    site = one(make_site, "wpwiw")
    site.run()
    (row,) = site.rows()
    assert row["actions"] == [["forward", "left_click_1"], ["forward"]]
    assert row["n"] == [2, 1]


def test_pitch_turns_use_turn_deg_of_the_actions_config(make_site):
    """`wwvww`: pitch 0.01 t, 12 degrees higher from line 2: 12.02 / 4 -> 3 turns."""
    site = one(make_site, "wwvww")
    site.run()
    (row,) = site.rows()
    assert row["actions"] == [["forward"] + ["turn_down"] * 3, ["forward"]]
    assert row["n"] == [4, 1] and row["pitch"] == [0.0, 12.02, 12.04]


@pytest.mark.parametrize("cls", ["other", "pause"])
def test_cut_span_leaves_the_timeline_but_not_the_lines(make_site, cls):
    """`wiiiwwww`: the span 1..3 (u = 2) is cut; kept lines 0, 4, 5, 6, 7, one start; the
    world lines around the cut make one segment; lines.jsonl holds lines 0..7."""
    site = one(make_site, "wiiiwwww", classes={2: cls})
    gamma = site.run()
    (row,) = site.rows()
    assert row["states"] == [0, 5, 7] and row["actions"] == [["forward"], ["forward"]]
    lines = (site.out / "shards/000000/clips" / row["id"] / "lines.jsonl").read_bytes()
    assert lines == jsonl("wiiiwwww", False)
    assert gamma.by_stem() == {"a": [[2]]}


def test_foreign_line_invalidates_its_intervals(make_site):
    """`wwqwwwwww`: Q at line 2 excludes the intervals from 1 and 2; starts 3, 4."""
    site = one(make_site, "wwqwwwwww")
    site.run(per_episode=5)
    (rec,) = site.records()
    assert rec["valid_starts"] == 2
    assert all(r["states"][0] in (3, 4) for r in site.rows())


@pytest.mark.parametrize("cls, valid", [("other", 3), ("inv", 6)])
def test_press_on_no_element_is_foreign(make_site, cls, valid):
    """`wwiipwwwwwww`: interval 2 (lines 2, 3) has no action; on an `other` frame the
    press at 4 is foreign and excludes intervals 3, 4 (starts 5, 6, 7), on `inv` it is a
    click (starts all but 0, 2)."""
    site = one(make_site, "wwiipwwwwwww", classes={4: cls})
    site.run(per_episode=5)
    (rec,) = site.records()
    assert rec["valid_starts"] == valid


def test_too_short_an_episode_is_no_clip(make_site):
    site = one(make_site, "wwww")
    site.run()
    (rec,) = site.records()
    assert rec["reason"] == "no_clip" and rec["valid_starts"] == 0
