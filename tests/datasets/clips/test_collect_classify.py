"""(3), (5) of specs/perception-data.md: calls of the classifier gamma (needed frames, then
the GUI states of each drawn clip; at most once per frame, chunks of classifier_batch, no
empty call, RGB fallback or the classifier's YUV capability, serialised) and `accept`
rejecting a clip."""

from types import SimpleNamespace

import numpy as np
import pytest

from datasets.clips.collect_gamma import Classifier
from datasets.clips.collect_world import Ep
from datasets.vpt.action_lines import episode as action_episode
from mcpref.datasets.clips.episode import accept, classify_frames, serialised
from mcpref.gui.actions import excluded_states


@pytest.mark.parametrize("batch", [1, 2, 3, 64])
def test_calls_are_the_needed_then_the_state_chunks(site, batch):
    """One worker: per candidate, exactly the calls of the transcription."""
    gamma = site.run(classifier_batch=batch)
    ex = site.expected(classifier_batch=batch)
    assert gamma.bad == []
    assert gamma.by_stem() == {s: c for s, c in ex.calls.items() if c}
    assert all(1 <= len(c) <= batch for c in gamma.calls)


def test_each_frame_is_classified_at_most_once_with_three_workers(site):
    gamma = site.run(workers=3)
    for stem, calls in gamma.by_stem().items():
        frames = [t for c in calls for t in c]
        assert len(frames) == len(set(frames)), stem
    assert all(1 <= len(c) <= 2 for c in gamma.calls)
    written = {s: c for s, c in site.expected().calls.items() if c}
    assert {s: gamma.by_stem()[s] for s in written} == written


def test_calls_are_serialised_across_workers(site):
    gamma = Classifier(site.world, pause=0.03)
    site.run(gamma, workers=3)
    assert gamma.max_active == 1 and len(gamma.calls) >= 6


def test_classifier_gets_the_rgb_view_of_each_frame(make_site):
    """gamma's frames: a list of H x W x 3 uint8 arrays equal to vpt-gui-frames (5) of the
    decoded frame (the fake finds each one by its exact view; random chroma)."""
    site = make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", "wiiwwwiiw")]})
    gamma = site.run(per_episode=3)
    assert gamma.bad == [] and gamma.calls[0] == [("a", 2), ("a", 7)]
    assert gamma.by_stem() == site.expected(per_episode=3).calls


def test_yuv_capability_gets_raw_frames_in_the_same_chunks():
    """A capable gamma gets raw YUV, not an RGB conversion; serialised keeps its capability."""

    class Gamma:
        def __init__(self):
            self.calls = []

        def __call__(self, frames):
            raise AssertionError("RGB fallback used for a YUV-capable classifier")

        def classify_yuv(self, frames):
            self.calls.append(frames)
            return [f"class{x[0]}" for x in frames]

    gamma = Gamma()
    cfg = SimpleNamespace(classifier_batch=2, width=4, height=4)
    pipe = SimpleNamespace(cfg=cfg, classify=serialised(gamma))
    frames = {t: bytes([t]) * 24 for t in (1, 2, 3)}
    classes = {}
    classify_frames(pipe, frames, [3, 1, 2], classes)
    assert gamma.calls == [[frames[3], frames[1]], [frames[2]]]
    assert classes == {3: "class3", 1: "class1", 2: "class2"}


def test_accept_retains_previous_gui_exclusions():
    class Gamma:
        def classify_yuv(self, frames):
            return ["other" if x == b"\x01" else "inv" for x in frames]

    pipe = SimpleNamespace(cfg=SimpleNamespace(classifier_batch=2), classify=Gamma())
    lines, frames, classes, excluded = (
        action_episode("w @ @ w"),
        {1: b"\x01", 2: b"\x02"},
        {},
        set(),
    )
    cut = frozenset()
    assert not accept(pipe, lines, frames, classes, cut, [1], excluded=excluded)
    assert not accept(pipe, lines, frames, classes, cut, [1, 2], excluded=excluded)
    assert accept(pipe, lines, frames, classes, cut, [2], excluded=excluded)
    assert excluded == set(excluded_states(lines, classes, cut)) == {1}


def test_episode_computes_gui_spans_once(make_site, monkeypatch):
    from mcpref.datasets.clips import episode
    from mcpref.gui import actions

    original, calls = actions.spans, []

    def counted(lines):
        calls.append(len(lines))
        return original(lines)

    monkeypatch.setattr(actions, "spans", counted)
    monkeypatch.setattr(episode, "spans", counted)
    site = make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", "wiiwwwiiw")]})
    site.run()
    assert calls == [9]


def rejecting(cls: str) -> dict:
    """`wwwwii` (delta 2, 3 states): starts p = 0 (states 0, 2, 4; frame 4 answered `cls`)
    and p = 1 (states 1, 3, 5; frame 5 is the span's first frame u); draw 0."""
    return {"6xx": [Ep("data/7.0/a.mp4", "wwwwii", classes={4: cls})]}


def first_draw(seed: int) -> int:
    """The first position the episode generator of draw 0 picks among 2 starts."""
    return int(np.random.Generator(np.random.PCG64([seed, 1, 0])).integers(2))


@pytest.mark.parametrize("cls", ["other", "pause"])
def test_accept_rejects_a_clip_with_an_excluded_state(make_site, cls):
    seed = next(s for s in range(100) if first_draw(s) == 0)
    site = make_site(episodes=rejecting(cls))
    gamma = site.run(seed=seed)
    (record,) = site.records()
    assert record["valid_starts"] == 2 and record["clips"] == ["7.0_a_1"]
    assert gamma.by_stem() == {"a": [[5], [4]]}  # needed u = 5; p = 0 rejected; p = 1
    (row,) = site.rows()
    assert row["states"] == [1, 3, 5] and row["classes"] == [None, None, "inv"]


def test_overlapping_start_taken_first_needs_no_rejection(make_site):
    seed = next(s for s in range(100) if first_draw(s) == 1)
    site = make_site(episodes=rejecting("other"))
    gamma = site.run(seed=seed)
    assert site.records()[0]["clips"] == ["7.0_a_1"]
    assert gamma.by_stem() == {"a": [[5]]}


def test_every_clip_rejected_is_no_clip(make_site):
    """`wiiwiiwiiw`: starts 0 and 3 each hold a first span line answered `other`."""
    eps = {"6xx": [Ep("data/7.0/r.mp4", "wiiwiiwiiw", classes={1: "other", 4: "other", 7: "other"})]}  # fmt: skip
    site = make_site(episodes=eps)
    gamma = site.run()
    (record,) = site.records()
    assert record["reason"] == "no_clip" and record["valid_starts"] == 2
    assert record["overlay_start"] is False and record["clips"] is None
    calls = gamma.by_stem()["r"]
    assert calls[:2] == [[2, 5], [8]] and sorted(calls[2:]) == [[4], [7]]
    assert not [p for p in site.raw("shards") if "/clips/" in p]


@pytest.mark.parametrize(
    "script, gui, classes",
    [
        (
            "wwwwc",
            [False] * 3,
            [None] * 3,
        ),  # first line k >= 1 of a container: a world state
        (
            "wwwwi",
            [False, False, True],
            [None, None, "inv"],
        ),  # first line of the inventory
        ("wwwcc", [False, False, True], [None, None, "box"]),
        ("wiiww", [False, True, False], [None, "inv", None]),
        ("cwwww", [True, False, False], ["box", None, None]),  # a container from line 0
    ],
)
def test_state_flags_and_classes_of_the_only_clip(make_site, script, gui, classes):
    """T = 5: the one start p = 0 (states 0, 2, 4); its GUI states classified once."""
    site = make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", script)]})
    gamma = site.run()
    (row,) = site.rows()
    assert (
        row["states"] == [0, 2, 4] and row["gui"] == gui and row["classes"] == classes
    )
    span = [t for t, c in enumerate(script) if c != "w"]
    assert gamma.by_stem() == {"a": [[span[-1] if len(span) > 1 else span[0]]]}
