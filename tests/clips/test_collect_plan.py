"""(1) of specs/perception-data.md: the plan, the reasons and the records of
episodes.jsonl (draw order, nulls per reason), the cut at N."""

import pytest

from .collect_world import BAD_JSONL, EPISODES, Ep, options

REASONS = (
    "version",
    "missing",
    "empty",
    "malformed",
    "options",
    "video",
    "no_clip",
    None,
)
FILLED = {  # Interface: the record fields that are not null, per reason
    "version": set(), "missing": set(), "empty": set(), "malformed": set(),
    "options": {"options"}, "video": {"options", "starts_session"},
    "no_clip": {"options", "starts_session", "overlay_start", "valid_starts"},
    None: {"options", "starts_session", "overlay_start", "valid_starts", "clips"},
}  # fmt: skip
OPTIONAL = ("options", "starts_session", "overlay_start", "valid_starts", "clips")


def test_records_follow_the_draws_with_every_reason(site):
    site.run()
    records = site.read_jsonl("episodes.jsonl")
    assert records == site.expected().records
    assert {r["reason"] for r in records} == set(REASONS)
    assert len(records) == 13  # every pool emptied: 14 relpaths, one exact repeat


def test_record_fields_are_null_unless_computed(site):
    site.run()
    for r in site.read_jsonl("episodes.jsonl"):
        assert set(r) == {"subset", "relpath", "reason", *OPTIONAL}
        assert {k for k in OPTIONAL if r[k] is not None} == FILLED[r["reason"]], r


def test_exact_repeats_are_drawn_once(site):
    """7xx lists data/7.1/b7.mp4 twice."""
    site.run()
    relpaths = [r["relpath"] for r in site.read_jsonl("episodes.jsonl")]
    assert relpaths.count("data/7.1/b7.mp4") == 1
    assert len(relpaths) == len(set(relpaths))


@pytest.mark.parametrize("where", [0, 1], ids=["inside", "at_end"])
def test_cut_at_n_commits_the_first_n_clips(site, where):
    n = site.cut() + where
    site.run(n_clips=n)
    ex = site.expected(n_clips=n)
    records, clips = site.read_jsonl("episodes.jsonl"), site.read_jsonl("clips.jsonl")
    assert records == ex.records and clips == ex.clips and len(clips) == n
    full = site.expected()
    last = len(records) - 1
    assert records[:last] == full.records[:last]  # nothing drawn after the cut
    assert records[last]["clips"] == full.records[last]["clips"][: 1 + where]
    assert set(site.tree("clips")) == set(ex.files) - set(site.tree("options"))


def test_episodes_drawn_after_the_cut_leave_nothing(site):
    """Workers ahead of the commit: the work for later episodes is discarded."""
    n = site.cut()
    site.run(n_clips=n, workers=3)
    ex = site.expected(n_clips=n)
    assert site.read_jsonl("episodes.jsonl") == ex.records
    files = site.tree()
    for name in ("run.json", "episodes.jsonl", "clips.jsonl"):
        files.pop(name)
    assert files == ex.files


def test_version_is_decided_without_a_download(site):
    site.run()
    assert not [k for k in site.asked() if "/data/6.9/" in k[1]]


@pytest.mark.parametrize(
    "ep, reason",
    [
        (Ep("data/7.0/a.mp4", options=None), "missing"),
        (Ep("data/7.0/a.mp4", jsonl=None), "missing"),
        (Ep("data/7.0/a.mp4", options=b"", jsonl=None), "missing"),
        (Ep("data/7.0/a.mp4", options=b""), "empty"),
        (Ep("data/7.0/a.mp4", jsonl=b""), "empty"),
        (Ep("data/7.0/a.mp4", options=b'{"guiScale": 2,'), "malformed"),
        (Ep("data/7.0/a.mp4", jsonl=BAD_JSONL), "malformed"),
        (Ep("data/7.0/a.mp4", options=options(autoJump=True), jsonl=b""), "empty"),
        (Ep("data/7.0/a.mp4", options=options(guiScale=3)), "options"),
        (Ep("data/7.0/a.mp4", options=options(**{"key_key.forward": "key.keyboard.z"})),
         "options"),
        (Ep("data/6.11/a.mp4", options=None), "version"),
        (Ep("data/6.12/a.mp4", "w" * 6), None),
    ],
)  # fmt: skip
def test_reasons_in_their_order(make_site, ep, reason):
    site = make_site(episodes={"9xx": [ep]})
    site.run()
    assert [r["reason"] for r in site.read_jsonl("episodes.jsonl")] == [reason]


def test_required_options_and_min_version_come_from_the_config(site):
    """min_version 8.0: every 6.x and 7.x episode is `version`; fov required: every
    episode read in full is `options`."""
    changes = {"min_version": "8.0",
               "required_options": site.fields["required_options"] | {"fov": 0.25}}  # fmt: skip
    site.run(**changes)
    records = site.read_jsonl("episodes.jsonl")
    assert records == site.expected(**changes).records
    reasons = {r["relpath"]: r["reason"] for r in records}
    for relpath, reason in reasons.items():
        old = relpath.split("/")[1].split(".")[0] in ("6", "7")
        assert (reason == "version") == old, relpath
    assert reasons["data/10.0/d10.mp4"] == reasons["data/9.2/n9.mp4"] == "options"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subsets_and_pools_are_drawn_as_in_the_transcription(make_site, seed):
    """Pools of different sizes, an empty subset; one clip per episode, N = 7."""
    eps = {s: [Ep(f"data/7.0/{s}{k}.mp4", "w" * 6) for k in range(n)]
           for s, n in zip(EPISODES, (3, 1, 0, 2, 4))}  # fmt: skip
    site = make_site(episodes=eps)
    changes = {"seed": seed, "n_clips": 7, "per_episode": 1}
    site.run(**changes)
    assert site.read_jsonl("episodes.jsonl") == site.expected(**changes).records


def test_options_reason_records_the_options_object(site):
    site.run()
    (rec,) = [r for r in site.read_jsonl("episodes.jsonl") if r["reason"] == "options"]
    assert rec["options"] == {"guiScale": 2, "autoJump": True,
                              "key_key.forward": "key.keyboard.w", "fov": 0.5}  # fmt: skip
