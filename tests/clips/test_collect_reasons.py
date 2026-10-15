"""The checks of (1) of specs/perception-data.md: the reasons in their order, decided by
the config's `min_version` (integer tuples) and `required_options` (Python ==), with the
`.jsonl` not asked when the options file is missing."""

import pytest

from .collect_out import assert_outputs
from .collect_world import BAD_JSONL, Ep, options


def test_version_is_decided_without_a_download(site):
    site.run()
    assert not [k for k in site.asked() if "/data/6.9/" in k[1]]


def test_missing_options_file_leaves_the_jsonl_unasked(make_site):
    site = make_site(episodes={"9xx": [Ep("data/7.0/a.mp4", options=None)]})
    site.run()
    assert site.records()[0]["reason"] == "missing"
    assert site.asked(site.path("data/7.0/a.mp4", ".jsonl")) == []
    assert site.asked(site.path("data/7.0/a.mp4", "-options.json"))


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
        (Ep("data/7.0/a.mp4", options=b"[1, 2]"), "malformed"),  # not a JSON object
        (Ep("data/7.0/a.mp4", options=b'"guiScale"'), "malformed"),
        (Ep("data/7.0/a.mp4", options=b"null"), "malformed"),
        (Ep("data/7.0/a.mp4", options=b"[1, 2]", jsonl=BAD_JSONL), "malformed"),
        (Ep("data/7.0/a.mp4", options=options(autoJump=True), jsonl=b""), "empty"),
        (Ep("data/7.0/a.mp4", options=options(guiScale=3)), "options"),
        (Ep("data/7.0/a.mp4", options=options(guiScale="2")), "options"),
        (Ep("data/7.0/a.mp4", options=b'{"guiScale": 2, "autoJump": false}'), "options"),
        (Ep("data/7.0/a.mp4", options=options(**{"key_key.forward": "key.keyboard.z"})),
         "options"),
        (Ep("data/7.0/a.mp4", "w" * 6, options=options(guiScale=2.0)), None),  # 2 == 2.0
        (Ep("data/7.0/a.mp4", "w" * 6, options=options(autoJump=0)), None),  # 0 == False
        (Ep("data/6.11/a.mp4", options=None), "version"),
        (Ep("data/6.12/a.mp4", "w" * 6), None),
    ],
)  # fmt: skip
def test_reasons_in_their_order(make_site, ep, reason):
    site = make_site(episodes={"9xx": [ep]})
    site.run()
    assert [r["reason"] for r in site.records()] == [reason]


def test_malformed_keeps_the_options_file_and_no_record_options(make_site):
    site = make_site(episodes={"9xx": [Ep("data/7.0/a.mp4", options=b"[1, 2]")]})
    site.run()
    assert site.records()[0]["options"] is None
    assert site.raw("shards")["shards/000000/options/7.0_a.json"] == b"[1, 2]"


def test_required_options_and_min_version_come_from_the_config(site):
    """min_version 8.0: every 6.x and 7.x episode is `version`; fov required: every
    episode read in full is `options`."""
    changes = {"min_version": "8.0",
               "required_options": site.fields["required_options"] | {"fov": 0.25}}  # fmt: skip
    site.run(**changes)
    assert_outputs(site, site.expected(**changes))
    reasons = {r["relpath"]: r["reason"] for r in site.records()}
    for relpath, reason in reasons.items():
        old = relpath.split("/")[1].split(".")[0] in ("6", "7")
        assert (reason == "version") == old, relpath
    assert reasons["data/10.0/d10.mp4"] == reasons["data/9.2/n9.mp4"] == "options"


def test_versions_compare_as_integer_tuples(make_site):
    """min_version 6.12: 6.9 is below it, 6.100 and 10.0 are not (not string order)."""
    eps = [Ep("data/6.9/a.mp4", "w" * 6), Ep("data/6.100/b.mp4", "w" * 6), Ep("data/10.0/c.mp4", "w" * 6)]  # fmt: skip
    site = make_site(episodes={"6xx": eps})
    site.run()
    reasons = {r["relpath"]: r["reason"] for r in site.records()}
    assert reasons == {"data/6.9/a.mp4": "version", "data/6.100/b.mp4": None, "data/10.0/c.mp4": None}  # fmt: skip
