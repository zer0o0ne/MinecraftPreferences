"""(8') of specs/preference-pool.md: run.json written after the check and before (0); a
rerun aborts, changing nothing, when an output field of Notation (as JSON text; of this
config or of perception-data's), a hash or the classifier's id differs, with the names
`config.<name>`, `sha256.<key>`, `classifier` in order; the other fields may differ."""

import shutil

import pytest
import yaml
from clips.collect_site import HASHED

from .pool_site import assert_outputs


def assert_untouched_abort(site, names: list, before: dict, asked: int, decoded: int, gamma) -> None:  # fmt: skip
    """RuntimeError(f"{path} holds other values of {names}"), path = out_dir/run.json;
    nothing written or removed, asked, decoded or classified."""
    assert type(site.error) is RuntimeError
    path, sep, tail = str(site.error).rpartition(" holds other values of ")
    assert sep and tail == str(names), str(site.error)
    assert (site.repo / path).resolve() == (site.out / "run.json").resolve()
    assert site.raw() == before and len(site.server.requests) == asked
    assert len(site.decodes()) == decoded and gamma.calls == []


def rerun_aborts(
    site, names: list, gid: str = "fake-gamma-1", p=None, **changes
) -> None:
    before, asked, decoded = site.raw(), len(site.server.requests), len(site.decodes())
    gamma = site.aborted(gid=gid, p=p, **changes)
    assert_untouched_abort(site, names, before, asked, decoded, gamma)


def copy(site, name: str) -> str:
    """The path, relative to the repository root, of a copy of the config `name`."""
    dst = site.paths[name].with_name(f"{name}-copy.yaml")
    shutil.copyfile(site.paths[name], dst)
    return site.rel(dst)


@pytest.fixture
def ran(site):
    """A site after a run aborted in a draw (shards written), staging left with a file."""
    ex = site.expected()
    k = next(j for j, s in enumerate(ex.shards) if j >= 2 and s.records[0]["reason"] is None)  # fmt: skip
    path = site.path(ex.shards[k].records[0]["relpath"], ".mp4")
    site.faults[site.head(path)] = [503] * 2
    site.aborted()
    site.faults.clear()
    assert len(site.complete()) == k
    (site.out / "staging" / "x").mkdir(parents=True, exist_ok=True)
    (site.out / "staging" / "x" / "left.mp4").write_bytes(b"x")
    return site


def reordered(mapping: dict) -> dict:
    return dict(reversed(mapping.items()))


THEIRS = [
    ("index_urls", lambda s: s.fields["index_urls"] | {"9xx": s.host + "/idx/9xx.json?v=2"}),
    ("index_urls", lambda s: reordered(s.fields["index_urls"])),
    ("min_version", lambda s: "6.14"),
    ("required_options", lambda s: s.fields["required_options"] | {"fov": 0.5}),
    ("required_options", lambda s: reordered(s.fields["required_options"])),
    ("required_options", lambda s: s.fields["required_options"] | {"guiScale": 2.0}),
    ("width", lambda s: 10), ("height", lambda s: 14),
    ("clips", lambda s: copy(s, "clips")), ("screens", lambda s: copy(s, "screens")),
    ("gui_actions", lambda s: copy(s, "gui_actions")),
    ("overlay", lambda s: copy(s, "overlay")), ("states", lambda s: 4),
]  # fmt: skip


@pytest.mark.parametrize("name, change", THEIRS)
def test_output_field_of_perception_changed_aborts_untouched(ran, name, change):
    """Compared as JSON text: a reordered mapping and 2.0 for 2 differ."""
    rerun_aborts(ran, [f"config.{name}"], p={name: change(ran)})


OURS = [
    ("entries_per_shard", 4), ("tick_ms", 51), ("tick_slack", 0.04),
    ("resetting_stats", ["minecraft.custom:minecraft.time_since_death",
                         "minecraft.custom:minecraft.time_since_rest"]),
    ("episodes", 3), ("per_episode", 3), ("n_clips", 99), ("partners", 2),
    ("cycles", 0.25), ("seed", 1),
]  # fmt: skip


@pytest.mark.parametrize("name, value", OURS)
def test_output_field_of_the_pool_changed_aborts_untouched(ran, name, value):
    rerun_aborts(ran, [f"config.{name}"], **{name: value})


def test_numbers_compared_as_json_text(site):
    """cycles 1.0, then 1."""
    site.run(cycles=1.0, n_clips=2)
    rerun_aborts(site, ["config.cycles"], cycles=1, n_clips=2)


@pytest.mark.parametrize("name", HASHED)
def test_hashed_file_changed_aborts_untouched(ran, name):
    with open(ran.paths[name], "a") as f:
        f.write("# changed\n")
    before, asked, decoded = ran.raw(), len(ran.server.requests), len(ran.decodes())
    gamma = ran.aborted(path=ran.pool_path)
    assert_untouched_abort(ran, [f"sha256.{name}"], before, asked, decoded, gamma)


def test_classifier_changed_aborts_untouched(ran):
    rerun_aborts(ran, ["classifier"], gid="fake-gamma-2")


def test_names_in_order(ran):
    """Perception's fields, then the pool's, in the order of Notation; then the hashes;
    then the classifier."""
    ran.sub["threshold"] = 0.5  # the overlay config: sha256.overlay
    names = ["config.min_version", "config.states", "config.tick_ms", "config.seed",
             "sha256.overlay", "classifier"]  # fmt: skip
    rerun_aborts(ran, names, gid="g2", p={"states": 4, "min_version": "6.14"}, seed=3, tick_ms=7)  # fmt: skip


def test_complete_run_aborts_on_a_change(site):
    site.run(n_clips=3)
    assert site.completed()
    rerun_aborts(site, ["config.n_clips"], n_clips=4)


def test_early_abort_keeps_run_json_for_the_check(site):
    """run.json is written before (0): a run aborted at the first index leaves it, without
    counts, and the next run is checked against it."""
    site.indices["6xx"] = None
    site.aborted()
    assert site.run_json() == site.expected_run()
    del site.indices["6xx"]
    rerun_aborts(site, ["classifier"], gid="another")


def other_perception(site) -> str:
    """A copy of perception-data's config elsewhere."""
    return copy(site, "perception-data")


THEIRS_FREE = [("per_episode", 7), ("n_clips", 3), ("seed", 9), ("workers", 3),
               ("out_dir", "elsewhere"), ("x_ms_version", "2020-04-08"),
               ("fetch_streams", 2), ("http_retries", 0), ("classifier_batch", 3)]  # fmt: skip


@pytest.mark.parametrize("name, value", THEIRS_FREE)
def test_other_fields_of_perception_may_differ(ran, name, value):
    """The continuation ends as the uninterrupted run (staging removed)."""
    ran.run(p={name: value})
    assert_outputs(ran, ran.expected())


@pytest.mark.parametrize("name, value", [("head_bytes", 64), ("head_workers", 1), ("workers", 3), ("perception", other_perception)])  # fmt: skip
def test_other_fields_of_the_pool_may_differ(ran, name, value):
    """run.json records the new value."""
    value = value(ran) if callable(value) else value
    ran.run(**{name: value})
    assert_outputs(ran, ran.expected())
    assert ran.run_json()["config"][name] == value


def test_config_compared_by_values_not_by_text(ran):
    """The pool config rewritten in reverse key order, with a comment and other spacing:
    no abort; run.json keeps the keys in the order of PreferencePoolConfig."""
    path = ran.config()
    doc = yaml.safe_load(path.read_text())
    text = yaml.safe_dump(dict(reversed(doc.items())), sort_keys=False, indent=4)
    path.write_text("# rewritten\n" + text)
    ran.collect(path)
    assert_outputs(ran, ran.expected())
