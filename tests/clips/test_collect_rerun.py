"""(8) of specs/perception-data.md: run.json written before (0); a rerun aborts, changing
nothing, when an output field (as JSON text), a hash or the classifier's id differs from
run.json, with the names in order; the other fields may differ and are recorded."""

import shutil

import pytest
import yaml

from .collect_out import assert_outputs
from .collect_site import HASHED


def assert_untouched_abort(site, names: list, before: dict, asked: int, decoded: int, gamma) -> None:  # fmt: skip
    """RuntimeError(f"{path} holds other values of {names}"), path = out_dir/run.json;
    nothing written, asked, decoded or classified."""
    assert type(site.error) is RuntimeError
    path, sep, tail = str(site.error).rpartition(" holds other values of ")
    assert sep and tail == str(names), str(site.error)
    assert (site.repo / path).resolve() == (site.out / "run.json").resolve()
    assert site.raw() == before and len(site.server.requests) == asked
    assert len(site.decodes()) == decoded and gamma.calls == []


def rerun_aborts(site, names: list, gid: str = "fake-gamma-1", **changes) -> None:
    before, asked, decoded = site.raw(), len(site.server.requests), len(site.decodes())
    gamma = site.aborted(gid=gid, **changes)
    assert_untouched_abort(site, names, before, asked, decoded, gamma)


def copy(site, name: str) -> str:
    """The path, relative to the repository root, of a copy of the config `name`."""
    dst = site.paths[name].with_name(f"{name}-copy.yaml")
    shutil.copyfile(site.paths[name], dst)
    return site.rel(dst)


@pytest.fixture
def ran(site):
    """A site after a run aborted in draw 2 (two shards written)."""
    site.aborted(site.failing_gamma(2))
    assert len(site.complete()) == 2
    return site


def reordered(mapping: dict) -> dict:
    return dict(reversed(mapping.items()))


@pytest.mark.parametrize(
    "name, change",
    [
        ("index_urls", lambda s: s.fields["index_urls"] | {"9xx": s.host + "/idx/9xx.json?v=2"}),
        ("index_urls", lambda s: reordered(s.fields["index_urls"])),  # same mapping
        ("min_version", lambda s: "6.13"),
        ("required_options", lambda s: s.fields["required_options"] | {"fov": 0.5}),
        ("required_options", lambda s: reordered(s.fields["required_options"])),
        ("required_options", lambda s: s.fields["required_options"] | {"guiScale": 2.0}),
        ("width", lambda s: 10), ("height", lambda s: 14),
        ("clips", lambda s: copy(s, "clips")), ("screens", lambda s: copy(s, "screens")),
        ("gui_actions", lambda s: copy(s, "gui_actions")),
        ("overlay", lambda s: copy(s, "overlay")),
        ("states", lambda s: 4), ("per_episode", lambda s: 3), ("n_clips", lambda s: 99),
        ("seed", lambda s: 1),
    ],
)  # fmt: skip
def test_output_field_changed_aborts_untouched(ran, name, change):
    """Compared as JSON text: a reordered mapping and 2.0 for 2 differ; a copied config
    file differs in its path only."""
    rerun_aborts(ran, [name], **{name: change(ran)})


@pytest.mark.parametrize("name", HASHED)
def test_hashed_file_changed_aborts_untouched(ran, name):
    with open(ran.paths[name], "a") as f:
        f.write("# changed\n")
    path = ran.paths["perception-data"]
    before, asked, decoded = ran.raw(), len(ran.server.requests), len(ran.decodes())
    gamma = ran.aborted(path=path)
    assert_untouched_abort(ran, [f"sha256.{name}"], before, asked, decoded, gamma)


def test_classifier_changed_aborts_untouched(ran):
    rerun_aborts(ran, ["classifier"], gid="fake-gamma-2")


def test_names_are_the_fields_then_the_hashes_then_the_classifier(ran):
    ran.sub["threshold"] = 0.5  # the overlay config: sha256.overlay
    rerun_aborts(ran, ["seed", "sha256.overlay", "classifier"], gid="g2", seed=3)


def test_complete_run_aborts_on_a_change(site):
    site.run(n_clips=3)
    assert site.completed()
    rerun_aborts(site, ["n_clips"], n_clips=4)


def test_early_abort_keeps_run_json_for_the_check(site):
    """run.json is written before (0): a run aborted at the first index leaves it, without
    counts, and the next run is checked against it."""
    site.indices["6xx"] = None
    site.aborted()
    assert site.run_json() == site.expected_run()
    del site.indices["6xx"]
    rerun_aborts(site, ["classifier"], gid="another")


def other_ffmpeg(site) -> str:
    dst = site.tmp / "bin" / "ffmpeg"
    dst.parent.mkdir()
    shutil.copy2(site.ffmpeg, dst)
    return str(dst)


@pytest.mark.parametrize(
    "name, change",
    [
        ("x_ms_version", lambda s: "2020-04-08"), ("fetch_streams", lambda s: 2),
        ("http_timeout", lambda s: 3), ("http_retries", lambda s: 0),
        ("retry_delay", lambda s: 0.01), ("ffmpeg", other_ffmpeg),
        ("ffmpeg_timeout", lambda s: 4.5), ("workers", lambda s: 2),
        ("classifier_batch", lambda s: 3), ("out_dir", lambda s: str(s.out)),
        ("classifier_config", lambda s: s.rel(s.tmp / "cfg" / "other-classifier.yaml")),
    ],
)  # fmt: skip
def test_other_fields_may_differ(ran, name, change):
    """The continuation ends as the uninterrupted run; run.json records the new value."""
    value = change(ran)
    ran.run(**{name: value})
    assert_outputs(ran, ran.expected())
    assert ran.run_json()["config"][name] == value


def test_config_compared_by_values_not_by_text(ran):
    """The block's config rewritten with its fields in reverse order, a comment, other
    spacing: no abort."""
    path = ran.config()
    doc = yaml.safe_load(path.read_text())
    text = yaml.safe_dump(dict(reversed(doc.items())), sort_keys=False, indent=4)
    path.write_text("# rewritten\n" + text)
    ran.collect(path)
    assert_outputs(ran, ran.expected())
