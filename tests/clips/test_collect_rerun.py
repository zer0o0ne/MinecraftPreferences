"""(6)-(8) of specs/perception-data.md: outputs independent of `workers`; run.json before
planning; the cleanup of what no record lists; the rerun abort on any change of run.json's
inputs. Interruptions: test_collect_interrupt.py; a rerun with nothing left to do:
test_collect_bars.py."""

import json

import pytest

from .collect_site import assert_outputs


def use_out(site, name: str) -> None:
    site.out = site.tmp / name
    site.fields["out_dir"] = site.rel(site.out)


@pytest.mark.parametrize("cut", [False, True])
def test_outputs_do_not_depend_on_workers(site, cut):
    n = site.cut() if cut else 100
    trees = {}
    for workers in (1, 3):
        use_out(site, f"out{workers}")
        site.run(workers=workers, n_clips=n)
        trees[workers] = site.tree()
        run = json.loads(trees[workers].pop("run.json"))
        assert run["config"].pop("workers") == workers
        assert run["config"].pop("out_dir") == site.fields["out_dir"]
        trees[workers]["run.json"] = run
    assert trees[1] == trees[3]
    assert_outputs(site, site.expected(n_clips=n))


def test_rerun_removes_what_no_record_lists(site):
    site.run()
    before = site.tree()
    rows = site.read_jsonl("clips.jsonl")
    stray = dict(rows[0], id="9.9_zz_0")
    with open(site.out / "clips.jsonl", "a") as f:
        f.write(json.dumps(stray) + "\n")
    for name in (
        "clips/9.9_zz_0/frames.yuv",
        "staging/3.mp4",
        "staging/3/x.part/frames.yuv",
    ):
        (site.out / name).parent.mkdir(parents=True, exist_ok=True)
        (site.out / name).write_bytes(b"stray")
    (site.out / "clips/9.9_zz_1").mkdir()
    site.run()
    assert site.tree() == before and not (site.out / "clips/9.9_zz_1").exists()
    assert_outputs(site, site.expected())


CHANGES = [
    ("config", {"seed": 1}), ("config", {"workers": 2}), ("config", {"n_clips": 99}),
    ("config", {"classifier_batch": 3}), ("config", {"ffmpeg_timeout": 4.0}),
    ("file", "clips"), ("file", "actions"), ("file", "screens"), ("file", "gui_actions"),
    ("file", "overlay"), ("file", "overlay_screens"), ("gamma", "fake-gamma-2"),
    ("etag", "0x8DA5AB90A1B2FFF"), ("etag", None),
]  # fmt: skip


@pytest.mark.parametrize("kind, change", CHANGES)
def test_rerun_with_any_input_changed_aborts_untouched(site, kind, change):
    site.run(n_clips=3)
    before, decoded = site.tree(), len(site.decodes())
    path = site.config(**({"n_clips": 3} | (change if kind == "config" else {})))
    if kind == "file":
        with open(site.paths[change], "a") as f:
            f.write("# changed\n")
    if kind == "etag":
        site.etags["7xx"] = change
    gamma = site.aborted(path=path, gid=change if kind == "gamma" else "fake-gamma-1")
    assert site.tree() == before
    assert gamma.calls == [] and len(site.decodes()) == decoded


def test_config_compared_as_its_mapping(site):
    """A comment in the block's own config leaves its mapping: the rerun does no work."""
    site.run(n_clips=3)
    before, path = site.tree(), site.config(n_clips=3)
    with open(path, "a") as f:
        f.write("# a comment\n")
    gamma = site.collect(path)
    assert site.tree() == before and gamma.calls == []


def test_run_json_is_written_before_planning(site):
    """An HTTP failure after the retries on the first file fetched aborts the run; run.json
    holds the config, the ETags, the hashes and gamma's id, without counts."""
    first = next(r for r in site.expected().records if r["reason"] != "version")
    path = site.path(first["subset"], first["relpath"], "-options.json")
    site.faults[site.head(path)] = [503] * 5
    site.aborted()
    run = site.run_json()
    assert set(run) == {"config", "index_etags", "sha256", "classifier"}
    assert run["classifier"] == "fake-gamma-1" and run["index_etags"] == site.etags
    assert site.asked(path) == [site.head(path)] * 2  # http_retries = 1
    site.faults.clear()
    gamma = site.aborted(gid="another gamma")
    assert gamma.calls == [] and site.run_json() == run
    site.run()
    assert_outputs(site, site.expected())
