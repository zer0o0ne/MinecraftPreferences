"""Unique clip objectives are independent of the number of labeled partners."""

from pathlib import Path

import pytest

from training.pretrain.data.index_run import aborted, items
from training.pretrain.data.index_world import edit, read, setting

PAIR_B = "w/pair_clips/shards/000000/clips/b_0"


def clip_paths(site) -> list[str]:
    return [
        row["clip"]
        for shard in items(site.out, "train")
        for row in shard
        if "clip" in row
    ]


def test_pair_endpoints_have_one_standalone_clip_item(make_site):
    site = make_site(validation="0", items_per_shard=100)
    result = site.run()
    assert result.code == 0, result.err
    [rows] = items(site.out, "train")
    clips = [row["clip"] for row in rows if "clip" in row]
    # d_3 belongs to two comparisons, but owns one clip objective.
    assert clips.count("w/pair_clips/shards/000000/clips/d_3") == 1
    assert len(clips) == 7
    assert sum("a" in row for row in rows) == 3
    assert "w/clips/shards/000000/clips/b_0" in clips
    assert PAIR_B not in clips


def test_full_history_defines_identity_even_with_same_id_and_start(make_site):
    site = make_site(validation="0")
    edit(site.repo / PAIR_B / "clip.json", setting("states", list(range(0, 32, 2))))
    result = site.run()
    assert result.code == 0, result.err
    assert len(clip_paths(site)) == 8
    assert PAIR_B in clip_paths(site)


def test_storage_id_is_not_clip_identity(make_site):
    site = make_site(validation="0")
    edit(site.repo / PAIR_B / "clip.json", setting("id", "different-storage-alias"))
    result = site.run()
    assert result.code == 0, result.err
    assert len(clip_paths(site)) == 7
    assert PAIR_B not in clip_paths(site)


def test_conflicting_targets_for_same_identity_abort(make_site):
    site = make_site()
    edit(site.repo / PAIR_B / "clip.json", setting("pitch", [1] * 16))
    aborted(
        site.run(),
        f"{PAIR_B} conflicts with clip metadata of w/clips/shards/000000/clips/b_0",
    )
    assert not site.out.exists()


def test_pair_episode_must_match_its_metadata(make_site):
    site = make_site()
    edit(site.repo / PAIR_B / "clip.json", setting("relpath", "another-episode"))
    aborted(site.run(), f"{PAIR_B} metadata disagrees with episode 6.14/b.mp4")


def test_metadata_hash_changes_without_changing_source_fingerprints(make_site):
    site = make_site()
    assert site.run().code == 0
    before = site.tree()
    sources = site.fingerprints()
    path = site.repo / PAIR_B / "clip.json"
    path.write_bytes(path.read_bytes() + b" ")
    aborted(site.run(), "clip_metadata changed since the run started; delete w/index")
    assert site.fingerprints() == sources
    assert site.tree() == before


def test_only_used_metadata_read_once_without_clip_payload(make_site):
    site = make_site()
    unused = site.repo / "w/pair_clips/shards/000001/clips/f_8/clip.json"
    unused.write_text("not JSON")
    reads = []
    original = Path.read_bytes

    def read_bytes(path):
        assert path.name not in ("frames.yuv", "lines.jsonl")
        if path.name == "clip.json":
            reads.append(path)
        return original(path)

    result = site.run(lambda mp: mp.setattr(Path, "read_bytes", read_bytes))
    assert result.code == 0, result.err
    # Four U files plus four used Q_c files; d_3 and b_0 each serve two pairs.
    assert len(reads) == len(set(reads)) == 8
    assert unused not in reads


@pytest.mark.parametrize(
    "change",
    [
        lambda run: run.pop("semantics"),
        setting("semantics", "occurrence-weighted"),
        lambda run: run["sha256"].pop("clip_metadata"),
    ],
)
def test_legacy_index_is_rejected_before_reusing_its_shards(make_site, change):
    site = make_site()
    assert site.run().code == 0
    edit(site.out / "run.json", change)
    before = site.tree()
    assert aborted(site.run(), "w/index must be rebuilt for unique-clips-v1") == []
    assert site.tree() == before


def test_shards_without_a_compatible_run_header_are_not_trusted(make_site):
    site = make_site()
    assert site.run().code == 0
    (site.out / "run.json").unlink()
    before = site.tree()
    aborted(site.run(), "w/index has index shards without run.json; rebuild it")
    assert site.tree() == before


def test_unreferenced_pair_pool_clip_is_not_added(make_site):
    site = make_site(validation="0")
    assert site.run().code == 0
    assert all("f_8" not in path for path in clip_paths(site))
    run = read(site.out / "run.json")
    assert run["semantics"] == "unique-clips-v1"
    assert len(run["sha256"]["clip_metadata"]) == 64
