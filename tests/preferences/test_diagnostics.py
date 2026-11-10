"""Block 33 vote/label/graph wiring, outboxes, identities and watermarks."""

import hashlib
import json
import shutil
import sys

from progress_bars import Bar, screen

from mcpref.preferences.acyclic import load_config, shard_pairs

from .acyclic_site import make_site as make_acyclic
from .acyclic_world import HAND as ACYCLIC_HAND
from .acyclic_world import entries
from .labels_world import HAND


def diagnostics(site):
    return site.repo / "w/diagnostics"


def state(site):
    return json.loads((diagnostics(site) / "run.json").read_text())


def source_digest(site, sources):
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.encode())
        digest.update(b"\0")
        digest.update((site.repo / source).read_bytes())
    return digest.hexdigest()


def patch_alias(mp, real, name, replacement):
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, name, None) is real:
            mp.setattr(module, name, replacement)


def test_shard_pairs_returns_aligned_categories_without_a_second_correction(tmp_path):
    site = make_acyclic(tmp_path)
    comparisons, _ = entries(ACYCLIC_HAND[0], 0)
    pairs, counts, categories = shard_pairs(0, comparisons, load_config(site.path))
    assert (pairs, counts) == site.expected()[0]
    assert categories == [
        "comparable",
        "comparable",
        "removed",
        "incomparable",
        "dropped",
        "comparable",
        "comparable",
        "comparable",
    ]


def test_vote_source_identity_records_and_completion(make_site):
    site = make_site(shards=HAND[:1])
    assert site.run().code == 0
    run = state(site)
    source = "w/votes/shards/000000/answers.json"
    assert run["sha256"]["inputs"]["vote"] == source_digest(site, [source])
    assert run["channels"]["vote"] == {
        "watermark": 0,
        "next": 1,
        "events": [{"id": "vote:0", "source": [source]}],
        "records": 1,
        "complete": True,
    }
    assert run["counts"] == {"vote": {"events": 1, "records": 1}}
    assert run["complete"] is True
    record = json.loads((diagnostics(site) / "targets/000000/metrics.json").read_text())
    assert record["id"] == "vote:0" and record["index"] == 0
    assert record["context"]["artifact_paths"] == [source]


def test_acyclic_channels_have_independent_identities_and_completion(tmp_path):
    site = make_acyclic(tmp_path, world=ACYCLIC_HAND[:1])
    assert site.run().code == 0
    run = state(site)
    sources = {
        "label": ["w/labels/shards/000000/pairs.json"],
        "graph": ["w/labels/shards/000000/counts.json"],
    }
    for channel, source in sources.items():
        assert run["sha256"]["inputs"][channel] == source_digest(site, source)
        assert run["channels"][channel]["next"] == 1
        assert run["channels"][channel]["complete"] is True
    assert run["complete"] is True
    assert [
        json.loads(path.read_text())["id"]
        for path in sorted(diagnostics(site).glob("targets/*/metrics.json"))
    ] == ["label:0", "graph:0"]


def test_vote_crash_before_source_recomputes_event(make_site):
    from mcpref.shards import store

    site = make_site(shards=HAND[:1])

    def before_source(mp):
        def fail(*args, **kwargs):
            raise OSError("after outbox, before source")

        patch_alias(mp, store.write_shard, "write_shard", fail)

    assert site.run(before_source).code == 1
    outboxes = list(diagnostics(site).glob("outbox/vote/*.json"))
    assert len(outboxes) == 1
    assert not (site.out / "shards/000000").exists()
    assert site.run().code == 0
    assert len(site.fake.calls) == 2
    assert not list(diagnostics(site).glob("outbox/vote/*.json"))


def test_vote_crash_after_source_recovers_outbox_without_ask(make_site):
    from mcpref.diagnostics import artifacts

    site = make_site(shards=HAND[:1])

    def after_source(mp):
        def fail(*args, **kwargs):
            raise OSError("after source, before record")

        patch_alias(mp, artifacts.write_record, "write_record", fail)

    assert site.run(after_source).code == 1
    source = site.out / "shards/000000/answers.json"
    before = source.read_bytes()
    assert len(list(diagnostics(site).glob("outbox/vote/*.json"))) == 1
    calls = len(site.fake.calls)
    assert site.run().code == 0
    assert len(site.fake.calls) == calls
    assert source.read_bytes() == before
    assert (diagnostics(site) / "targets/000000/metrics.json").is_file()
    assert not list(diagnostics(site).glob("outbox/vote/*.json"))


def test_acyclic_outbox_recovery_does_not_repeat_correction(tmp_path):
    import mcpref.preferences.acyclic as producer
    from mcpref.diagnostics import artifacts

    site = make_acyclic(tmp_path, world=ACYCLIC_HAND[:1])
    calls = []
    real_pairs = producer.shard_pairs

    def pairs(*args, **kwargs):
        calls.append(args[0])
        return real_pairs(*args, **kwargs)

    def patch(mp):
        def fail(*args, **kwargs):
            raise OSError("after source, before record")

        mp.setattr(producer, "shard_pairs", pairs)
        patch_alias(mp, artifacts.write_record, "write_record", fail)

    assert site.run(patch).code == 1
    source = site.out / "shards/000000/pairs.json"
    before = source.read_bytes()

    def recover(mp):
        mp.setattr(producer, "shard_pairs", pairs)

    assert site.run(recover).code == 0
    assert calls == [0]
    assert source.read_bytes() == before
    assert len(list(diagnostics(site).glob("targets/*/metrics.json"))) == 2
    assert not list(diagnostics(site).glob("outbox/*/*.json"))


def test_first_instrumented_run_never_backfills_and_regenerates_index(make_site):
    site = make_site(shards=HAND[:1])
    assert site.run().code == 0
    calls = len(site.fake.calls)
    shutil.rmtree(diagnostics(site))
    result = site.run()
    assert result.code == 0 and len(site.fake.calls) == calls
    run = state(site)
    assert run["channels"]["vote"]["watermark"] == 1
    assert run["channels"]["vote"]["next"] == 1
    assert run["counts"] == {"vote": {"events": 0, "records": 0}}
    assert not list(diagnostics(site).glob("targets/*/metrics.json"))
    assert "diagnostics targets: nothing to do" in screen(result.err)
    (diagnostics(site) / "index.json").unlink()
    assert site.run().code == 0
    assert (diagnostics(site) / "index.json").is_file()


def test_config_identity_mismatch_aborts_without_extra_calls(make_site):
    site = make_site(shards=HAND[:1])
    assert site.run().code == 0
    path = site.at("diagnostics")
    path.write_text(path.read_text().replace("seed: 0", "seed: 1"))
    calls = len(site.fake.calls)
    assert site.run().code == 1
    assert len(site.fake.calls) == calls


def test_target_progress_totals_and_already_done(make_site):
    site = make_site(shards=HAND[:1])
    first = site.run()
    assert Bar("diagnostics targets", 1, 1, "shard") in screen(first.err)
    second = site.run()
    assert "diagnostics targets: already done" in screen(second.err)
    assert len(site.fake.calls) == 1
