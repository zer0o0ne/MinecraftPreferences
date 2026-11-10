"""Direct indexed History lookup and base-only comparison diagnostics."""

import json

import mcpref.rl_pairs.history as history_module
from mcpref.rl_pairs.history import History
from mcpref.rl_pairs.types import CurrentComparison, StateRef

from .helpers import comparison, config, vertex, write_history


def naive_raw(history_dir, ref):
    for path in sorted((history_dir / "000000/shards").glob("*/vertices.json")):
        for record in json.loads(path.read_text()):
            if (record["episode"], record["index"]) == (ref.episode, ref.index):
                return record
    return None


def raw_json(value):
    if value is None:
        return None
    return {
        "episode": value.ref.episode,
        "index": value.ref.index,
        "key": list(value.key),
        "unused": value.unused,
        "comparisons": [
            {
                "episode": c.ref.episode,
                "index": c.ref.index,
                "p": c.p,
                "inc": c.inc,
                "n": c.n,
            }
            for c in value.comparisons
        ],
    }


def test_raw_lookup_builds_one_index_and_reads_at_most_one_shard_per_ref(
    tmp_path, monkeypatch
):
    refs = [StateRef("old", i) for i in range(6)]
    cfg = config(tmp_path, history_per_shard=2)
    write_history(cfg, 0, [vertex(ref, [float(i)]) for i, ref in enumerate(refs)])
    history = History.load(tmp_path / "history", 2)
    reads = []
    real_read = history_module.read_json_shard

    def counted_read(directory, shard, filename):
        reads.append(shard)
        return real_read(directory, shard, filename)

    monkeypatch.setattr(history_module, "read_json_shard", counted_read)

    found = history.get(refs[5])
    assert raw_json(found) == naive_raw(tmp_path / "history", refs[5])
    assert reads == [0, 1, 2]

    reads.clear()
    assert history.get(refs[5]) == found
    assert history.get(refs[0]).key == (0.0,)
    assert history.get(StateRef("missing", 0)) is None
    assert history.get(StateRef("missing", 1)) is None
    assert reads == [0]
    for ref in [*refs, StateRef("missing", 2)]:
        assert raw_json(history.get(ref)) == naive_raw(tmp_path / "history", ref)


def test_older_comparison_reads_only_base_despite_staging_and_removal(tmp_path):
    a, b, c, d = [StateRef("old", i) for i in range(4)]
    cfg = config(tmp_path, history_per_shard=2)
    write_history(
        cfg,
        0,
        [
            vertex(a, [0.0], comparisons=[comparison(a, b, 0.75, 0.1, 2)]),
            vertex(b, [1.0], comparisons=[comparison(b, a, 0.25, 0.1, 2)]),
            vertex(c, [2.0]),
            vertex(d, [3.0]),
        ],
    )
    history = History.load(tmp_path / "history", 2)
    history.stage_comparison(CurrentComparison(c, d, 0.6, 0.2))

    assert not history.older_comparison(c, d)
    assert history.older_comparison(a, b)
    history._removed.add(a)
    assert history.get(a) is None
    assert history.older_comparison(a, b)
