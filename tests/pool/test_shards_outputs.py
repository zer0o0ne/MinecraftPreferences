"""Outputs of `collect` of specs/preference-pool.md ((0), (H), (P), (W), (8'); Datasets)
against the transcription of pool_expected.py on the default world: the head shards, the
shards (records.json, state.json, options/, clips/, comparisons.json), the indices,
run.json with its counts, nothing else."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from . import graph_oracle
from .pool_site import assert_outputs
from .pool_world import chain, small_episodes


def test_outputs_of_a_run_through_every_session(site):
    """Every pool emptied: eight sessions, eight shards, every reason of (3)."""
    site.run()
    ex = site.expected()
    assert_outputs(site, ex)
    assert len(ex.shards) == 8 and len(ex.sessions) == 7
    reasons = {r["reason"] for s in ex.shards for r in s.records}
    assert reasons == {None, "missing", "malformed", "options", "video", "no_clip"}


def test_json_files_are_indented_dumps_ending_with_a_newline(site):
    site.run()
    for path, data in site.raw().items():
        if path.endswith(".json") and "/options/" not in path and not path.startswith("indices/"):  # fmt: skip
            text = data.decode("ascii")
            assert text == json.dumps(json.loads(text), indent=2) + "\n", path


RECORD = ("subset", "relpath", "reason", "options", "starts_session", "overlay_start",
          "valid_starts", "clips")  # fmt: skip


def test_json_keys_in_the_order_of_datasets(site):
    site.run()
    for path, value in site.tree().items():
        name = path.rsplit("/", 1)[-1]
        if name == "records.json":
            assert all(tuple(r) == RECORD for r in value), path
        elif name == "state.json":
            assert list(value) == ["generator", "clips", "session"], path
            assert list(value["session"]) == ["subset", "start"], path
        elif name == "comparisons.json":
            assert all(list(x) == ["a", "b"] and list(x["a"]) == list(x["b"]) == ["id", "episode"] for x in value)  # fmt: skip
        elif name == "sessions.json":
            assert list(value) == ["sessions", "entries"]
            assert all(list(x) == ["subset", "pieces"] for x in value["sessions"])


def test_comparisons_refer_to_the_clips_of_their_shard(site):
    """Each clip of a shard is compared at least once ((G) connects them); `episode` is
    the clip's relpath; no clip compared with itself."""
    site.run()
    tree = site.tree("shards")
    for name in site.complete():
        clips = {p.split("/")[3]: x for p, x in tree.items()
                 if p.startswith(f"shards/{name}/clips/") and p.endswith("clip.json")}  # fmt: skip
        pairs = tree[f"shards/{name}/comparisons.json"]
        seen = {x[s]["id"] for x in pairs for s in "ab"}
        assert seen == set(clips)
        assert all(x[s]["episode"] == clips[x[s]["id"]]["relpath"] for x in pairs for s in "ab")  # fmt: skip
        assert all(x["a"]["id"] != x["b"]["id"] for x in pairs)


def shard_of(site, name: str) -> tuple[str, dict]:
    """The shard of the session whose start's name begins with `name`, and its files."""
    tree = site.tree("shards")
    for k in site.complete():
        state = tree[f"shards/{k}/state.json"]
        if state["session"]["start"].split("/")[2].startswith(name):
            return k, {p.split("/", 2)[2]: x for p, x in tree.items() if p.startswith(f"shards/{k}/")}  # fmt: skip
    raise AssertionError(name)


@pytest.mark.parametrize("name, reasons", [("beta", [None]), ("eta", [None, "no_clip"]), ("iota", [None, "malformed"]), ("zeta", ["missing", "video"])])  # fmt: skip
def test_session_with_fewer_than_two_clips(site, name, reasons):
    """Its records (a valid episode with `clips` []), no clip, no comparison; c unchanged."""
    site.run()
    k, files = shard_of(site, name)
    assert [r["reason"] for r in files["records.json"]] == reasons
    assert all(r["clips"] == [] for r in files["records.json"] if r["reason"] is None)
    assert files["comparisons.json"] == [] and not [p for p in files if p.startswith("clips/")]  # fmt: skip
    before = site.tree("shards")[f"shards/{int(k) - 1:06d}/state.json"]["clips"] if int(k) else 0  # fmt: skip
    assert files["state.json"]["clips"] == before


def test_valid_episodes_list_their_clips_in_the_order_taken(site):
    """kap-pa: two pieces of two clips each; four clips, comparisons over them."""
    site.run()
    _, files = shard_of(site, "kap-pa")
    ids = [r["clips"] for r in files["records.json"]]
    assert [len(x) for x in ids] == [2, 2]
    assert sorted(p.split("/")[1] for p in files if p.endswith("clip.json")) == sorted(ids[0] + ids[1])  # fmt: skip


@pytest.mark.parametrize("n", [2, 4, 5, 9])
def test_stop_after_the_shard_reaching_n_clips(site, n):
    """The shard bringing c to n or above is the last, its session written whole."""
    full = site.expected()
    site.run(n_clips=n)
    ex = site.expected(n_clips=n)
    assert_outputs(site, ex)
    cs = [s.clips for s in ex.shards]
    assert cs[-1] >= n and (len(cs) == 1 or cs[-2] < n)
    assert [s.files for s in ex.shards] == [s.files for s in full.shards[: len(cs)]]
    assert site.run_json()["clips"] == cs[-1]


def test_run_json_counts(site):
    """episodes: the reasons of every record; sessions: the shards; clips: c; comparisons:
    over all shards."""
    site.run(gid="gamma v2")
    ex, run = site.expected(), site.run_json()
    records = [r for s in ex.shards for r in s.records]
    assert run["classifier"] == "gamma v2" and run["sessions"] == 8
    assert run["episodes"] == {"null": 10, "missing": 1, "video": 1, "no_clip": 1, "options": 1, "malformed": 1}  # fmt: skip
    assert sum(run["episodes"].values()) == len(records) == 15
    assert run["clips"] == ex.shards[-1].clips == len(site.rows())
    assert run["comparisons"] == sum(len(s.comparisons) for s in ex.shards) > 0
    assert run["config"]["out_dir"] == "out" and run["config"]["perception"] == "cfg/perception-data.yaml"  # fmt: skip


def use_out(site, name: str) -> None:
    site.out = site.repo / name


@pytest.mark.parametrize("n", [100, 5])
def test_outputs_do_not_depend_on_workers(make_site, n):
    """The small world, byte for byte: run.json but its `workers` and `out_dir`."""
    site, trees = make_site(episodes=small_episodes()), {}
    for workers in (1, 3):
        use_out(site, f"out{workers}")
        site.run(workers=workers, n_clips=n)
        assert_outputs(site, site.expected(workers=workers, n_clips=n))
        trees[workers] = site.raw()
        run = json.loads(trees[workers].pop("run.json"))
        assert run["config"].pop("workers") == workers
        assert run["config"].pop("out_dir") == f"out{workers}"
        trees[workers]["run.json"] = run
    assert trees[1] == trees[3]


def ten_clips() -> dict:
    """One session of five `w`*16 pieces: ten clips."""
    return {"8xx": chain("8.1", "ten", "ab" * 6, ["w" * 16] * 5)}


def cycles_before_top_up(seed: int) -> int:
    """|E| - n + 1 after steps 1-2 of (G) on draw 0's ten clips, partners 1."""
    g = np.random.Generator(np.random.PCG64([seed, 5, 0]))
    edges = graph_oracle.partners(10, 1, g)
    graph_oracle.connect(10, edges, g)
    return len(edges) - 9


@pytest.mark.parametrize(
    "rho, written, cycles",
    [(0.1, "0.1", 1), (0.4, "0.4", 4), (0.1, "0.10000000000000001", 1), (0.4, "0.40000000000000001", 4)],
)  # fmt: skip
def test_cycles_taken_exactly_from_the_config(make_site, rho, written, cycles):
    """ceil(rho n) with n = 10 and the exact value of Notation (the shortest repr of the
    float read): 1 for 0.1, 4 for 0.4, also when written with 17 digits (from the floats'
    own values, 0.1000...0555 and 0.4000...0222, or the 17-digit decimals as written, it
    gives 2 and 5); a seed whose steps 1-2 leave no cycle."""
    assert math.ceil(Fraction(rho) * 10) == cycles + 1
    assert math.ceil(Fraction(written) * 10) == cycles + (written != str(rho))
    seed = next(s for s in range(100) if cycles_before_top_up(s) == 0)
    site = make_site(episodes=ten_clips())
    path = site.config(episodes=5, cycles=rho, seed=seed)
    text = path.read_text().replace(f"cycles: {rho}\n", f"cycles: {written}\n")
    assert f"cycles: {written}\n" in text
    path.write_text(text)
    site.collect(path)
    ex = site.expected(episodes=5, cycles=rho, seed=seed)
    assert_outputs(site, ex)
    assert ex.shards[0].clips == 10 and len(ex.shards[0].comparisons) == 9 + cycles


def test_paths_resolve_against_the_repository_root(site):
    """The working directory, where every relative config path leads nowhere, stays
    empty."""
    site.run(n_clips=2)
    assert site.completed() and list(Path.cwd().iterdir()) == []
