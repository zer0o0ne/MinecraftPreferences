"""(W), (3'), (4) of specs/acyclic.md through `main`: the shards of `{out_dir}` and its
`run.json` on a hand world (derived by hand below), on random worlds against the
transcription on the whole G, and on the starting weights (float64 semantics of (2) on the
exact costs)."""

import json
from fractions import Fraction

import pytest
from graphs.feedback_oracle import igraph_eades
from pretrain.index_run import interrupting

from .acyclic_oracle import end
from .acyclic_site import make_site
from .acyclic_world import (
    INC,
    MB,
    SB,
    START,
    SW,
    B,
    W,
    comparison,
    json_bytes,
    random_world,
    ref,
)


def pair(k: int, u: str, v: str, p) -> dict:
    """An entry of pairs.json: labels (4), p None for {"p": null, "inc": 1}."""
    labels = {"p": None, "inc": 1} if p is None else {"p": p, "inc": 0}
    return {"a": end(k, ref(u)), "b": end(k, ref(v)), "labels": labels}


def counts(comparable=0, removed=0, incomparable=0, dropped=0) -> dict:
    return {"comparable": comparable, "removed": removed,
            "incomparable": incomparable, "dropped": dropped}  # fmt: skip


# HAND, weights DYADIC, s = 1, tau = 1/2 (acyclic_world.py). Shard 0: arcs p -> q, q -> r,
# r -> p (costs 1/4), r -> t (1/2), q -> t (1/8), t -> s (0); vertices p, q, r, t, s =
# 0..4 (first appearance, u first); s, then t are sinks; the 3-cycle has delta 0 everywhere:
# v* = p, its in-arc r -> p removed, kept out (p -> q -> r).
# Shard 3: vertices v0, v3, v1, v2 = 0..3; delta = 1/4, 3/8, -3/8, -1/4: v* = v3, arc
# v0 -> v3; then delta(v0) = 1/4 is the largest: arc v1 -> v0; put back by cost: v1 -> v0
# stays (v0 -> v2 -> v1), v0 -> v3 back (v3 -> v1 -> v0 is cut).
HAND_SHARDS = [
    ([pair(0, "p_1", "q_1", 0.75), pair(0, "q_1", "r_1", 0.75), pair(0, "r_1", "p_1", None),
      pair(0, "p_1", "t_1", None), pair(0, "r_1", "t_1", 1.0), pair(0, "q_1", "t_1", 0.625),
      pair(0, "t_1", "s_1", 0.5)],
     counts(comparable=5, removed=1, incomparable=1, dropped=1)),
    ([], counts()),
    ([], counts(dropped=1)),
    ([pair(3, "v0_3", "v3_3", 0.5), pair(3, "v1_3", "v0_3", None),
      pair(3, "v3_3", "v1_3", 0.875), pair(3, "v2_3", "v1_3", 0.625),
      pair(3, "v0_3", "v2_3", 0.875)],
     counts(comparable=4, removed=1)),
]  # fmt: skip
HAND_COUNTS = {
    "pairs": 12,
    **counts(comparable=9, removed=2, incomparable=1, dropped=2),
}


def test_hand_world(tmp_path):
    """The shards by hand: pairs (u, v) of comparable verdicts, (a, b) otherwise, in order,
    `dropped` omitted, removed arcs `{"p": null, "inc": 1}`, a shard without comparisons
    and one with only dropped comparisons (`pairs.json` = []); counts.json; run.json."""
    site = make_site(tmp_path)
    r = site.run()
    assert (r.code, r.stdout) == (0, ""), r.err
    tree = site.tree()
    for k, (pairs, c) in enumerate(HAND_SHARDS):
        assert json.loads(tree.pop(f"shards/{k:06d}/pairs.json")) == pairs, k
        assert tree.pop(f"shards/{k:06d}/counts.json") == json_bytes(c), k
    run = json.loads(tree.pop("run.json"))
    assert (run["complete"], run["counts"]) == (True, HAND_COUNTS)
    assert tree == {}


def test_hand_world_bytes(tmp_path):
    """Every file of {out_dir} byte for byte as (W) writes it (the transcription agrees with
    the hand derivation)."""
    site = make_site(tmp_path)
    assert site.expected() == HAND_SHARDS
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()


# seeds whose worlds hold removed arcs, arcs put back, incomparable and dropped verdicts
@pytest.mark.parametrize("seed", [0, 23, 54, 57])
def test_random_worlds_on_the_whole_graph(tmp_path, seed):
    """(3'): the shards computed one at a time equal (1)-(4) on the whole G (vertices and
    arcs numbered in shard order), byte for byte; s = 2."""
    site = make_site(tmp_path, random_world(seed, n_shards=5))
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()


def test_starting_weights_follow_float64_semantics(tmp_path):
    """Weights of configs/acyclic.yaml: arcs x -> y, z -> x, y -> z with w = 0.8, 0.7, 0.6
    (vertices x, y, z = 0..2); exact costs 3/10, 1/5, 1/10 as float64: delta(x) =
    0.3 - 0.2 = 0.09999999999999998 < delta(z) = 0.2 - 0.1 = 0.1, so v* = z and y -> z is
    removed (an exact tie would remove z -> x; costs float(w) - 0.5 would too)."""
    world = [[
        comparison("x_1", "y_1", B, W),  # (0.8 + (1 - 0.2)) / 2
        comparison("z_1", "x_1", B, SW),  # (0.8 + 0.6) / 2
        comparison("y_1", "z_1", SB, SW),  # (0.6 + 0.6) / 2
    ]]  # fmt: skip
    site = make_site(tmp_path, world, weights=START)
    assert site.run().code == 0
    pairs = json.loads(site.tree()["shards/000000/pairs.json"])
    assert [p["labels"] for p in pairs] == [
        {"p": 0.8, "inc": 0}, {"p": 0.7, "inc": 0}, {"p": None, "inc": 1}
    ]  # fmt: skip
    assert site.tree() == site.expected_tree(igraph_eades)


@pytest.mark.parametrize("seed", [4, 13, 64])  # as above
def test_random_worlds_starting_weights(tmp_path, seed):
    """(3') with the starting weights: the shards equal F of igraph's Eades on the whole G,
    put back."""
    site = make_site(tmp_path, random_world(seed, n_shards=5, s=1), weights=START)
    assert site.run().code == 0
    assert site.tree() == site.expected_tree(igraph_eades)


def test_stored_p_is_the_rounded_exact_mean(tmp_path):
    """(4): p = float(w), w exact: (1 + 1 + (1 - 0.8)) / 3 = 11/15 -> 0.7333333333333333
    (float64 arithmetic: 0.7333333333333334)."""
    world = [[comparison("x_1", "y_1", MB, MB, B, "invalid")]]
    site = make_site(tmp_path, world, weights=START)
    assert site.run().code == 0
    pairs = json.loads(site.tree()["shards/000000/pairs.json"])
    assert pairs[0]["labels"]["p"] == float(Fraction(11, 15)) == 0.7333333333333333


def test_run_json_after_inputs(tmp_path):
    """(I): {config as written, sha256: {config, votes}, pair_clips} once the inputs are
    read, before any shard; votes over the complete shards of V_l only."""
    site = make_site(tmp_path)
    r = site.run(interrupting("correction shards", 1))
    assert r.code == 1
    assert (site.out / "run.json").read_bytes() == site.expected_run()


def test_run_json_keeps_the_config_as_written(tmp_path):
    """`config` of run.json is the file's mapping as written (keys in the file's order,
    numbers as written), its sha256 that of the file's bytes (a comment included)."""
    site = make_site(tmp_path)
    site.path.write_text(
        "# a comment\nrefusal_share: 0.50\nout_dir: w/labels\nvotes_dir: w/votes\n"
        "weights: {MUCH_BETTER: 1, BETTER: 0.75, SLIGHTLY_BETTER: 0.625, SLIGHTLY_WORSE:"
        " 0.375, WORSE: 0.25, MUCH_WORSE: 0, INCOMPARABLE: null}\n"
    )
    assert site.run().code == 0
    run = json.loads((site.out / "run.json").read_bytes())
    assert list(run["config"]) == ["refusal_share", "out_dir", "votes_dir", "weights"]
    assert site.tree() == site.expected_tree()


def test_no_shards(tmp_path):
    """V_l without shards: run.json complete with zero counts, no shard."""
    site = make_site(tmp_path, [])
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()
    zero = {"pairs": 0, **counts()}
    assert json.loads(site.tree()["run.json"])["counts"] == zero


def test_incomparable_world_keeps_its_order(tmp_path):
    """A shard of incomparable verdicts only (no arc): pairs (a, b) in order."""
    world = [[comparison("b_1", "a_1", INC, INC), comparison("a_1", "c_1", INC, MB)]]
    site = make_site(tmp_path, world, refusal_share=0.25)
    assert site.run().code == 0
    pairs = json.loads(site.tree()["shards/000000/pairs.json"])
    assert pairs == [pair(0, "b_1", "a_1", None), pair(0, "a_1", "c_1", None)]
