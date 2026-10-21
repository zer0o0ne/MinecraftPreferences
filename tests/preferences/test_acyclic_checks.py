"""(R), (I) of specs/acyclic.md: each check, its message (paths as written) and order; (R)
before the inputs (no bar, nothing written); the fingerprints of (I) compared after the
pass over the inputs (its bar drawn), in the order config, votes, pair_clips."""

import json

import pytest
from progress_bars import Bar

from .acyclic_site import aborted, make_site
from .acyclic_world import HAND, TOKENS, json_bytes, sha, votes_run

NOT_COMPLETE = "w/votes is not complete"
INPUTS = Bar("correction inputs", 4, 4, "shard")  # HAND: 4 complete shards of V_l


def run_without(key: str, value=None) -> dict:
    run = votes_run(HAND)
    if value is None:
        del run[key]
    else:
        run[key] = value
    return run


@pytest.mark.parametrize(
    "run",
    [None, run_without("complete"), run_without("complete", False),
     {**votes_run(HAND), "complete": None}],
    ids=["no run.json", "no marker", "false", "null"],
)  # fmt: skip
def test_votes_not_complete(tmp_path, run):
    """(R) 1: `finished_run` of V_l, marker `complete`; before any bar, nothing written."""
    site = make_site(tmp_path, run=run)
    r = site.run()
    assert aborted(r, NOT_COMPLETE) == []
    assert not site.out.exists()


def answers_run(tokens: list) -> dict:
    run = votes_run(HAND)
    run["config"]["answers"] = [{"token": t, "description": t} for t in tokens]
    return run


@pytest.mark.parametrize(
    "tokens",
    [[*TOKENS[1:], TOKENS[0]], list(TOKENS[:-1]), [*TOKENS, "MAYBE"]],
    ids=["order", "missing", "extra"],
)
def test_weights_differ_from_answers(tmp_path, tokens):
    """(R) 2: the keys of `weights`, in order, equal the tokens of V_l's answers."""
    site = make_site(tmp_path, run=answers_run(tokens))
    r = site.run()
    message = f"weights of {site.path} differ from the answers of w/votes"
    assert aborted(r, message) == []
    assert not site.out.exists()


def test_completeness_checked_first(tmp_path):
    """(R) 1 before (R) 2."""
    run = answers_run(list(TOKENS[:-1]))
    del run["complete"]
    assert aborted(make_site(tmp_path, run=run).run(), NOT_COMPLETE) == []


def change_config(site) -> None:
    site.path.write_text(site.path.read_text() + "# edited\n")


def change_votes(site) -> None:
    path = site.votes / "shards/000000/answers.json"
    path.write_bytes(
        path.read_bytes().replace(b'"label": "WORSE"', b'"label": "BETTER"', 1)
    )


def change_pool(site) -> None:
    path = site.votes / "run.json"
    run = json.loads(path.read_bytes())
    run["sha256"]["pool"] = sha(b"other clips")
    path.write_bytes(json_bytes(run))


CHANGES = {"config": change_config, "votes": change_votes, "pair_clips": change_pool}


@pytest.mark.parametrize(
    "names",
    [["config"], ["votes"], ["pair_clips"], ["config", "votes"], ["votes", "pair_clips"],
     ["config", "pair_clips"]],
)  # fmt: skip
def test_fingerprints_after_the_inputs(tmp_path, names):
    """(I): `check_fingerprints` of the existing run.json after the pass over the inputs:
    the first changed name of config, votes, pair_clips aborts; {out_dir} unchanged."""
    site = make_site(tmp_path)
    assert site.run().code == 0
    before = site.tree()
    for n in names:
        CHANGES[n](site)
    r = site.run()
    message = f"{names[0]} changed since the run started; delete w/labels"
    assert aborted(r, message) == [INPUTS]
    assert site.tree() == before


def test_weights_checked_before_the_fingerprints(tmp_path):
    """(R) 2 aborts before the inputs are read, also when a fingerprint changed."""
    site = make_site(tmp_path)
    assert site.run().code == 0
    change_votes(site)
    (site.votes / "run.json").write_bytes(json_bytes(answers_run(list(TOKENS[:-1]))))
    message = f"weights of {site.path} differ from the answers of w/votes"
    assert aborted(site.run(), message) == []


def test_unchanged_inputs_pass(tmp_path):
    """A rerun with the same config and inputs passes the checks (exit 0)."""
    site = make_site(tmp_path)
    assert site.run().code == 0
    assert site.run().code == 0
    assert site.tree() == site.expected_tree()
