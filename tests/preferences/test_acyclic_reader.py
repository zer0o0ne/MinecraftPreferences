"""Produces of specs/acyclic.md: block train-index (`specs/train-index.md`) reads a $Q_\\ell$
written here — its (R) 4 (`complete`) and the `pair_clips` check of (R') pass when V_l
labels the Q_c given to it, and the latter aborts when V_l labels other clips."""

import contextlib
import io
import json

import pytest
import yaml
from pretrain.index_oracle import clips_fingerprint
from pretrain.index_run import repo_root
from pretrain.index_world import clips_run, record, write_clips
from progress_bars import assert_aborted, tqdm_defaults

from mcpref.pretrain.index import main as train_index

from .acyclic_site import make_site, shard_files
from .acyclic_world import HAND, sha, votes_run

INDEX = {"clips_dir": "w/clips", "pair_clips_dir": "w/pair_clips",
         "pair_labels_dir": "w/labels", "out_dir": "w/index", "validation": "0.3",
         "seed": 7, "items_per_shard": 3}  # fmt: skip


def pair_clips(world: list) -> list:
    """Q_c's records: per shard, one record per episode of its clips (`[]`: no clip)."""
    shards = []
    for shard in world:
        episodes = {}
        for a, b, _ in shard:
            for x in (a, b):
                episodes.setdefault(x["episode"], []).append(x["id"])
        records = [record(e, tuple(dict.fromkeys(ids))) for e, ids in episodes.items()]
        shards.append(records or [record("6.13/none.mp4", ())])
    return shards


def run_index(site) -> tuple[int, str]:
    """train-index's main on INDEX under the site's repository root."""
    path = site.tmp / "train-index.yaml"
    path.write_text(yaml.safe_dump(INDEX, sort_keys=False))
    err = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, repo_root(site.repo):
        tqdm_defaults(mp)
        mp.chdir(site.cwd)
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = train_index([str(path)])
    return code, err.getvalue()


def inputs(tmp_path, labels_other_clips: bool):
    """C, Q_c (the clips of HAND), then V_l labelling Q_c (or other clips) and this
    block's run on it."""
    site = make_site(tmp_path, write=False)
    q_shards = pair_clips(HAND)
    q_run = clips_run(sum(len(r["clips"]) for s in q_shards for r in s))
    write_clips(site.repo / "w/clips", [[record("6.13/u.mp4", ("u_1",))]], clips_run(1))
    write_clips(site.repo / "w/pair_clips", q_shards, q_run)
    fp = clips_fingerprint(
        q_run, shard_files(site.repo / "w/pair_clips/shards", "records.json")
    )
    site.write(votes_run(HAND, pool=sha(b"other") if labels_other_clips else fp))
    assert site.run().code == 0
    return site


def test_train_index_reads_the_labels(tmp_path):
    """train-index runs to completion on the output: its 12 pairs indexed."""
    site = inputs(tmp_path, labels_other_clips=False)
    code, err = run_index(site)
    assert code == 0, err
    run = json.loads((site.repo / "w/index/run.json").read_text())
    assert run["complete"] is True
    assert run["counts"]["train"]["pairs"] + run["counts"]["validation"]["pairs"] == 12


def test_train_index_aborts_on_other_clips(tmp_path):
    """V_l labels other clips: `pair_clips` differs, train-index's (R') aborts."""
    site = inputs(tmp_path, labels_other_clips=True)
    code, err = run_index(site)
    assert_aborted(code, err)
    message = "RuntimeError: w/labels labels other clips than w/pair_clips"
    assert err.splitlines()[-1] == f"mcpref.pretrain.index: {message}"
