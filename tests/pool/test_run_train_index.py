"""Produces of specs/preference-pool.md: train-index (specs/train-index.md (R) 1-3) accepts
a Q_c written here with a C written by perception-data with the same config — (R) 4, the
missing pair labels, is the first check failing; a C classified by another gamma fails
(R) 3."""

import yaml
from clips.collect_gamma import Classifier

from mcpref.clips.collect import collect as perception_collect
from mcpref.clips.collect import load_config as load_perception
from mcpref.pretrain.index import main

from .pool_world import chain

PREFIX = "mcpref.pretrain.index: RuntimeError: "


def last_line(site, capfd, gid: str) -> str:
    """C by perception-data with gamma_id `gid`, Q_c by the pool; the last line of
    train-index's stderr."""
    site.run()
    p = load_perception(site.paths["perception-data"])
    perception_collect(p, Classifier(site.world), gid)
    config = {"clips_dir": "perception-out", "pair_clips_dir": "out",
              "pair_labels_dir": "labels", "validation": "0.05", "seed": 0,
              "items_per_shard": 4, "out_dir": "index"}  # fmt: skip
    path = site.repo / "train-index.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    capfd.readouterr()
    assert main([str(path)]) == 1
    return capfd.readouterr().err.splitlines()[-1]


def world() -> dict:
    return {"9xx": chain("9.1", "t", "ab" * 6, ["w" * 16] * 2)}


def test_checks_1_to_3_pass_on_q_c(make_site, capfd):
    site = make_site(episodes=world())
    assert last_line(site, capfd, site.gid) == PREFIX + "labels is not complete"


def test_check_3_fails_for_another_classifier(make_site, capfd):
    site = make_site(episodes=world())
    got = last_line(site, capfd, "another gamma")
    assert got == PREFIX + "out differs from perception-out in ['classifier']"
