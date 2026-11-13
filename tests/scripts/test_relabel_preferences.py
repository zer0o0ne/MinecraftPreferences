"""Saved-vote migration, graph restoration, resume and source preservation."""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from datasets.preferences.acyclic_site import CONFIG, make_site
from datasets.preferences.acyclic_world import (
    MB,
    SB,
    START,
    SW,
    B,
    comparison,
    json_bytes,
    votes_run,
    write_votes,
)
from datasets.preferences.config_files import DIAGNOSTICS
from datasets.preferences.test_acyclic import pair
from training.pretrain.data.index_run import interrupting, repo_root

from mcpref.datasets.preferences import acyclic
from mcpref.diagnostics.label_distribution import read_distribution

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/relabel_preferences.py"
SPEC = importlib.util.spec_from_file_location("relabel_preferences", SCRIPT)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)
WORLD = [
    [
        comparison("a_1", "b_1", MB, B),
        comparison("b_1", "c_1", SB, SW),
        comparison("c_1", "a_1", SB, SW),
    ],
    [],
    [comparison("a_2", "b_2", SB, SB)],
]
OUTPUT = "w/corrected"


def tree(directory):
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def site(tmp_path):
    site = make_site(tmp_path, WORLD, weights=START)
    assert site.run().code == 0
    # Historical result: equal-cost 3-cycle cuts c->a; a->b was still comparable.
    shard = site.out / "shards/000000"
    (shard / "pairs.json").write_bytes(
        json_bytes(
            [
                pair(0, "a_1", "b_1", 0.6),
                pair(0, "b_1", "c_1", 0.6),
                pair(0, "c_1", "a_1", None),
            ]
        )
    )
    counts = {"comparable": 2, "removed": 1, "incomparable": 0, "dropped": 0}
    (shard / "counts.json").write_bytes(json_bytes(counts))
    head = json.loads((site.out / "run.json").read_text())
    head.pop("verdict_rule")
    head["counts"] = {"pairs": 4, **counts, "comparable": 3}
    (site.out / "run.json").write_bytes(json_bytes(head))
    return site


def run(site):
    with repo_root(site.repo):
        return migration.main([str(site.path), OUTPUT])


def test_rebuild_restores_graph_edge_and_preserves_sources(site):
    before = tree(site.repo)
    assert run(site) == 0
    for name, data in before.items():
        assert (site.repo / name).read_bytes() == data
    out = site.repo / OUTPUT
    pairs = json.loads((out / "shards/000000/pairs.json").read_text())
    assert pairs == [
        pair(0, "a_1", "b_1", None),
        pair(0, "b_1", "c_1", 0.6),
        pair(0, "c_1", "a_1", 0.6),
    ]
    report = read_distribution(out)
    assert report["counts"] == {"0.5": 1, "0.6": 2, "incomparable": 1}
    head = json.loads((out / "run.json").read_text())
    assert head["counts"] == {
        "pairs": 4,
        "comparable": 3,
        "removed": 0,
        "incomparable": 1,
        "dropped": 0,
    }
    assert head["verdict_rule"] == "two-order-conflict-v1"
    diag = json.loads((out / "diagnostics/run.json").read_text())
    assert diag["complete"]
    assert set(diag["channels"]) == {"label", "graph"}
    expected = tree(out)
    assert run(site) == 0
    assert tree(out) == expected


def test_interrupted_migration_matches_ordinary_run(site, monkeypatch):
    with monkeypatch.context() as mp:
        interrupting("correction shards", 1)(mp)
        assert run(site) == 1
    out = site.repo / OUTPUT
    kept = tree(out / "shards/000000")
    assert run(site) == 0
    assert tree(out / "shards/000000") == kept
    for k, (pairs, counts) in enumerate(site.expected()):
        assert (out / f"shards/{k:06d}/pairs.json").read_bytes() == json_bytes(pairs)
        assert (out / f"shards/{k:06d}/counts.json").read_bytes() == json_bytes(counts)


@pytest.mark.parametrize("change", ["votes", "config", "diagnostics"])
def test_changed_inputs_reject_resume(site, change, capsys):
    assert run(site) == 0
    out = site.repo / OUTPUT
    if change == "votes":
        path = site.votes / "shards/000000/answers.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = (
            site.path if change == "config" else site.repo / site.config["diagnostics"]
        )
        raw = yaml.safe_load(path.read_text())
        raw["refusal_share" if change == "config" else "seed"] = 0
        if change == "diagnostics":
            raw["seed"] = 7
        path.write_text(yaml.safe_dump(raw))
    before = tree(out)
    assert run(site) == 1
    assert "changed" in capsys.readouterr().err
    assert tree(out) == before


@pytest.mark.parametrize(
    "output", ["w", "w/votes/new", "w/labels", "w/diagnostics/new"]
)
def test_reject_overlapping_destination(site, output):
    before = tree(site.repo)
    with repo_root(site.repo):
        assert migration.main([str(site.path), output]) == 1
    assert tree(site.repo) == before


def test_reject_foreign_destination_and_resume_first_config_write(site):
    out = site.repo / OUTPUT
    out.mkdir()
    note = out / "note.txt"
    note.write_text("keep")
    assert run(site) == 1
    assert note.read_text() == "keep"
    note.unlink()
    (out / "acyclic.yaml.part").write_text("interrupted config")
    assert run(site) == 0


def test_cli_with_saved_votes(repo, tmp_path):
    with tempfile.TemporaryDirectory(dir=repo) as directory:
        work = Path(directory)
        relative = work.relative_to(repo).as_posix()
        write_votes(work / "votes", WORLD, votes_run(WORLD))
        diag = {**DIAGNOSTICS, "out_dir": f"{relative}/old-diagnostics"}
        (work / "diagnostics.yaml").write_text(yaml.safe_dump(diag))
        cfg = {
            **CONFIG,
            "weights": START,
            "votes_dir": f"{relative}/votes",
            "out_dir": f"{relative}/old-labels",
            "diagnostics": f"{relative}/diagnostics.yaml",
        }
        path = work / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), f"{relative}/new-labels"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["counts"]["incomparable"] == 1
        with repo_root(repo):
            effective = work / "new-labels/acyclic.yaml"
            acyclic.run(acyclic.load_config(effective), effective)
