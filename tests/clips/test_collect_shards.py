"""(6) of specs/perception-data.md: shards written in draw order, the lookahead bound of the
planner, the cut at N with the running work ignored, outputs independent of `workers`."""

import json
import threading
import time

import pytest

from .collect_expected import CHECKS
from .collect_gamma import Classifier
from .collect_out import assert_outputs
from .collect_site import asked_files


def use_out(site, name: str) -> None:
    site.out = site.tmp / name
    site.fields["out_dir"] = site.rel(site.out)


@pytest.mark.parametrize("cut", [False, True])
def test_outputs_do_not_depend_on_workers(site, cut):
    """Byte for byte: the shards, the indices, run.json but its `workers` and `out_dir`."""
    n = site.expected().shards[2].clips - 1 if cut else 100
    trees = {}
    for workers in (1, 3):
        use_out(site, f"out{workers}")
        site.run(workers=workers, n_clips=n)
        assert_outputs(site, site.expected(n_clips=n))
        trees[workers] = site.raw()
        run = json.loads(trees[workers].pop("run.json"))
        assert run["config"].pop("workers") == workers
        assert run["config"].pop("out_dir") == site.fields["out_dir"]
        trees[workers]["run.json"] = run
    assert trees[1] == trees[3]


def wait(condition, seconds: float = 4.0) -> bool:
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.02)
    return condition()


@pytest.mark.parametrize("seed, a, workers", [(1, 5, 1), (1, 5, 2), (0, 1, 3)])
def test_lookahead_bound(site, seed, a, workers):
    """The video of draw a's candidate is held: the planner starts every draw up to the
    one holding the `workers`-th candidate from a (seed 1: draw 6 has no candidate), no
    later one; the shards before a are written, none from a on."""
    ex = site.expected(seed=seed)
    cands = [
        k for k, s in enumerate(ex.shards) if s.records[-1]["reason"] not in CHECKS
    ]
    last = cands[cands.index(a) + workers - 1]
    gate = threading.Event()
    candidate = ex.shards[a].records[-1]["relpath"]
    site.gates[site.path(candidate, ".mp4")] = gate
    thread = site.in_thread(Classifier(site.world), seed=seed, workers=workers)
    try:
        firsts = [next(r["relpath"] for r in s.records if r["reason"] != "version")
                  for s in ex.shards[: last + 1]]  # fmt: skip
        assert wait(lambda: all(asked_files(site, [x]) for x in firsts))
        assert wait(lambda: site.complete() == [f"{k:06d}" for k in range(a)])
        time.sleep(0.3)
        later = [r["relpath"] for s in ex.shards[last + 1 :] for r in s.records]
        assert later and asked_files(site, later) == []
        assert site.complete() == [f"{k:06d}" for k in range(a)]
    finally:
        gate.set()
        thread.join(8.0)
    assert site.error is None
    assert_outputs(site, ex)


@pytest.mark.parametrize("n", [3, 4], ids=["inside", "at_end"])
def test_cut_at_n_ignores_the_running_work(site, n):
    """Seed 0, three workers: shard 2 (a6) reaches N; its video is held until gamma has
    raised for draw 3's candidate (b7), which ran meanwhile: the run ends normally with the
    shards of N, every decode ended, staging removed."""
    ex = site.expected(n_clips=n)
    assert len(ex.shards) == 3 and ex.shards[1].clips < n <= ex.shards[2].clips
    gamma = Classifier(site.world, fail_stem="b7")
    site.gates[site.path("data/6.12/a6.mp4", ".mp4")] = gamma.raised_now
    site.run(gamma, workers=3, n_clips=n)
    assert gamma.raised_now.is_set()
    assert_outputs(site, ex)
    assert len(site.decodes()) == len(site.decodes("end"))


def staging(site) -> dict | None:
    """site.tree("staging"), None while a file changes under it (the run goes on)."""
    try:
        return site.tree("staging")
    except (FileNotFoundError, ValueError):
        return None


def test_unwritten_candidates_wait_in_staging(site):
    """Seed 0, three workers, c8's video (draw 1) held: draws 2 (a6) and 3 (b7) are
    processed; their clips wait in staging/<k>/<id>.part/ with the files of Datasets, their
    videos deleted; shard 0 written and its staging removed."""
    ex = site.expected()
    gate = threading.Event()
    site.gates[site.path("data/8.1/c8.mp4", ".mp4")] = gate
    thread = site.in_thread(Classifier(site.world), workers=3)
    try:
        want = {f"staging/{k}/{p.replace('clips/', '', 1).replace('/', '.part/', 1)}": x
                for k in (2, 3) for p, x in ex.shards[k].files.items()
                if p.startswith("clips/")}  # fmt: skip
        assert wait(lambda: staging(site) == want), sorted(staging(site) or {})
        assert site.complete() == ["000000"] and not (site.out / "staging/0").exists()
    finally:
        gate.set()
        thread.join(8.0)
    assert site.error is None
    assert_outputs(site, ex)
