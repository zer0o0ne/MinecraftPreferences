"""Reruns of `pretrain` of specs/pretrain.md (T) 1-3 (Interface: checkpoints): a run
interrupted after a checkpoint inside an epoch and at an epoch end, continued with fresh
components and torch's generator elsewhere, gives bitwise the parameters, checkpoints and
`losses.json` of an uninterrupted run; the checkpoints written before the bar advances; the
cleanup of `checkpoint.pt` files a crash left (prune at the start)."""

import os
from pathlib import Path

import pytest
import torch
from progress_bars import Bar, screen, states

from .index_run import Interrupted, interrupting
from .loop_checks import checkpoint_steps
from .loop_modules import Phi
from .loop_run import Loop

T = 10  # M = 5 mini-batches per epoch, E = 2


def equal_states(a: dict, b: dict) -> bool:
    """{name: state_dict}: the same tensors, bitwise."""
    return a.keys() == b.keys() and all(
        a[k].keys() == b[k].keys() and all(a[k][n].equal(b[k][n]) for n in a[k])
        for k in a
    )


def equal_optimizers(a: dict, b: dict) -> bool:
    same = (
        a["param_groups"] == b["param_groups"]
        and a["state"].keys() == b["state"].keys()
    )
    return same and all(
        all(a["state"][i][k].equal(b["state"][i][k]) for k in a["state"][i])
        for i in a["state"]
    )


def assert_same_outputs(cut: Loop, whole: Loop) -> None:
    """Every losses.json byte-equal; the checkpoint.pt kept by both (the same ones) and
    model.pt with bitwise equal contents."""
    n = len(checkpoint_steps(T, whole.config["checkpoint_steps"]))
    for j in range(n):
        a, b = cut.checkpoint_dir(j), whole.checkpoint_dir(j)
        assert (a / "losses.json").read_bytes() == (b / "losses.json").read_bytes()
        assert (a / "checkpoint.pt").exists() == (b / "checkpoint.pt").exists()
        if (b / "checkpoint.pt").exists():
            x, y = cut.checkpoint(j), whole.checkpoint(j)
            assert [
                x[k] for k in ("step", "epoch", "schedule_rng", "batch_in_epoch")
            ] == [y[k] for k in ("step", "epoch", "schedule_rng", "batch_in_epoch")]
            assert equal_states(x["modules"], y["modules"])
            assert equal_optimizers(x["optimizer"], y["optimizer"])
            assert x["torch_rng"].equal(y["torch_rng"]) and x["cuda_rng"] is None
    assert not cut.checkpoint_dir(n).exists()
    assert equal_states(cut.model(), whole.model())


CUTS = {  # N_c, the step at which the bar raises, the step of the last checkpoint
    "inside epoch 1": (3, 5, 3),
    "at checkpoint 6, inside epoch 2": (3, 6, 6),
    "at the end of epoch 1": (5, 5, 5),
    "after the end of epoch 1": (5, 8, 5),
}


@pytest.mark.parametrize("n_c, at, last", CUTS.values(), ids=CUTS)
def test_continued_run_is_bitwise_equal(tmp_path, n_c, at, last):
    whole = Loop(tmp_path / "whole", checkpoint_steps=n_c, keep_checkpoints=20).write()
    whole.run()
    cut = Loop(tmp_path / "cut", checkpoint_steps=n_c, keep_checkpoints=20).write()
    cut.run(patch=interrupting("pretrain", at), raises=Interrupted)
    done = [t for t in checkpoint_steps(T, n_c) if t <= at]
    assert done[-1] == last
    assert sorted(p.name for p in (cut.out / "checkpoints").iterdir()) == [
        f"{j:06d}" for j in range(len(done))
    ]
    assert not (cut.out / "model.pt").exists()
    torch.manual_seed(12345)
    torch.rand(7)
    r = cut.run()
    drawn = states(r.err, "pretrain")
    assert drawn[0].n == last and drawn[-1].n == T and drawn[0].total == T
    assert screen(r.err)[0] == Bar("pretrain inputs", 6, 6, "shard")
    assert_same_outputs(cut, whole)


def test_kept_checkpoint_across_a_rerun(tmp_path):
    """k = 1, N_c = 3, cut at step 8 (checkpoints at 3, 6): checkpoint.pt of the last
    checkpoint only, before and after the rerun."""
    whole = Loop(tmp_path / "whole", keep_checkpoints=1).write()
    whole.run()
    cut = Loop(tmp_path / "cut", keep_checkpoints=1).write()
    cut.run(patch=interrupting("pretrain", 8), raises=Interrupted)
    kept = [(cut.checkpoint_dir(j) / "checkpoint.pt").exists() for j in range(2)]
    assert kept == [False, True]
    cut.run()
    assert_same_outputs(cut, whole)


def crash_at_prune(mp) -> None:
    """os.unlink raises on a checkpoint.pt: a crash before a deletion of (T) 3."""
    real = os.unlink

    def unlink(path, *args, **kwargs):
        if Path(path).name == "checkpoint.pt":
            raise Interrupted("crash")
        return real(path, *args, **kwargs)

    mp.setattr(os, "unlink", unlink)


def test_rerun_deletes_what_a_crash_left(tmp_path):
    """N_c = 1, k = 2: a crash at the deletion after checkpoint 2 leaves checkpoints 0-2
    with checkpoint.pt; the rerun deletes checkpoint 0's (then syncs its directory) before
    its first step, and ends as an uninterrupted run."""
    cfg = {"checkpoint_steps": 1, "keep_checkpoints": 2}
    whole = Loop(tmp_path / "whole", **cfg).write()
    whole.run()
    cut = Loop(tmp_path / "cut", **cfg).write()
    cut.run(patch=crash_at_prune, raises=Interrupted)
    pts = [cut.checkpoint_dir(j) / "checkpoint.pt" for j in range(4)]
    assert [p.exists() for p in pts] == [True, True, True, False]
    seen, synced = [], []

    def watch(mp):
        real_forward, real_fsync = Phi.forward, os.fsync

        def forward(self, clips):
            seen.append([p.exists() for p in pts[:3]])
            return real_forward(self, clips)

        def fsync(fd):
            synced.append((os.readlink(f"/proc/self/fd/{fd}"), pts[0].exists()))
            real_fsync(fd)

        mp.setattr(Phi, "forward", forward)
        mp.setattr(os, "fsync", fsync)

    cut.run(patch=watch)
    assert seen[0] == [False, True, True]
    assert (str(cut.checkpoint_dir(0)), False) in synced
    assert_same_outputs(cut, whole)
