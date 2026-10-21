"""`pretrain` of specs/pretrain.md (T) against the float64 oracle of loop_oracle.py: every
step's (1)-(7) through the checkpoints, `losses.json` and `model.pt` (with pairs: mini-batches
with and without a pair, clips in two items; $b_\\mu$ dividing $b$, not dividing it, $\\ge b$;
without pairs; pairs only outside the training part); the modes; `run.json`; the
`already done` paths."""

import json

import pytest
import yaml
from progress_bars import Bar, assert_progress, screen

from .index_oracle import json_bytes, sha
from .loop_checks import assert_run
from .loop_modules import Phi
from .loop_oracle import train
from .loop_run import Loop
from .loop_world import NO_PAIRS, VALIDATION_PAIRS

CASES = {  # sizes 3, 3, 2
    "b 2, micro 1 (T = 10)": {},
    "b 3, micro 2 (T = 6)": {"batch": 3, "micro_batch": 2},
    "b 2, micro 5": {"micro_batch": 5},
}


@pytest.mark.parametrize("changes", CASES.values(), ids=CASES)
def test_training_matches_oracle(tmp_path, changes):
    """Each checkpoint (all kept) and model.pt against the oracle's steps."""
    loop = Loop(tmp_path, keep_checkpoints=20, **changes).write()
    loop.run()
    steps, states = train(loop.world, loop.config)
    assert_run(loop, steps, states, kept=20)


def test_both_branches_of_clipping_and_pairs(tmp_path):
    """The starting case (grad_clip = 1, b = 2): steps with the gradients scaled (4) and
    without; mini-batches with a pair and without (P null)."""
    loop = Loop(tmp_path, keep_checkpoints=20).write()
    loop.run()
    steps, states = train(loop.world, loop.config)
    assert {s.clipped for s in steps} == {True, False}
    assert {s.losses["P"] is None for s in steps} == {True, False}
    assert_run(loop, steps, states, kept=20)


@pytest.mark.parametrize(
    "world", [NO_PAIRS, VALIDATION_PAIRS], ids=["none", "validation"]
)
def test_without_pairs_in_the_training_part(tmp_path, world):
    """N_R = 0: the four modules, no P term, `P` null in every entry; the inputs bar counts
    Q_c and Q_l when the index has pairs."""
    loop = Loop(tmp_path, world=world, keep_checkpoints=20).write()
    r = loop.run()
    steps, states = train(world, loop.config)
    assert all(s.losses["P"] is None for s in steps)
    assert_run(loop, steps, states, kept=20)
    shards = 6 if world.pairs else 2
    assert screen(r.err)[0] == Bar("pretrain inputs", shards, shards, "shard")


def test_kept_checkpoints_and_files(tmp_path):
    """k = 2: checkpoint.pt of the last two checkpoints only, losses.json of all;
    {out_dir} holds run.json, model.pt and the checkpoints, nothing else."""
    loop = Loop(tmp_path).write()
    loop.run()
    steps, states = train(loop.world, loop.config)
    assert_run(loop, steps, states, kept=2)
    files = {"run.json", "model.pt"}
    for j in range(4):
        files.add(f"checkpoints/{j:06d}/losses.json")
    files |= {"checkpoints/000002/checkpoint.pt", "checkpoints/000003/checkpoint.pt"}
    assert set(loop.tree()) == files


def test_losses_json_format(tmp_path):
    """losses.json: json.dumps(value, indent=2) + "\\n", UTF-8."""
    loop = Loop(tmp_path).write()
    loop.run()
    for j in range(4):
        data = (loop.checkpoint_dir(j) / "losses.json").read_bytes()
        assert data == json_bytes(json.loads(data))


def test_modules_put_in_training_mode(tmp_path):
    """Modules given in eval mode: every call of a run in training mode."""
    loop = Loop(tmp_path).write()
    mods = loop.modules()
    log = []
    for m in mods.values():
        m.eval()
        m.log = log
    loop.run(mods)
    assert {name for name, _, _ in log} == set(mods)
    assert all(training for _, training, _ in log)


def test_run_json(tmp_path):
    """After (R) 6 the run's config and fingerprints; after (T) `complete` added."""
    loop = Loop(tmp_path).write()
    first = {}

    def peek(mp):
        def forward(self, clips):
            first.setdefault("run", (loop.out / "run.json").read_bytes())
            return real(self, clips)

        real = Phi.forward
        mp.setattr(Phi, "forward", forward)

    loop.run(patch=peek)
    value = {
        "config": yaml.safe_load(loop.path.read_bytes()),
        "sha256": {
            "config": sha(loop.path.read_bytes()),
            "index": sha((loop.repo / "w/index/run.json").read_bytes()),
        },
    }
    assert first["run"] == json_bytes(value)
    assert (loop.out / "run.json").read_bytes() == json_bytes(
        value | {"complete": True}
    )


def test_model_present_already_done(tmp_path):
    """model.pt present: the line `pretrain: already done`, no module called, model.pt and
    the checkpoints untouched, run.json complete."""
    loop = Loop(tmp_path).write()
    loop.run()
    before = loop.tree()
    run = json.loads(before["run.json"])
    (loop.out / "run.json").write_bytes(json_bytes({k: run[k] for k in ("config", "sha256")}))  # fmt: skip
    mods = loop.modules()
    r = loop.run(mods)
    assert screen(r.err) == [Bar("pretrain inputs", 6, 6, "shard"), "pretrain: already done"]  # fmt: skip
    assert_progress(r.err)
    assert all(m.calls == 0 for m in mods.values())
    assert loop.tree() == before and r.out == ""


def test_restored_last_step_already_done(tmp_path):
    """model.pt deleted after a run: the checkpoint at T restored, the line `pretrain:
    already done` instead of the bar, no module called, model.pt written again with the
    same tensors; the other files untouched."""
    loop = Loop(tmp_path).write()
    loop.run()
    before, model = loop.tree(), loop.model()
    (loop.out / "model.pt").unlink()
    mods = loop.modules()
    r = loop.run(mods)
    assert screen(r.err) == [Bar("pretrain inputs", 6, 6, "shard"), "pretrain: already done"]  # fmt: skip
    assert all(m.calls == 0 for m in mods.values())
    after = loop.tree()
    assert set(after) == set(before)
    assert all(after[k] == before[k] for k in after if k != "model.pt")
    again = loop.model()
    assert set(again) == set(model)
    for name in model:
        assert set(again[name]) == set(model[name])
        assert all(again[name][k].equal(v) for k, v in model[name].items())
