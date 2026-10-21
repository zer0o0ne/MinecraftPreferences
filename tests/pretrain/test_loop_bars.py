"""The progress bars of specs/pretrain.md (Interface): `pretrain inputs` (unit shard, the
complete shards of C, Q_c, Q_l) and `pretrain` (unit step, total T, its postfix `phi`, `pi`,
`P` (left out when null), `loss` of the last step by `set_postfix`); nothing on stdout."""

import pytest
import tqdm
from progress_bars import Bar, assert_progress, screen

from .loop_run import Loop
from .loop_world import NO_PAIRS, WORLD


def recording(calls: list):
    """A patch recording the mapping of every `set_postfix` call of the bar `pretrain`."""
    real = tqdm.tqdm.set_postfix

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if self.desc == "pretrain":
            calls.append({**(ordered_dict or {}), **kwargs})
        return real(self, ordered_dict, refresh, **kwargs)

    return lambda mp: mp.setattr(tqdm.tqdm, "set_postfix", set_postfix)


def shown(entry: dict) -> dict:
    """The postfix of a losses.json entry: phi, pi, P unless null, loss."""
    keys = ["phi", "pi", "P", "loss"]
    return {k: entry[k] for k in keys if entry[k] is not None}


RUNS = [(WORLD, 6, 10, 4), (NO_PAIRS, 2, 6, 2)]  # shards of the inputs, T, checkpoints


@pytest.mark.parametrize("world, shards, T, n", RUNS, ids=["pairs", "none"])
def test_bars_of_a_run(tmp_path, world, shards, T, n):
    """The two bars, each drawn from 0 to its total and closed; the postfix of every step
    its losses (P left out on the steps without a pair); the closed bar shows the last
    step's, formatted by tqdm."""
    loop = Loop(tmp_path, world=world).write()
    calls = []
    r = loop.run(patch=recording(calls))
    assert r.out == ""
    assert_progress(r.err)
    entries = [e for j in range(n) for e in loop.losses(j)]
    assert calls == [shown(e) for e in entries]
    assert {"P" in c for c in calls} == ({True, False} if world.pairs else {False})
    inputs, bar = screen(r.err)
    assert inputs == Bar("pretrain inputs", shards, shards, "shard")
    assert (bar.desc, bar.n, bar.total, bar.unit) == ("pretrain", T, T, "step")
    postfix = dict(x.split("=", 1) for x in bar.postfix.split(", "))
    assert postfix == {
        k: tqdm.tqdm.format_num(v) for k, v in shown(entries[-1]).items()
    }
