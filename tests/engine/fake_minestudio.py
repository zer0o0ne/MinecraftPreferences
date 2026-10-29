"""A fake `minestudio.simulator` for `check` of specs/engine.md (8)-(10), with the contract of
[MS]: `MinecraftSim(**kwargs)`, `reset()`, `noop_action()`, `step(action)` returning
`(obs, reward, terminated, truncated, info)`, `close()`. It is served by a meta path finder,
so that its import runs when the code imports it and records the variables of (7) then.
Every call is logged in order and advances a fake `time.perf_counter` clock by a dyadic
cost (the warm-up steps, `MinecraftSim`, `reset` and `close` by much more), so that ρ is
exact and a timed interval in the wrong place shows."""

import importlib.abc
import importlib.util
import os
import sys
from dataclasses import dataclass, field

import tqdm

from .engine_world import VARS

STEP, NOOP, UPDATE = (
    1 / 32,
    1 / 64,
    1 / 64,
)  # a timed step costs 1/16 s with its bar update
WARM = 4.0  # a warm-up step's and its noop_action's cost


@dataclass
class Fake:
    """`warmup`: the steps before the timed ones (their cost WARM); `end`: (k, flag) — step
    k returns `flag` (`terminated` or `truncated`) true; `fail`: `init`, `reset` or a step
    k raising RuntimeError."""

    warmup: int
    step_cost: float = STEP
    end: tuple | None = None
    fail: object = None
    t: float = 1000.0
    calls: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    def perf_counter(self) -> float:
        return self.t

    def call(self, name: str, *details, cost: float = 0.0) -> None:
        self.calls.append((name, *details))
        self.t += cost
        if self.fail == name or (name == "step" and self.fail == details[0]):
            raise RuntimeError(f"{name} failed in the fake")

    def sim_class(self) -> type:
        fake = self

        class MinecraftSim:
            def __init__(self, *args, **kwargs):
                print("fake minestudio: launching")
                fake.call("init", args, kwargs, cost=100.0)

            def reset(self):
                fake.call("reset", cost=100.0)
                print("fake minestudio: reset", file=sys.stderr)
                return {"image": None}, {}

            def noop_action(self):
                warm = len(fake.actions) < fake.warmup
                fake.call("noop_action", cost=WARM if warm else NOOP)
                fake.actions.append(object())
                return fake.actions[-1]

            def step(self, action):
                k = sum(c[0] == "step" for c in fake.calls) + 1
                fake.call(
                    "step",
                    k,
                    action is fake.actions[-1],
                    cost=WARM if k <= fake.warmup else fake.step_cost,
                )
                flags = {f: (k, f) == fake.end for f in ("terminated", "truncated")}
                return {"image": None}, 0.0, flags["terminated"], flags["truncated"], {}

            def close(self):
                fake.call("close", cost=1000.0)
                print("fake minestudio: closed")

        return MinecraftSim


class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves `minestudio` and `minestudio.simulator`; records the variables at the
    simulator's import."""

    def __init__(self, fake: Fake):
        self.fake = fake

    def find_spec(self, name, path, target=None):
        if name in ("minestudio", "minestudio.simulator"):
            return importlib.util.spec_from_loader(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        if module.__name__ == "minestudio.simulator":
            self.fake.imports.append({v: os.environ.get(v) for v in VARS})
            module.MinecraftSim = self.fake.sim_class()


def install(fake: Fake, monkeypatch) -> None:
    """The fake on sys.meta_path, the real modules out of sys.modules, the clock as
    `time.perf_counter`, advanced by UPDATE per update of the `check` bar; undone by
    `uninstall` and monkeypatch."""
    for name in [n for n in sys.modules if n.split(".")[0] == "minestudio"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [Finder(fake), *sys.meta_path])
    monkeypatch.setattr("time.perf_counter", fake.perf_counter)
    real = tqdm.tqdm.update

    def update(bar, n=1):
        fake.t += UPDATE if bar.desc == "check" else 0.0
        return real(bar, n)

    monkeypatch.setattr(tqdm.tqdm, "update", update)


def uninstall() -> None:
    """The fake's modules out of sys.modules (before monkeypatch restores the others)."""
    for name in [n for n in sys.modules if n.split(".")[0] == "minestudio"]:
        del sys.modules[name]
