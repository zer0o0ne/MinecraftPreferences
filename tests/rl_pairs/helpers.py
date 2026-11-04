"""Black-box fixtures and disk readers for specs/rl-preference-pairs.md."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mcpref.rl_pairs.config import RLPreferencePairsConfig
from mcpref.rl_pairs.types import StateRef


def config(root: Path, **changes) -> RLPreferencePairsConfig:
    values = {
        "history_dir": str(root / "history"),
        "out_dir": str(root / "out"),
        "sample_states": 2,
        "temporal_per_state": 2,
        "nearest_per_state": 0,
        "branch": 0,
        "beam_width": 0,
        "beam_depth": 0,
        "capacity": 100,
        "items_per_shard": 2,
        "history_per_shard": 2,
        "base_threshold": 0.5,
        "temporal_penalty": 1.0,
        "threshold_scale": 0.95,
    }
    return RLPreferencePairsConfig(**(values | changes))


def as_dict(ref: StateRef) -> dict:
    return {"episode": ref.episode, "index": ref.index}


def from_dict(value: dict) -> StateRef:
    return StateRef(value["episode"], value["index"])


def logit(p: float) -> float:
    return math.log(p / (1 - p))


class Encoder:
    def __init__(self, values: dict[StateRef, tuple[float, ...]]):
        self.values = values
        self.calls = []

    def __call__(self, refs):
        assert not torch.is_grad_enabled()
        refs = tuple(refs)
        self.calls.append(refs)
        return [torch.tensor([[self.values.get(r, (float(r.index + 100),))]],
                             dtype=torch.float64) for r in refs]  # fmt: skip


class Preference(torch.nn.Module):
    def __init__(self, results=None, default=(0.5, 0.0)):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))
        self.results = results or {}
        self.default = default
        self.calls = []

    def forward(self, a, b):
        assert not torch.is_grad_enabled() and not self.training
        assert a.shape[0] == b.shape[0] == 1
        av, bv = tuple(a[0, -1].tolist()), tuple(b[0, -1].tolist())
        self.calls.append((av, bv))
        p, inc = self.results.get((av, bv), self.default)
        return (
            torch.tensor([logit(p)], dtype=a.dtype),
            torch.tensor([logit(inc)], dtype=a.dtype)
            if inc not in (0, 1)
            else torch.tensor([math.copysign(30.0, inc - 0.5)], dtype=a.dtype),
        )


def encoder_for(states, dimension=2):
    return Encoder({r: (float(i + 1), *(0.0 for _ in range(dimension - 1)))
                    for i, r in enumerate(states)})  # fmt: skip


def shards(directory: Path, filename: str) -> list:
    if not directory.exists():
        return []
    return [
        json.loads((p / filename).read_text())
        for p in sorted(directory.iterdir())
        if p.is_dir() and not p.name.endswith(".part")
    ]


def items(run: Path) -> list[dict]:
    return [item for shard in shards(run / "shards", "items.json") for item in shard]


def vertices(history: Path, cycle: int) -> list[dict]:
    return [v for shard in shards(history / f"{cycle:06d}" / "shards", "vertices.json")
            for v in shard]  # fmt: skip


def work(run: Path) -> list[dict]:
    directory = run / "work"
    return [
        json.loads((p / "result.json").read_text()) for p in sorted(directory.iterdir())
    ]


def comparison(a: StateRef, b: StateRef, p: float, inc: float, n: int) -> dict:
    return as_dict(b) | {"p": p, "inc": inc, "n": n}


def vertex(ref: StateRef, key, unused=0, comparisons=()) -> dict:
    return as_dict(ref) | {"key": list(key), "unused": unused,
                           "comparisons": list(comparisons)}  # fmt: skip


def write_history(cfg, cycle: int, records: list[dict]) -> None:
    root = Path(cfg.history_dir) / f"{cycle:06d}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "cycle": cycle,
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )
    for start in range(0, len(records), cfg.history_per_shard):
        shard = root / "shards" / f"{start // cfg.history_per_shard:06d}"
        shard.mkdir(parents=True, exist_ok=True)
        part = records[start : start + cfg.history_per_shard]
        (shard / "vertices.json").write_text(json.dumps(part, indent=2) + "\n")
    if not records:
        shard = root / "shards" / "000000"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "vertices.json").write_text("[]\n")


def write_work(directory: Path, comparisons: list[list[dict]], seed: StateRef) -> None:
    for k, group in enumerate(comparisons):
        shard = directory / f"{k:06d}"
        shard.mkdir(parents=True, exist_ok=True)
        result = {"seed": as_dict(seed), "alternatives": [], "comparisons": group,
                  "rng": {}}  # fmt: skip
        (shard / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def raw(a: StateRef, b: StateRef, p: float, inc: float) -> dict:
    return {"a": as_dict(a), "b": as_dict(b), "p": p, "inc": inc}


def pair(item: dict) -> tuple[tuple[str, int], tuple[str, int]]:
    return ((item["a"]["episode"], item["a"]["index"]),
            (item["b"]["episode"], item["b"]["index"]))  # fmt: skip
