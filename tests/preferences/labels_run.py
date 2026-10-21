"""Runs of `main` of specs/vlm-labels.md on the worlds of labels_world.py with a fake
`ask_many`; the expected calls and `{out_dir}` in labels_oracle.py. The config's paths are relative to a temporary
repository root (`mcpref.REPO` patched as in tests/pretrain/index_run.py); the working
directory is one from which they lead nowhere."""

import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from pretrain.index_run import repo_root
from progress_bars import tqdm_defaults

import mcpref.vlm.client
from mcpref.preferences.labels import main

from .config_files import TINY, write_config
from .labels_oracle import Expected
from .labels_world import ANSWERS, HAND, pool_run, write_pool

PREFIX = "mcpref.preferences.labels: "


@dataclass
class Run:
    code: int
    err: str
    stdout: str


@dataclass
class Call:
    prompts: list
    seeds: list
    cfg: object
    run_json: bytes | None  # {out_dir}/run.json when called
    shards: list  # entries of {out_dir}/shards when called


class FakeAskMany:
    """ask_many: the answer of position p = seed - `seed0` is ANSWERS[p % 20][0]; call
    `fail_on` (counted over runs) raises `error`; records the calls."""

    def __init__(self, out: Path, seed0: int):
        self.out, self.seed0, self.calls = out, seed0, []
        self.fail_on, self.error = None, RuntimeError

    def __call__(self, prompts, seeds, cfg) -> list:
        run, shards = self.out / "run.json", self.out / "shards"
        listed = sorted(p.name for p in shards.iterdir()) if shards.exists() else []
        text = run.read_bytes() if run.exists() else None
        made = Call([list(q) for q in prompts], list(seeds), cfg, text, listed)
        self.calls.append(made)
        if len(self.calls) == self.fail_on:
            raise self.error("HTTP 400: rejected")
        return [ANSWERS[(s - self.seed0) % len(ANSWERS)][0] for s in seeds]


@dataclass
class Site(Expected):
    """tmp/repo — the repository root; tmp/cwd — the working directory; tmp/vlm-labels.yaml
    — the config; repo/w/vlm.yaml — a copy of configs/vlm.yaml whose server refuses."""

    tmp: Path
    shards: list = field(default_factory=lambda: HAND)
    config: dict = field(default_factory=lambda: dict(TINY))

    def __post_init__(self) -> None:
        self.repo, self.cwd = self.tmp / "repo", self.tmp / "cwd"
        self.path = self.tmp / "vlm-labels.yaml"
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.fake = FakeAskMany(self.out, self.config["seed"])

    def at(self, key: str) -> Path:
        return self.repo / self.config[key]

    @property
    def out(self) -> Path:
        return self.at("out_dir")

    def write(self, run: dict | None = None) -> None:
        write_config(self.path, self.config)
        vlm = yaml.safe_load(
            (Path(__file__).parents[2] / "configs/vlm.yaml").read_text()
        )
        vlm["client"].update(base_url="http://127.0.0.1:9", max_resends=0)
        self.at("vlm").parent.mkdir(parents=True, exist_ok=True)
        self.at("vlm").write_text(yaml.safe_dump(vlm))
        write_pool(self.at("pool_dir"), self.shards, run or pool_run(self.shards))

    def run(self, patch=None, argv=None) -> Run:
        """main on the config from the working directory; tqdm with its defaults; the fake
        in place of ask_many in every mcpref module holding it."""
        err, out = io.StringIO(), io.StringIO()
        real = mcpref.vlm.client.ask_many
        with pytest.MonkeyPatch.context() as mp, repo_root(self.repo):
            tqdm_defaults(mp)
            mp.chdir(self.cwd)
            for module in [m for n, m in sys.modules.items() if n.startswith("mcpref")]:
                if getattr(module, "ask_many", None) is real:
                    mp.setattr(module, "ask_many", self.fake)
            if patch:
                patch(mp)
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = main([str(self.path)] if argv is None else argv)
        return Run(code, err.getvalue(), out.getvalue())

    def vlm_config(self):
        return mcpref.vlm.client.load_config(self.at("vlm"))[0]

    def tree(self) -> dict:
        """Every file under {out_dir}: path relative to it -> bytes."""
        if not self.out.exists():
            return {}
        files = sorted(p for p in self.out.rglob("*") if p.is_file())
        return {p.relative_to(self.out).as_posix(): p.read_bytes() for p in files}
