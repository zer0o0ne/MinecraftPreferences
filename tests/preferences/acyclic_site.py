"""Runs of `main` of specs/acyclic.md on the worlds of acyclic_world.py and readers of
`{out_dir}`. The config's paths are relative to a temporary repository root (`mcpref.REPO`
patched as in tests/pretrain/index_run.py); the working directory is one from which they
lead nowhere."""

import contextlib
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from graphs.feedback_oracle import eades
from pretrain.index_run import repo_root
from progress_bars import assert_aborted, tqdm_defaults

from mcpref.preferences.acyclic import main

from . import acyclic_oracle as oracle
from .acyclic_world import DYADIC, HAND, POOL, name, votes_run, write_votes

PREFIX = "mcpref.preferences.acyclic: "
CONFIG = {  # AcyclicConfig in field order
    "votes_dir": "w/votes", "out_dir": "w/labels", "weights": DYADIC, "refusal_share": 0.5,
}  # fmt: skip


@dataclass
class Run:
    code: int
    err: str
    stdout: str


def shard_files(directory: Path, file: str) -> list[bytes]:
    """The bytes of `file` in the complete shards 000000, 000001, ... of `directory`."""
    n = 0
    while (directory / name(n)).is_dir():
        n += 1
    return [(directory / name(k) / file).read_bytes() for k in range(n)]


@dataclass
class Site:
    """tmp/repo — the repository root of the runs; tmp/cwd — the working directory;
    tmp/acyclic.yaml — the config."""

    tmp: Path
    world: list
    config: dict = field(default_factory=lambda: dict(CONFIG))

    def __post_init__(self) -> None:
        self.repo, self.cwd = self.tmp / "repo", self.tmp / "cwd"
        self.path = self.tmp / "acyclic.yaml"
        self.cwd.mkdir(parents=True, exist_ok=True)

    @property
    def votes(self) -> Path:
        return self.repo / self.config["votes_dir"]

    @property
    def out(self) -> Path:
        return self.repo / self.config["out_dir"]

    def write_config(self, extra: str = "") -> None:
        self.path.write_text(yaml.safe_dump(self.config, sort_keys=False) + extra)

    def write(self, run: dict | None) -> None:
        """The config and V_l with `run` as its run.json (None: absent)."""
        self.write_config()
        write_votes(self.votes, self.world, run)

    def run(self, patch: Callable | None = None, argv: list | None = None) -> Run:
        """main on the config from the working directory, tqdm with its defaults."""
        err, out = io.StringIO(), io.StringIO()
        with pytest.MonkeyPatch.context() as mp, repo_root(self.repo):
            tqdm_defaults(mp)
            mp.chdir(self.cwd)
            if patch:
                patch(mp)
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = main([str(self.path)] if argv is None else argv)
        return Run(code, err.getvalue(), out.getvalue())

    def answers(self) -> list[bytes]:
        return shard_files(self.votes / "shards", "answers.json")

    def expected(self, f0: Callable = eades) -> list[tuple]:
        weights, tau = self.config["weights"], self.config["refusal_share"]
        return oracle.shards(self.world, weights, tau, f0)

    def expected_tree(self, f0: Callable = eades, pool: str = POOL) -> dict:
        config = self.path.read_bytes()
        return oracle.tree(self.expected(f0), config, self.answers(), pool)

    def expected_run(self, pool: str = POOL) -> bytes:
        """{out_dir}/run.json after (I)."""
        return oracle.run_json(self.path.read_bytes(), self.answers(), pool)

    def tree(self) -> dict:
        """Every file under {out_dir}: path relative to it -> bytes."""
        if not self.out.exists():
            return {}
        files = sorted(p for p in self.out.rglob("*") if p.is_file())
        return {p.relative_to(self.out).as_posix(): p.read_bytes() for p in files}


DEFAULT = object()


def make_site(
    tmp: Path, world: list = HAND, write: bool = True, run=DEFAULT, **changes
) -> Site:
    """A Site on `world`, the config with `changes`, written unless `write=False`; `run`:
    {V_l}/run.json (default: complete, labelling Q_c `POOL`; None: absent)."""
    site = Site(tmp, world, {**CONFIG, **changes})
    if write:
        site.write(votes_run(world) if run is DEFAULT else run)
    return site


def aborted(r: Run, message: str) -> list:
    """Exit 1, `failed` of a RuntimeError as the last line; the lines above it."""
    above = assert_aborted(r.code, r.err)
    assert r.err.splitlines()[-1] == f"{PREFIX}RuntimeError: {message}", r.err
    return above
