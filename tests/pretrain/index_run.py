"""Runs of `main` of specs/train-index.md on the worlds of index_world.py and readers of
`{out_dir}`. The config's paths are relative to a temporary repository root: `mcpref.REPO`,
which the spec names as the root they are resolved against, is patched wherever a module of
`mcpref` holds it; the working directory is one from which they lead nowhere."""

import contextlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import tqdm
import yaml
from progress_bars import Bar, assert_aborted, tqdm_defaults

import mcpref
from mcpref.pretrain.index import main

from . import index_oracle as oracle
from .index_oracle import json_bytes, name, shard_files
from .index_world import (
    CONFIG,
    World,
    clips_run,
    n_clips,
    read,
    write_clips,
    write_labels,
)

PREFIX = "mcpref.pretrain.index: "
INPUTS = Bar("index inputs", 6, 6, "shard")  # HAND: 2 shards of C, Q_c and Q_l each


@dataclass
class Run:
    code: int
    err: str
    stdout: str


class Interrupted(RuntimeError):
    """A failure the tests raise inside a bar."""


def interrupting(desc: str, at: int, error=Interrupted):
    """A patch making tqdm's `update` raise `error` once the bar `desc` reaches `at`."""
    real = tqdm.tqdm.update

    def update(self, n=1):
        shown = real(self, n)
        if self.desc == desc and self.n >= at:
            raise error("interrupted by the test")
        return shown

    return lambda mp: mp.setattr(tqdm.tqdm, "update", update)


@contextlib.contextmanager
def repo_root(root: Path):
    """mcpref.REPO is `root` in every module of mcpref holding it, also in one imported
    meanwhile, restored afterwards."""
    original = mcpref.REPO

    def holders(value) -> list:
        mods = [m for n, m in list(sys.modules.items()) if n.split(".")[0] == "mcpref"]
        return [m for m in mods if getattr(m, "REPO", None) is value]

    patched = holders(original)
    for m in patched:
        m.REPO = root
    try:
        yield
    finally:
        for m in patched + holders(root):
            m.REPO = original


def records(directory: Path) -> list[bytes]:
    return shard_files(directory / "shards", "records.json")


@dataclass
class Site:
    """tmp/repo — the repository root of the runs; tmp/cwd — the working directory;
    tmp/train-index.yaml — the config."""

    tmp: Path
    world: World
    config: dict = field(default_factory=lambda: dict(CONFIG))

    def __post_init__(self) -> None:
        self.repo, self.cwd = self.tmp / "repo", self.tmp / "cwd"
        self.path = self.tmp / "train-index.yaml"
        self.cwd.mkdir(parents=True, exist_ok=True)

    def at(self, key: str) -> Path:
        """A config path resolved against the repository root."""
        return self.repo / self.config[key]

    @property
    def out(self) -> Path:
        return self.at("out_dir")

    @property
    def pairs_on(self) -> bool:
        return self.config["pair_clips_dir"] is not None

    def write_config(self, extra: str = "") -> None:
        text = yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True)
        self.path.write_text(text + extra)

    def write(self, c_run: dict | None = None, q_run: dict | None = None) -> None:
        """The config and the inputs; c_run, q_run: run.json of C and Q_c (default:
        complete, the same clip-defining fields, Q_c's with other values of other keys);
        Q_l's run.json labels this Q_c."""
        self.write_config()
        w = self.world
        write_clips(self.at("clips_dir"), w.clips, c_run or clips_run(n_clips(w.clips)))
        if self.pairs_on:
            q_run = q_run or clips_run(n_clips(w.pair_clips), {"config.n_clips": 7})
            write_clips(
                self.at("pair_clips_dir"), w.pair_clips, q_run | {"note": "Q_c"}
            )
            fp = self.fingerprints()["pair_clips"]
            labels = {"complete": True, "pair_clips": fp, "model": "fake"}
            write_labels(self.at("pair_labels_dir"), w.pairs, labels)

    def run(self, patch=None, argv=None) -> Run:
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

    def fingerprints(self) -> dict:
        """(4) of C, Q_c, Q_l (None without pairs) from the files on disk."""
        fp = {"clips": None, "pair_clips": None, "pair_labels": None}
        for key, d in [("clips", "clips_dir"), ("pair_clips", "pair_clips_dir")]:
            if self.config[d] is not None:
                run = read(self.at(d) / "run.json")
                fp[key] = oracle.clips_fingerprint(run, records(self.at(d)))
        if self.pairs_on:
            pairs = shard_files(self.at("pair_labels_dir") / "shards", "pairs.json")
            fp["pair_labels"] = oracle.labels_fingerprint(pairs)
        return fp

    def expected(self) -> oracle.Expected:
        pairs = self.world.pairs if self.pairs_on else None
        return oracle.expected(self.world.clips, pairs, self.config)

    def expected_run(self, complete: bool = True) -> bytes:
        """{out_dir}/run.json after (R') or, complete, after (W)."""
        c = read(self.at("clips_dir") / "run.json")
        value = oracle.run_json(self.path.read_bytes(), self.fingerprints(), c)
        if complete:
            value |= {"complete": True, "counts": self.expected().counts}
        return json_bytes(value)

    def expected_tree(self) -> dict:
        """{out_dir} after a complete run: path -> bytes."""
        tree = {"run.json": self.expected_run()}
        for part, shards in zip(oracle.PARTS, self.expected().shards):
            for s, items in enumerate(shards):
                tree[f"{part}/{name(s)}/items.json"] = oracle.items_bytes(items)
        return tree

    def tree(self) -> dict:
        """Every file under {out_dir}: path relative to it -> bytes."""
        if not self.out.exists():
            return {}
        files = sorted(p for p in self.out.rglob("*") if p.is_file())
        return {p.relative_to(self.out).as_posix(): p.read_bytes() for p in files}


def items(out: Path, part: str) -> list[list]:
    """The items of each complete shard of {out_dir}/<part>/, in shard order."""
    return [json.loads(b)["items"] for b in shard_files(out / part, "items.json")]


def aborted(r: Run, message: str) -> list:
    """Exit 1, `failed` of a RuntimeError as the last line; the lines above it."""
    above = assert_aborted(r.code, r.err)
    assert r.err.splitlines()[-1] == f"{PREFIX}RuntimeError: {message}", r.err
    return above
