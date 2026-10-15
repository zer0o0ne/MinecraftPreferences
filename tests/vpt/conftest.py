"""Fake Azure server and fake ffmpeg for `main` of specs/vpt-gui-frames.md."""

import json
import os
import shlex
import sys
from pathlib import Path

import pytest
import yaml
from fake_azure import FakeAzure
from progress_bars import assert_aborted, tqdm_defaults
from vpt_checks import EPISODES, read_shards
from vpt_world import (
    FIELDS,
    SUBSETS,
    joined,
    make_world,
    reference_draws,
    reference_shards,
)

from mcpref.vpt.gui_frames import main

FAKE_FFMPEG = Path(__file__).with_name("fake_ffmpeg.py")
PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
              "all_proxy", "NO_PROXY", "no_proxy")  # fmt: skip
WRAPPER = '#!/bin/sh\nexec {python} -I -S {fake} {ctl} {log} "$0" "$@"\n'


class Site(FakeAzure):
    """Fake Azure server (fake_azure.py) of `world`, a fake ffmpeg at `self.ffmpeg`, configs
    with out_dir relative to the repository root. The server sends ETags, which (a) and (d)
    ignore."""

    def __init__(self, http, tmp: Path, repo: Path, episodes: dict):
        super().__init__(http)
        self.tmp, self.out = tmp, tmp / "out"
        self.world = make_world(episodes, host=self.host)
        self.etags = {s: f"0x8DA5AB90A1B2F{k:02X}" for k, s in enumerate(SUBSETS)}
        self.behaviours, self.sleep = {}, 3.0  # fake ffmpeg, see fake_ffmpeg.py
        self.noise = {}  # fake ffmpeg key -> bytes written to stderr, one per attempt
        self.ctl, self.log, self.ffmpeg = (
            tmp / "ctl.json",
            tmp / "ffmpeg.log",
            tmp / "ffmpeg",
        )
        quoted = {
            "python": sys.executable,
            "fake": FAKE_FFMPEG,
            "ctl": self.ctl,
            "log": self.log,
        }
        self.ffmpeg.write_text(
            WRAPPER.format(**{k: shlex.quote(str(v)) for k, v in quoted.items()})
        )
        self.ffmpeg.chmod(0o755)
        self.fields = FIELDS | {"index_urls": self.world.index_urls, "ffmpeg": str(self.ffmpeg),
                                "out_dir": os.path.relpath(self.out, repo)}  # fmt: skip

    def data(self, path: str) -> bytes | None:
        return self.world.fetch(self.host + path)

    def path(self, subset: str, relpath: str, suffix: str) -> str:
        url = self.world.url(subset, self.world.eps[subset, relpath], suffix)
        return url.removeprefix(self.host)

    def first_fetched(self, suffix: str) -> str:
        """Path of `suffix` for the first drawn episode that is downloaded."""
        first = next(r for r in self.expected()[1] if r["reason"] != "version")
        return self.path(first["subset"], first["relpath"], suffix)

    def config(self, **changes) -> str:
        """Writes the config and the fake ffmpeg's control file; returns the config path."""
        fields = self.fields | changes
        frame_size = fields["width"] * fields["height"] * 3 // 2
        noise = {k: [b.hex() for b in v] for k, v in self.noise.items()}
        ctl = {"fps": fields["fps"], "frame_size": frame_size, "sleep": self.sleep,
               "behaviours": self.behaviours, "stderr": noise}  # fmt: skip
        self.ctl.write_text(json.dumps(ctl))
        path = self.tmp / "config.yaml"
        path.write_text(yaml.safe_dump(fields, sort_keys=False))
        return str(path)

    def draws(self, **changes) -> list[dict]:
        """The draws of (3) for the fields with `changes`."""
        f = self.fields | changes
        return reference_draws(self.world, f["n_frames"], f["per_episode"], f["seed"])

    def expected(self, **changes) -> tuple[list, list]:
        """Frames and episode records of the plan (3), in draw order."""
        draws = self.draws(**changes)
        return joined(draws, "frames"), joined(draws, "episodes")

    def shards(self, before: tuple = (), **changes) -> list[dict]:
        """Plan shards (3') after the interrupted runs `before` ((h, K) each)."""
        k = (self.fields | changes)["plan_shard_draws"]
        return reference_shards(self.draws(**changes), k, before)

    def plan(self) -> list[dict]:
        """The complete shards of out_dir/plan."""
        return read_shards(self.out / "plan")

    def planned(self) -> list[dict]:
        """The frames of out_dir/plan in shard order."""
        return joined(self.plan(), "frames")

    def stop_in_plan(self, capfd, **changes) -> None:
        """`main run` aborted in the draw that starts shard 1: the HEAD of either file of its
        first downloaded episode fails (no retries). Shard 0 stays, as (3) cuts it."""
        k = (self.fields | changes)["plan_shard_draws"]
        draw = self.draws(**changes)[k]
        first = next(r for r in draw["episodes"] if r["reason"] != "version")
        for suffix in ("-options.json", ".jsonl"):
            path = self.path(first["subset"], first["relpath"], suffix)
            self.faults[self.head(path)] = [503]
        code = main(["run", self.config(**(changes | {"http_retries": 0}))])
        assert_aborted(code, capfd.readouterr().err)
        self.faults.clear()
        assert self.plan() == self.shards(**changes)[:1]

    def snapshot(self) -> dict:
        """Every file under out_dir: path -> bytes."""
        return {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}

    def run_json(self) -> dict:
        return json.loads((self.out / "run.json").read_text())

    def files(self) -> dict:
        """Every file under out_dir/frames: name -> bytes."""
        root = self.out / "frames"
        return {p.name: p.read_bytes() for p in root.iterdir()} if root.exists() else {}

    def calls(self, event: str = "start") -> list[dict]:
        """Fake ffmpeg log records, oldest first."""
        if not self.log.exists():
            return []
        records = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [r for r in records if r["event"] == event]


@pytest.fixture
def make_site(http, tmp_path, repo, monkeypatch):
    """Site factory, listening unless `listen=False`. The cwd is chosen so that out_dir
    resolved against it lands elsewhere; proxy and TQDM_* variables are unset."""
    rel = Path(os.path.relpath(tmp_path / "out", repo))
    cwd = tmp_path.joinpath(*["cwd"] * (rel.parts.count("..") + 1))
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    tqdm_defaults(monkeypatch)

    def make(listen: bool = True, episodes: dict = EPISODES) -> Site:
        site = Site(http, tmp_path, repo, episodes)
        if listen:
            site.server.listen()
        return site

    return make


@pytest.fixture
def site(make_site) -> Site:
    return make_site()
