"""A run of `collect` (specs/perception-data.md) against fakes: the fake Azure server of
tests/vpt/fake_azure.py serving a world of collect_world.py under one basedir (Domain), a
fake ffmpeg (fake_decoder.py), a fake classifier, and the config files of the block (paths
relative to the repository root, a hand-made font with one glyph for the overlay prefix)."""

import hashlib
import json
import os
import re
import shlex
import sys
import threading
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "vpt"))
from fake_azure import FakeAzure
from vpt_world import SUBSETS, World

from mcpref.clips.collect import collect, load_config

from .collect_expected import Expected, expected
from .collect_gamma import Classifier
from .collect_out import OutDir
from .collect_world import CATALOGUE, FRAME, REQUIRED, TEXTURES, H, W, font

WRAPPER = '#!/bin/sh\nexec {python} -I -S {fake} {ctl} {log} "$0" "$@"\n'
FAKE = Path(__file__).with_name("fake_decoder.py")
FIELDS = {  # PerceptionDataConfig fields but index_urls, ffmpeg, the config paths, out_dir
    "x_ms_version": "2019-12-12", "min_version": "6.12", "required_options": REQUIRED,
    "width": W, "height": H, "fetch_streams": 3, "http_timeout": 2.0, "http_retries": 1,
    "retry_delay": 0.05, "ffmpeg_timeout": 3.0, "states": 3, "per_episode": 2,
    "n_clips": 100, "seed": 0, "workers": 1, "classifier_batch": 2,
}  # fmt: skip
SUB = {  # the values of the other configs
    "delta": 2, "turn_deg": 4, "r_min": 3, "double_click_ms": 250, "prefix": "A",
    "text_luma": {"normal": 200, "dimmed": 60}, "tolerance": {"normal": 10, "dimmed": 0},
    "box_max": {"normal": 250, "dimmed": 0}, "threshold": 0.75,
}  # fmt: skip
NAMES = ("perception-data", "clips", "actions", "screens", "gui_actions", "overlay",
         "overlay_screens")  # fmt: skip
HASHED = ("actions", "clips", "screens", "gui_actions", "overlay", "overlay_screens")
GID = "fake-gamma-1"


def index_body(world: World, subset: str) -> bytes:
    """A served index: indented, an extra field, raw UTF-8; a re-serialised copy differs."""
    doc = {"basedir": world.basedirs[subset], "note": "é",
           "relpaths": [e.relpath for e in world.episodes[subset]]}  # fmt: skip
    return json.dumps(doc, indent=3, ensure_ascii=False).encode()


class Site(FakeAzure, OutDir):
    """The fake Azure server of `episodes`, a fake ffmpeg at `self.ffmpeg`, the configs
    under tmp/cfg; `behaviours`, `noise` (stem -> per decode attempt) drive the ffmpeg;
    `indices` (subset -> body) replaces a served index; `gates` (URL path -> Event) hold
    each request of a path until the event is set (at most 5 s)."""

    def __init__(self, http, tmp: Path, repo: Path, episodes: dict):
        super().__init__(http)
        self.tmp, self.repo, self.out = tmp, repo, tmp / "out"
        full = {s: episodes.get(s, []) for s in SUBSETS}
        urls = {s: f"{self.host}/idx/{s}.json" for s in SUBSETS}
        self.world = World(full, urls, dict.fromkeys(SUBSETS, f"{self.host}/b/"))
        self.etags = {s: f"0x8DA5AB90A1B2F{k:02X}" for k, s in enumerate(SUBSETS)}
        self.indices, self.gates = {}, {}
        self.behaviours, self.noise, self.sleep = {}, {}, 6.0
        self.ctl, self.log, self.ffmpeg = tmp / "ctl.json", tmp / "ffmpeg.log", tmp / "ffmpeg"  # fmt: skip
        quoted = {"python": sys.executable, "fake": FAKE, "ctl": self.ctl, "log": self.log}  # fmt: skip
        self.ffmpeg.write_text(WRAPPER.format(**{k: shlex.quote(str(v)) for k, v in quoted.items()}))  # fmt: skip
        self.ffmpeg.chmod(0o755)
        self.paths = {n: tmp / "cfg" / f"{n}.yaml" for n in NAMES}
        self.fields = FIELDS | {"index_urls": urls, "ffmpeg": str(self.ffmpeg),
                                "out_dir": self.rel(self.out)}  # fmt: skip
        self.fields |= {n: self.rel(self.paths[n]) for n in ("clips", "screens", "gui_actions", "overlay")}  # fmt: skip
        self.sub, self.gid = dict(SUB), GID

    def app(self, req):
        if req.path in self.gates:
            self.gates[req.path].wait(5.0)
        return super().app(req)

    def rel(self, path: Path) -> str:
        return os.path.relpath(path, self.repo)

    def data(self, path: str) -> bytes | None:
        subset = re.fullmatch(r"/idx/(.+)\.json", path)
        if subset:
            return self.indices.get(subset[1], self.index_body(subset[1]))
        return self.world.fetch(self.host + path)

    def index_body(self, subset: str) -> bytes:
        return index_body(self.world, subset)

    def path(self, relpath: str, suffix: str) -> str:
        """URL path of the file `suffix` of an episode."""
        return self.world.basedirs[SUBSETS[0]].removeprefix(self.host) + relpath[:-4] + suffix  # fmt: skip

    def values(self, **changes) -> dict:
        """Every config value: the fields of PerceptionDataConfig and SUB."""
        return self.fields | self.sub | changes

    def config(self, **changes) -> Path:
        """Writes the configs and the fake ffmpeg's control file; returns the config path.
        Only the overlay's catalogue has the font under its textures directory."""
        v, p = self.values(**changes), self.paths
        font(self.tmp / "tex/font/ascii.png")
        docs = {
            "perception-data": {k: v[k] for k in self.fields},
            "clips": {"delta": v["delta"], "actions": self.rel(p["actions"])},
            "actions": {"turn_deg": v["turn_deg"], "r_min": v["r_min"]},
            "screens": {"classes": CATALOGUE, "textures": TEXTURES | {"out_dir": self.rel(self.tmp / "no-tex")}},
            "gui_actions": {"double_click_ms": v["double_click_ms"]},
            "overlay": {"screens": self.rel(p["overlay_screens"])}
            | {k: v[k] for k in ("prefix", "text_luma", "tolerance", "box_max", "threshold")},
            "overlay_screens": {"classes": CATALOGUE, "textures": TEXTURES | {"out_dir": self.rel(self.tmp / "tex")}},
        }  # fmt: skip
        p["clips"].parent.mkdir(exist_ok=True)
        for name, doc in docs.items():
            p[name].write_text(yaml.safe_dump(doc, sort_keys=False))
        noise = {k: [b.hex() for b in x] for k, x in self.noise.items()}
        ctl = {"frame_size": FRAME, "sleep": self.sleep, "behaviours": self.behaviours, "stderr": noise}  # fmt: skip
        self.ctl.write_text(json.dumps(ctl))
        return p["perception-data"]

    def expected(self, broken=frozenset(), **changes) -> Expected:
        return expected(self.world, self.values(**changes), broken)

    def failing_gamma(self, k: int | None = None) -> Classifier:
        """A classifier raising at each call for the candidate of draw k (default: the
        third draw whose candidate calls it)."""
        ex = self.expected()
        stems = [s.records[-1]["relpath"].split("/")[2][:-4] for s in ex.shards]
        calling = [j for j, x in enumerate(stems) if ex.calls.get(x)]
        return Classifier(self.world, fail_stem=stems[calling[2] if k is None else k])

    def run(
        self, gamma: Classifier | None = None, gid: str = GID, **changes
    ) -> Classifier:
        """`collect` on the config with `changes`; returns the classifier."""
        return self.collect(self.config(**changes), gamma, gid)

    def collect(
        self, path: Path, gamma: Classifier | None = None, gid: str = GID
    ) -> Classifier:
        gamma, self.gid = gamma or Classifier(self.world), gid
        collect(load_config(path), gamma, gid)
        return gamma

    def aborted(self, gamma=None, path=None, gid: str = GID, **changes) -> Classifier:
        """A run that aborts: `collect` raises (the exception kept in `self.error`)."""
        gamma = gamma or Classifier(self.world)
        with pytest.raises(Exception) as info:
            self.collect(path or self.config(**changes), gamma, gid)
        self.error = info.value
        return gamma

    def in_thread(self, gamma: Classifier, **changes) -> threading.Thread:
        """`collect` started in a thread (its exception, if any, in `self.error`)."""
        path, self.error = self.config(**changes), None

        def target():
            try:
                self.collect(path, gamma)
            except Exception as error:  # noqa: BLE001 - kept for the test's thread
                self.error = error

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def decodes(self, event: str = "start") -> list[dict]:
        """Fake ffmpeg log records, oldest first."""
        text = self.log.read_text() if self.log.exists() else ""
        return [r for r in map(json.loads, text.splitlines()) if r["event"] == event]

    def sha256(self, name: str) -> str:
        return hashlib.sha256(self.paths[name].read_bytes()).hexdigest()

    def expected_run(self, ex: Expected | None = None) -> dict:
        """run.json of the last config written: without counts, or with those of `ex`."""
        config = yaml.safe_load(self.paths["perception-data"].read_text())
        run = {"config": config, "sha256": {n: self.sha256(n) for n in HASHED},
               "classifier": self.gid}  # fmt: skip
        if ex is None:
            return run
        clips = ex.shards[-1].clips if ex.shards else 0
        return run | {"episodes": ex.counts(), "clips": clips}


def asked_files(site: Site, relpaths, since: int = 0) -> list[tuple]:
    """The requests received (from request `since` on) for the files of `relpaths`."""
    stems = {site.path(r, "") for r in relpaths}
    keys = site.asked()[since:]
    return [k for k in keys if re.sub(r"(-options\.json|\.jsonl|\.mp4)$", "", k[1]) in stems]  # fmt: skip
