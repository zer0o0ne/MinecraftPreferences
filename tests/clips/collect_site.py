"""A run of `collect` (specs/perception-data.md) against fakes: the fake Azure server of
tests/vpt/fake_azure.py serving a world of collect_world.py, a fake ffmpeg
(fake_decoder.py), a fake classifier, and the config files of the block (paths relative to
the repository root, a hand-made font with one glyph for the overlay prefix `A`)."""

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "vpt"))
from fake_azure import FakeAzure
from vpt_world import SUBSETS, make_world

from mcpref.clips.collect import collect, load_config

from .collect_expected import Expected, expected
from .collect_gamma import Classifier
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


class Site(FakeAzure):
    """The fake Azure server of `episodes`, a fake ffmpeg at `self.ffmpeg`, the configs
    under tmp/cfg; `behaviours`, `noise` (stem -> per decode attempt) drive the ffmpeg."""

    def __init__(self, http, tmp: Path, repo: Path, episodes: dict):
        super().__init__(http)
        self.tmp, self.repo, self.out = tmp, repo, tmp / "out"
        self.world = make_world(episodes, host=self.host)
        self.etags = {s: f"0x8DA5AB90A1B2F{k:02X}" for k, s in enumerate(SUBSETS)}
        self.behaviours, self.noise, self.sleep = {}, {}, 6.0
        self.ctl, self.log, self.ffmpeg = (
            tmp / "ctl.json",
            tmp / "ffmpeg.log",
            tmp / "ffmpeg",
        )
        quoted = {
            "python": sys.executable,
            "fake": FAKE,
            "ctl": self.ctl,
            "log": self.log,
        }
        self.ffmpeg.write_text(WRAPPER.format(**{k: shlex.quote(str(v)) for k, v in quoted.items()}))  # fmt: skip
        self.ffmpeg.chmod(0o755)
        self.paths = {n: tmp / "cfg" / f"{n}.yaml" for n in NAMES}
        self.fields = FIELDS | {"index_urls": self.world.index_urls, "ffmpeg": str(self.ffmpeg),
                                "out_dir": self.rel(self.out)}  # fmt: skip
        self.fields |= {
            n: self.rel(self.paths[n]) for n in NAMES[1:6] if n != "actions"
        }
        self.sub = dict(SUB)

    def rel(self, path: Path) -> str:
        return os.path.relpath(path, self.repo)

    def data(self, path: str) -> bytes | None:
        return self.world.fetch(self.host + path)

    def path(self, subset: str, relpath: str, suffix: str) -> str:
        """URL path of the file `suffix` of an episode."""
        url = self.world.url(subset, self.world.eps[subset, relpath], suffix)
        return url.removeprefix(self.host)

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

    def cut(self) -> int:
        """An N inside the first episode with two clips (N + 1: at its end)."""
        counts = [len(r["clips"] or []) for r in self.expected().records]
        return sum(counts[: counts.index(2)]) + 1

    def failing_gamma(self) -> Classifier:
        """A classifier raising at its first call for the third episode that calls it."""
        calls = [c for c in self.expected().calls.values() if c]
        return Classifier(self.world, fail_at=len(calls[0]) + len(calls[1]))

    def run(
        self, gamma: Classifier | None = None, gid: str = "fake-gamma-1", **changes
    ) -> Classifier:
        """`collect` on the config with `changes`; returns the classifier."""
        return self.collect(self.config(**changes), gamma, gid)

    def collect(
        self, path: Path, gamma: Classifier | None = None, gid: str = "fake-gamma-1"
    ) -> Classifier:
        gamma = gamma or Classifier(self.world)
        collect(load_config(path), gamma, gid)
        return gamma

    def aborted(
        self,
        gamma: Classifier | None = None,
        path: Path | None = None,
        gid: str = "fake-gamma-1",
        **changes,
    ) -> Classifier:
        """A run that aborts by (8): `collect` re-raises the exception (kept in
        `self.error`)."""
        gamma = gamma or Classifier(self.world)
        with pytest.raises(Exception) as info:
            self.collect(path or self.config(**changes), gamma, gid)
        self.error = info.value
        return gamma

    def completed(self) -> bool:
        """Whether run.json has the counts written at the end of a run that completes."""
        return {"episodes", "clips"} <= set(self.run_json())

    def read_jsonl(self, name: str) -> list[dict]:
        return [json.loads(x) for x in (self.out / name).read_text().splitlines()]

    def run_json(self) -> dict:
        return json.loads((self.out / "run.json").read_text())

    def tree(self, *parts: str) -> dict:
        """Every file under out_dir/<parts>: path relative to out_dir -> bytes."""
        root = self.out.joinpath(*parts)
        files = (
            sorted(x for x in root.rglob("*") if x.is_file()) if root.exists() else []
        )
        return {x.relative_to(self.out).as_posix(): x.read_bytes() for x in files}

    def decodes(self, event: str = "start") -> list[dict]:
        """Fake ffmpeg log records, oldest first."""
        text = self.log.read_text() if self.log.exists() else ""
        return [r for r in map(json.loads, text.splitlines()) if r["event"] == event]

    def sha256(self, name: str) -> str:
        return hashlib.sha256(self.paths[name].read_bytes()).hexdigest()


def assert_outputs(site: Site, ex: Expected) -> None:
    """The run's outputs equal `ex`; out_dir holds nothing else but run.json and an empty
    staging."""
    assert site.read_jsonl("episodes.jsonl") == ex.records
    assert site.read_jsonl("clips.jsonl") == ex.clips
    produced = site.tree()
    for name in ("run.json", "episodes.jsonl", "clips.jsonl"):
        produced.pop(name)
    assert produced == ex.files
    staging = site.out / "staging"
    assert not staging.exists() or list(staging.iterdir()) == []
