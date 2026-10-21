"""A run of `collect` of specs/preference-pool.md against the fakes of tests/clips (the fake
Azure server answering a ranged GET as [R] does, the fake ffmpeg, the fake classifier) on a
world of pool_world.py. Every file lives under tmp/repo, the repository root of the run
(`mcpref.REPO` is patched there by the fixture): the configs' paths are relative to it."""

import json
import os
import re
from pathlib import Path

import yaml
from clips.collect_gamma import Classifier
from clips.collect_site import GID, HASHED, Site
from fake_azure import request_key

from mcpref.clips.collect import load_config as load_perception
from mcpref.pool.run import collect, load_config

from .pool_expected import Expected, expected
from .pool_world import DEATH, REST

POOL = {  # PreferencePoolConfig in field order; perception and out_dir set by the Site
    "perception": None, "head_bytes": 1001, "head_workers": 3, "entries_per_shard": 3,
    "tick_ms": 50, "tick_slack": 0.03, "resetting_stats": [REST, DEATH], "episodes": 2,
    "per_episode": 2, "n_clips": 100, "partners": 1, "cycles": 0.5, "seed": 0,
    "workers": 1, "out_dir": None,
}  # fmt: skip
P_OUTPUT = ("index_urls", "min_version", "required_options", "width", "height", "clips",
            "screens", "gui_actions", "overlay", "states")  # fmt: skip
COUNTS = ("episodes", "sessions", "clips", "comparisons")
JSON_FILES = ("records.json", "state.json", "clip.json", "comparisons.json", "sessions.json")  # fmt: skip


def write_config(path: Path, **changes) -> Path:
    """A pool config file at `path` with POOL, its paths and `changes`, in field order."""
    doc = POOL | {"perception": "cfg/perception-data.yaml", "out_dir": "out"} | changes
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


class PoolSite(Site):
    """The Site of tests/clips with its root, configs and outputs under `root`; out_dir
    `root/out`; `content_ranges`: request key -> the Content-Range sent instead (None: no
    header)."""

    def __init__(self, http, root: Path, episodes: dict):
        super().__init__(http, root, root, episodes)
        self.out, self.fields["out_dir"] = root / "out", "perception-out"
        self.pool, self.pool_path = dict(POOL), root / "cfg" / "preference-pool.yaml"
        self.content_ranges = {}

    def app(self, req):
        """[R]: a ranged GET of an empty file answers 416."""
        if (
            req.method == "GET"
            and req.headers.get("Range")
            and self.data(req.path) == b""
        ):
            return 416, b""
        return super().app(req)

    def headers(self, handler) -> dict:
        out = super().headers(handler)
        key = request_key(handler.command, handler.path, handler.headers)
        if key in self.content_ranges:
            out.pop("Content-Range", None)
            if self.content_ranges[key] is not None:
                out["Content-Range"] = self.content_ranges[key]
        return out

    def pool_values(self, **changes) -> dict:
        paths = {"perception": self.rel(self.paths["perception-data"]), "out_dir": self.rel(self.out)}  # fmt: skip
        return self.pool | paths | changes

    def config(self, p: dict | None = None, **changes) -> Path:
        """The configs of perception-data with the changes `p`, the pool's with `changes`;
        the pool config's path."""
        super().config(**(p or {}))
        doc = yaml.safe_dump(self.pool_values(**changes), sort_keys=False)
        self.pool_path.write_text(doc)
        return self.pool_path

    def collect(self, path: Path, gamma: Classifier | None = None, gid: str = GID) -> Classifier:  # fmt: skip
        gamma, self.gid = gamma or Classifier(self.world), gid
        cfg = load_config(path)
        collect(cfg, load_perception(self.repo / cfg.perception), gamma, gid)
        return gamma

    def expected(self, p: dict | None = None, **changes) -> Expected:
        return expected(
            self.world, self.values(**(p or {})), self.pool_values(**changes)
        )

    def expected_run(self, ex: Expected | None = None) -> dict:
        """run.json of the configs last written: without counts, or with those of `ex`."""
        mine = yaml.safe_load(self.pool_path.read_text())
        theirs = yaml.safe_load(self.paths["perception-data"].read_text())
        config = {k: mine[k] for k in POOL} | {k: theirs[k] for k in P_OUTPUT}
        run = {"config": config, "sha256": {n: self.sha256(n) for n in HASHED},
               "classifier": self.gid}  # fmt: skip
        return run if ex is None else run | ex.counts()

    def tree(self, *parts: str) -> dict:
        return {p: json.loads(x) if p.rsplit("/", 1)[-1] in JSON_FILES else x
                for p, x in self.raw(*parts).items()}  # fmt: skip

    def url_stem(self, relpath: str) -> str:
        return self.host + self.path(relpath, "")

    def heads_asked(self, since: int = 0) -> list[str]:
        """The relpaths whose `.jsonl` got the GET of (1) (from request `since` on), in
        order; no part of a ranged fetch (f) of a `.jsonl` has the same Range."""
        head, q = f"bytes=0-{self.pool['head_bytes'] - 1}", self.fields["fetch_streams"]
        sizes = [
            len(x) for u, x in self.world.files.items() if u.endswith(".jsonl") and x
        ]
        assert all(min(-(-n // q), n) != self.pool["head_bytes"] for n in sizes)
        base = len(self.path("", ""))
        return [k[1][base : -len(".jsonl")] + ".mp4" for k in self.asked()[since:]
                if k[0] == "GET" and k[1].endswith(".jsonl") and k[2] == head]  # fmt: skip

    def worked(self, relpaths, since: int = 0) -> set:
        """The relpaths among `relpaths` whose files got a request (from request `since`
        on) other than the GET of (1)."""
        head = f"bytes=0-{self.pool['head_bytes'] - 1}"
        stems, out = {self.path(r, ""): r for r in relpaths}, set()
        for _, path, rng in self.asked()[since:]:
            stem = re.sub(r"(-options\.json|\.jsonl|\.mp4)$", "", path)
            if stem in stems and not (path.endswith(".jsonl") and rng == head):
                out.add(stems[stem])
        return out


def assert_outputs(site: PoolSite, ex: Expected) -> None:
    """out_dir holds run.json with the counts (keys in order), the indices as served, the
    head shards and shards of `ex`, nothing else."""
    files = site.tree()
    run = json.loads(files.pop("run.json"))
    assert run == site.expected_run(ex)
    assert list(run) == ["config", "sha256", "classifier", *COUNTS]
    assert list(run["config"]) == [*POOL, *P_OUTPUT]
    indices = {f"indices/{s}.json": site.index_body(s) for s in site.world.episodes}
    assert {p: files.pop(p) for p in list(files) if p.startswith("indices/")} == indices
    assert files == ex.tree()
    assert set(os.listdir(site.out)) <= {"indices", "run.json", "sessions", "shards"}
