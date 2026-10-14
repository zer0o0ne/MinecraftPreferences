"""Fake Azure server and fake ffmpeg for `main` of specs/vpt-gui-frames.md."""

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

import pytest
import yaml
from progress_bars import tqdm_defaults
from vpt_checks import EPISODES
from vpt_world import FIELDS, SUBSETS, make_world, reference_plan

FAKE_FFMPEG = Path(__file__).with_name("fake_ffmpeg.py")
HOLD = 3.0  # s a held reply waits; longer than every http_timeout of these tests
PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
              "all_proxy", "NO_PROXY", "no_proxy")  # fmt: skip
WRAPPER = '#!/bin/sh\nexec {python} -I -S {fake} {ctl} {log} "$0" "$@"\n'


class Site:
    """Server for `world` (indices with ETags, episode files, videos with byte ranges),
    a fake ffmpeg at `self.ffmpeg`, configs with out_dir relative to the repository root."""

    def __init__(self, http, tmp: Path, repo: Path, episodes: dict):
        self.tmp, self.out = tmp, tmp / "out"
        self.server = http(self.app, listen=False)
        self.host = f"http://127.0.0.1:{self.server.port}"
        self.world = make_world(episodes, host=self.host)
        self.etags = {s: f"0x8DA5AB90A1B2F{k:02X}" for k, s in enumerate(SUBSETS)}
        self.faults, self.redirects = {}, {}  # path -> first replies; path -> Location
        self.stalls = {}  # path -> s waited before the headers and again before the body
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
        site, base = self, self.server.server.RequestHandlerClass

        class Headed(base):
            def end_headers(self) -> None:
                for key, value in site.headers(self.path, self.headers).items():
                    self.send_header(key, value)
                super().end_headers()
                time.sleep(site.stalls.get(self.path, 0))

        self.server.server.RequestHandlerClass = Headed

    def data(self, path: str) -> bytes | None:
        return self.world.fetch(self.host + path)

    def span(self, path: str, headers) -> tuple[int, int] | None:
        """Byte range asked for a video: inclusive (first, last)."""
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", headers.get("Range") or "")
        if not path.endswith(".mp4") or not match:
            return None
        last = len(self.data(path)) - 1
        return int(match[1]), min(int(match[2] or last), last)

    def app(self, req):
        n = sum(r.path == req.path for r in self.server.requests) - 1
        faults = self.faults.get(req.path, [])
        if n < len(faults) and faults[n] == "hold":
            req.hold(HOLD)
        elif n < len(faults):
            return None if faults[n] == "drop" else (faults[n], b"")
        req.hold(self.stalls.get(req.path, 0))
        data, span = self.data(req.path), self.span(req.path, req.headers)
        if data is None:
            return 404, b""
        return (206, data[span[0] : span[1] + 1]) if span else (200, data)

    def headers(self, path: str, request_headers) -> dict:
        subset = re.fullmatch(r"/idx/(.+)\.json", path)
        out = {"Location": self.redirects[path]} if path in self.redirects else {}
        if subset and self.etags.get(subset[1]) is not None:
            out["ETag"] = self.etags[subset[1]]
        if path.endswith(".mp4") and self.data(path) is not None:
            out["Accept-Ranges"] = "bytes"
            if span := self.span(path, request_headers):
                out["Content-Range"] = (
                    f"bytes {span[0]}-{span[1]}/{len(self.data(path))}"
                )
        return out

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

    def expected(self, **changes) -> tuple[list, list]:
        """frames.jsonl rows (status "ok") and episodes.jsonl rows of (3)."""
        f = self.fields | changes
        frames, records = reference_plan(
            self.world, f["n_frames"], f["per_episode"], f["seed"]
        )
        rows = [{k: v for k, v in p.items() if k != "lines"} for p in frames]
        rows = [r | {"prev_line": p["lines"][0], "line": p["lines"][1], "status": "ok"}
                for r, p in zip(rows, frames)]  # fmt: skip
        keys = ("subset", "relpath", "reason", "options")
        return rows, [dict(zip(keys, r)) for r in records]

    def read_jsonl(self, name: str) -> list[dict]:
        return [json.loads(line) for line in (self.out / name).read_text().splitlines()]

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
