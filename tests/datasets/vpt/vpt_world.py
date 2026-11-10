"""Fake VPT data for specs/vpt-gui-frames.md: episodes, indices, a fake `fetch` and
transcriptions of (1), (3) and (3') as oracles."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

SUBSETS = ("6xx", "7xx", "8xx", "9xx", "10xx")
REQUIRED = {"guiScale": 2, "fullscreen": False, "framebufferWidth": 1280,
            "framebufferHeight": 720, "lang": "en_us", "forceUnicodeFont": False,
            "resourcePacks": "[]"}  # fmt: skip
DIFFERENT = {"guiScale": 3, "fullscreen": True, "framebufferWidth": 1920,
             "framebufferHeight": 1080, "lang": "en_gb", "forceUnicodeFont": True,
             "resourcePacks": '["vanilla"]'}  # fmt: skip
OPTIONS = REQUIRED | {"fov": 0.0, "gamma": 1.0, "renderDistance": 8}
GOOD = "good"  # the well-formed file
BAD_JSON = b'{"guiScale": 2, "lang": "en_'
WIDTH, HEIGHT = 4, 2
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2
FIELDS = {  # GuiFramesConfig fields except index_urls, ffmpeg and out_dir
    "x_ms_version": "2019-12-12", "min_version": "6.12", "required_options": REQUIRED,
    "width": WIDTH, "height": HEIGHT, "fps": 20, "n_frames": 6, "per_episode": 2,
    "seed": 0, "workers": 3, "fetch_streams": 4, "plan_shard_draws": 2,
    "http_timeout": 2.0, "http_retries": 2, "retry_delay": 0.1,
    "ffmpeg_rw_timeout": 7, "ffmpeg_timeout": 5.0,
    "verify_relpath": "data/10.0/verify-me.mp4", "verify_frames": [0, 1, 19, 20, 23],
}  # fmt: skip


def lines(gui: str) -> list[dict]:
    """One line per char of `gui`: '.' no GUI, '#' a container, 'i' the player inventory."""
    return [
        {
            "isGuiOpen": c != ".",
            "isGuiInventory": c == "i",
            "tick": k,
            "mouse": {"x": k},
        }
        for k, c in enumerate(gui)
    ]


def jsonl(gui: str) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines(gui)).encode()


BAD_JSONL = jsonl("##") + b'{"isGuiOpen": tr\n'


def options(**changes) -> bytes:
    return json.dumps(OPTIONS | changes).encode()


def eligible(gui: str) -> list[int]:
    """(1), transcribed: 1 <= i <= n-1, lines i-1 and i both have isGuiOpen."""
    return [i for i in range(1, len(gui)) if gui[i] != "." and gui[i - 1] != "."]


def frame(stem: str, i: int) -> bytes:
    """Raw frame i of the fake video of `stem`."""
    return hashlib.sha256(f"{stem}/{i}".encode()).digest()[:FRAME_SIZE]


@dataclass(frozen=True)
class Ep:
    """An episode; `options`, `jsonl`: served bytes, None (404) or GOOD; `reason` expected."""

    relpath: str
    gui: str = ".##.###."
    reason: str | None = None
    options: object = GOOD
    jsonl: object = GOOD

    @property
    def version(self) -> str:
        return self.relpath.split("/")[1]

    @property
    def stem(self) -> str:
        return self.relpath.split("/")[2].removesuffix(".mp4")

    def files(self) -> dict:
        """Suffix next to the stem -> served bytes, or None for 404."""
        video = b"".join(frame(self.stem, i) for i in range(len(self.gui) + 1))
        return {
            "-options.json": options() if self.options is GOOD else self.options,
            ".jsonl": jsonl(self.gui) if self.jsonl is GOOD else self.jsonl,
            ".mp4": video,
        }

    def record(self, subset: str) -> dict:
        """Its episode record of a plan shard: options parsed for the reasons `options`,
        `no_gui` and valid episodes, else null."""
        raw = self.files()["-options.json"]
        parsed = json.loads(raw) if self.reason in (None, "options", "no_gui") else None
        return {"subset": subset, "relpath": self.relpath, "reason": self.reason,
                "options": parsed}  # fmt: skip

    def planned(self, subset: str, i: int) -> dict:
        """Its frame i as a plan shard holds it."""
        id_ = f"{self.version}_{self.stem}_{i}"
        return {"id": id_, "subset": subset, "relpath": self.relpath,
                "version": self.version, "frame": i, "prev_line": lines(self.gui)[i - 1],
                "line": lines(self.gui)[i]}  # fmt: skip


class World:
    """Indices in config order (subset -> episodes in index order, repeats allowed)."""

    def __init__(self, episodes: dict, index_urls: dict, basedirs: dict):
        self.episodes, self.index_urls, self.basedirs = episodes, index_urls, basedirs
        self.eps = {(s, e.relpath): e for s, es in episodes.items() for e in es}
        self.files = {
            self.url(s, e, suffix): data
            for (s, _), e in self.eps.items()
            for suffix, data in e.files().items()
        }
        self.fetched = []

    def url(self, subset: str, e: Ep, suffix: str) -> str:
        return self.basedirs[subset] + e.relpath.removesuffix(".mp4") + suffix

    def index(self, subset: str) -> bytes:
        relpaths = [e.relpath for e in self.episodes[subset]]
        return json.dumps(
            {"basedir": self.basedirs[subset], "relpaths": relpaths}
        ).encode()

    def fetch(self, url: str) -> bytes | None:
        """The `fetch` of Interface: bytes, or None for 404."""
        self.fetched.append(url)
        by_url = {u: s for s, u in self.index_urls.items()}
        return self.index(by_url[url]) if url in by_url else self.files.get(url)


def make_world(episodes: dict, order=SUBSETS, host: str = "http://h.test") -> World:
    """Subsets of `order` absent from `episodes` have empty indices."""
    full = {s: episodes.get(s, []) for s in order}
    index_urls = {s: f"{host}/idx/{s}.json" for s in order}
    basedirs = {s: f"{host}/b{k}/" for k, s in enumerate(order)}
    return World(full, index_urls, basedirs)


def reference_draws(world: World, n: int, m: int, seed: int) -> list[dict]:
    """(3), transcribed: per draw its episode records, its frames and the generator state
    after it."""
    rng = np.random.Generator(np.random.PCG64(seed))
    pools = {
        s: list(dict.fromkeys(e.relpath for e in es))
        for s, es in world.episodes.items()
    }
    draws, planned = [], 0
    while planned < n and any(pools.values()):
        active = [s for s in world.episodes if pools[s]]
        s = active[rng.integers(len(active))]
        draw = {"episodes": [], "frames": []}
        while pools[s]:
            e = world.eps[s, pools[s].pop(rng.integers(len(pools[s])))]
            draw["episodes"].append(e.record(s))
            if e.reason is None:
                f_e = eligible(e.gui)
                r = min(m, len(f_e), n - planned)
                chosen = rng.choice(f_e, size=r, replace=False)
                draw["frames"] = [e.planned(s, int(i)) for i in chosen]
                planned += r
                break
        draws.append(draw | {"rng": rng.bit_generator.state})
    return draws


def reference_shards(draws: list[dict], k: int, before: tuple = ()) -> list[dict]:
    """(3'): `draws` cut into shards by the runs `before`, each (h, K): interrupted after
    writing h shards of K draws, then by a run with K = `k` ending the plan."""
    starts = [0]
    for h, size in before:
        starts.append(starts[-1] + h * size)
    sizes = [size for _, size in before] + [k]
    cuts = [
        (a, size, b) for a, size, b in zip(starts, sizes, starts[1:] + [len(draws)])
    ]
    chunks = [
        draws[a : min(a + size, b)] for a, size, b in cuts for a in range(a, b, size)
    ]
    return [{"episodes": joined(c, "episodes"), "frames": joined(c, "frames"),
             "rng": c[-1]["rng"]} for c in chunks]  # fmt: skip


def joined(shards: list[dict], key: str) -> list[dict]:
    """The episode records (key "episodes") or frames ("frames") of `shards` in order."""
    return [item for shard in shards for item in shard[key]]
