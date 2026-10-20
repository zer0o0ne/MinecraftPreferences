"""Config, (0), (8) of specs/perception-data.md and the run: the indices, run.json and
the rerun check; (1), (6), (9) in plan.py, (7) in store.py."""

import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from tqdm import tqdm

from mcpref import REPO, config_mapping, number, repo_path
from mcpref.clips.episode import Classify, Pipeline
from mcpref.clips.plan import pipeline, write_shards
from mcpref.clips.store import Pools, json_bytes, read_json, remove, resume
from mcpref.gui.label_files import sha256
from mcpref.shards.store import make_dirs, write_file
from mcpref.vpt.downloads import download
from mcpref.vpt.episodes import index

ANY, POSITIVE = (-math.inf, math.inf), (1, math.inf)
INTEGERS = {"width": ANY, "height": ANY, "fetch_streams": (1, 20), "http_retries": ANY}
INTEGERS |= {"states": (2, math.inf), "per_episode": POSITIVE, "n_clips": POSITIVE}
INTEGERS |= {"seed": (0, 2**32 - 1), "workers": POSITIVE, "classifier_batch": POSITIVE}
DURATIONS = ("http_timeout", "retry_delay", "ffmpeg_timeout")
STRINGS = ("x_ms_version", "min_version", "ffmpeg")
CONFIGS = ("clips", "screens", "gui_actions", "overlay")
SCALARS = (str, int, float, bool, type(None))
OUTPUT_FIELDS = ("index_urls", "min_version", "required_options", "width", "height")
OUTPUT_FIELDS += (*CONFIGS, "states", "per_episode", "n_clips", "seed")


@dataclass(frozen=True)
class PerceptionDataConfig:
    """Notation: configs/perception-data.yaml as written; states = L, per_episode = m,
    n_clips = N; paths relative to the repository root, resolved where used."""

    index_urls: dict[str, str]
    x_ms_version: str
    min_version: str
    required_options: dict[str, object]
    width: int
    height: int
    fetch_streams: int
    http_timeout: float
    http_retries: int
    retry_delay: float
    ffmpeg: str
    ffmpeg_timeout: float
    clips: str
    screens: str
    gui_actions: str
    overlay: str
    classifier_config: str
    states: int
    per_episode: int
    n_clips: int
    seed: int
    workers: int
    classifier_batch: int
    out_dir: str


def integer(name: str, value: object, low: float, high: float) -> int:
    """Notation: a YAML integer, not a boolean, in [low, high]."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} {value!r} is not an integer in [{low}, {high}]")
    return value


def string(name: str, value: object) -> str:
    """Notation: a string."""
    if type(value) is not str:
        raise ValueError(f"{name} {value!r} is not a string")
    return value


def mapping(name: str, value: object, types: tuple[type, ...], non_empty: bool) -> dict:
    """Notation: a mapping of strings to values of `types`, non-empty if required."""
    if not (
        type(value) is dict
        and (value or not non_empty)
        and all(type(k) is str and type(v) in types for k, v in value.items())
    ):
        raise ValueError(f"{name} {value!r} is not a mapping of strings to {types}")
    return value


def load_config(path: Path) -> PerceptionDataConfig:
    """Interface: the YAML file, validated, into PerceptionDataConfig as written."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PerceptionDataConfig)
    mapping("index_urls", raw["index_urls"], (str,), True)
    mapping("required_options", raw["required_options"], SCALARS, False)
    for k, bounds in INTEGERS.items():
        integer(k, raw[k], *bounds)
    for k in DURATIONS:
        number(k, raw[k])
    for k in STRINGS:
        string(k, raw[k])
    for k in (*CONFIGS, "classifier_config", "out_dir"):
        repo_path(k, raw[k])
    return PerceptionDataConfig(**raw)


def run_header(pipe: Pipeline, classifier_id: str) -> dict:
    """(8): the config as written, the SHA-256 of the named files, gamma_id."""
    named = {k: REPO / getattr(pipe.cfg, k) for k in CONFIGS}
    files = {"actions": pipe.clips.actions} | named
    files |= {"overlay_screens": pipe.overlay.screens}
    hashes = {k: sha256(path) for k, path in files.items()}
    config = dataclasses.asdict(pipe.cfg)
    return {"config": config, "sha256": hashes, "classifier": classifier_id}


def check_rerun(path: Path, header: dict) -> None:
    """(8): an existing run.json must hold this run's output fields, hashes and gamma_id,
    each compared as JSON text."""
    if not path.exists():
        return
    old = read_json(path)
    pairs = [(k, old["config"][k], header["config"][k]) for k in OUTPUT_FIELDS]
    pairs += [(f"sha256.{k}", old["sha256"][k], v) for k, v in header["sha256"].items()]
    pairs += [("classifier", old["classifier"], header["classifier"])]
    names = [k for k, a, b in pairs if json.dumps(a) != json.dumps(b)]
    if names:
        raise RuntimeError(f"{path} holds other values of {names}")


def is_index(body: bytes) -> bool:
    """(0): the body is a JSON object with a string basedir and a list relpaths."""
    try:
        data = json.loads(body)
    except ValueError:
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("basedir"), str)
        and isinstance(data.get("relpaths"), list)
    )


def store_indices(client: httpx.Client, cfg: PerceptionDataConfig, d: Path) -> None:
    """(0): each index without a stored file in d downloaded by `download` (one GET with
    its retries; a status other than 2xx aborts), in config order, checked and written by
    write_file; bar `indices`."""
    make_dirs(d)
    missing = [s for s in cfg.index_urls if not (d / f"{s}.json").exists()]
    if not missing:
        print("indices: already done", file=sys.stderr)
        return
    with tqdm(total=len(missing), desc="indices", unit="file") as bar:
        for s in missing:
            url = cfg.index_urls[s]
            body = download(client, cfg, url).content
            if not is_index(body):
                raise ValueError(
                    f"{url}: not a JSON object with a string basedir and a list relpaths"
                )
            write_file(d / f"{s}.json", body)
            bar.update(1)


def initial_pools(cfg: PerceptionDataConfig, d: Path) -> Pools:
    """Notation: per subset, in config order, its basedir and initial pool, from the
    index stored in d."""
    return {s: index((d / f"{s}.json").read_bytes()) for s in cfg.index_urls}


def collect(cfg: PerceptionDataConfig, classify: Classify, classifier_id: str) -> None:
    """Interface: (8) run.json checked and written, staging removed, (0) the indices,
    (7) the complete shards read, (1), (6) the plan continued, the counts added."""
    pipe = pipeline(cfg, classify)
    out, header = pipe.out, run_header(pipe, classifier_id)
    check_rerun(out / "run.json", header)
    write_file(out / "run.json", json_bytes(header))
    remove(out / "staging")
    with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}) as client:
        store_indices(client, cfg, out / "indices")
        pools = initial_pools(cfg, out / "indices")
        c, reasons = write_shards(pipe, client, resume(out / "shards", cfg.seed, pools))
    remove(out / "staging")
    counts = {"episodes": dict(reasons), "clips": c}
    write_file(out / "run.json", json_bytes(header | counts))
