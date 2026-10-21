"""(1), (S), (H) of specs/preference-pool.md: the head of an entry, the sessions of a
group, the head shards of sessions/."""

import json
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import httpx

from mcpref import exact
from mcpref.clips.store import Pools
from mcpref.recognition.stored import shard_bar
from mcpref.runs.files import json_bytes
from mcpref.shards.store import complete_shards, write_shard
from mcpref.vpt.downloads import fetch, send
from mcpref.vpt.episodes import version_of, version_ok

if TYPE_CHECKING:
    from mcpref.clips.collect import PerceptionDataConfig
    from mcpref.pool.run import PreferencePoolConfig

STATUSES = ("ok", "missing", "empty", "malformed")
FIELDS = ("stats", "serverTick", "milli")
Group = tuple[str, str, list[str]]
"""Notation: a group: its subset, its index's basedir, its relpaths in pool order."""


def content_size(url: str, response: httpx.Response) -> int:
    """(1): the size of a 206's Content-Range, group 3 of its match."""
    header = response.headers.get("Content-Range")
    match = None if header is None else re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", header)
    if match is None:
        raise RuntimeError(f"GET {url}: HTTP 206 without Content-Range")
    return int(match[3])


def whole_file(p: "PerceptionDataConfig", url: str) -> bytes | None:
    """(1) fallback: the file by the ranged fetch (f) on a client of its own."""
    with httpx.Client(headers={"x-ms-version": p.x_ms_version}) as own:
        return fetch(own, p, url)


def line_0(body: bytes) -> dict:
    """(1): line 0, the bytes before the first newline, parsed: `malformed`, or `ok`
    with its stats, serverTick and milli."""
    try:
        value = json.loads(body.split(b"\n", 1)[0])
    except ValueError:
        value = None
    if not (
        isinstance(value, dict)
        and isinstance(value.get("stats"), dict)
        and type(value.get("serverTick")) is int
        and type(value.get("milli")) is int
    ):
        return {"status": "malformed"}
    return {"status": "ok"} | {k: value[k] for k in FIELDS}


def head(
    client: httpx.Client, p: "PerceptionDataConfig", url_stem: str, head_bytes: int
) -> dict:
    """(1): one ranged GET of the .jsonl: 404 `missing`, 416 `empty`, 206 its body (the
    whole file when it holds no newline and the file exceeds B); line 0 parsed."""
    url = url_stem + ".jsonl"
    ranged = partial(client.get, headers={"Range": f"bytes=0-{head_bytes - 1}"})
    response = send(ranged, p, url)
    status = response.status_code
    if status in (404, 416):
        return {"status": "missing" if status == 404 else "empty"}
    if status != 206:
        raise RuntimeError(f"GET {url}: HTTP {status}")
    body = response.content
    if b"\n" not in body and content_size(url, response) > head_bytes:
        body = whole_file(p, url)
        if not body:
            return {"status": "missing" if body is None else "empty"}
    return line_0(body)


def joins(i: dict, j: dict, delta_t: int, eps: Fraction, Z: set[str]) -> bool:
    """(S): piece j continues piece i: d = t_j - t_i > 0, delta_t d (1 - eps) <=
    mu_j - mu_i (exact), every key of sigma_i not in Z in sigma_j with a value >= it."""
    d = j["serverTick"] - i["serverTick"]
    return (
        d > 0
        and delta_t * d * (1 - eps) <= j["milli"] - i["milli"]
        and all(
            k in j["stats"] and j["stats"][k] >= v
            for k, v in i["stats"].items()
            if k not in Z
        )
    )


def sessions(
    entries: Sequence[tuple[str, dict]], cfg: "PreferencePoolConfig"
) -> list[list[str]]:
    """(S) on one group's (relpath, head) pairs: its ok entries by (mu, relpath) in one
    pass holding the current chain; the sessions in order started, orphans dropped."""
    ok = [(r, h) for r, h in entries if h["status"] == "ok"]
    eps, Z = exact("tick_slack", cfg.tick_slack), set(cfg.resetting_stats)
    chains, last = [], None
    for r, h in sorted(ok, key=lambda e: (e[1]["milli"], e[0])):
        if h["stats"] and last is not None and joins(last, h, cfg.tick_ms, eps, Z):
            chains[-1][1].append(r)
        else:
            chains.append((not h["stats"], [r]))
        last = h
    return [pieces for started, pieces in chains if started]


def groups(pools: Pools, min_version: str) -> list[Group]:
    """Notation: the entries (each pool without the relpaths below min_version) grouped
    by (subset, alias, hex), in order of their first entry; entries in pool order."""
    found = {}
    for s, (basedir, pool) in pools.items():
        for r in pool:
            if version_ok(version_of(r), min_version):
                alias, hex_, _, _ = PurePosixPath(r).stem.rsplit("-", 3)
                found.setdefault((s, alias, hex_), (s, basedir, []))[2].append(r)
    return list(found.values())


def head_shards(found: list[Group], size: int) -> list[list[Group]]:
    """Notation: consecutive groups, a new head shard started after the group that
    brings the current one to >= entries_per_shard entries."""
    shards, current, n = [], [], 0
    for group in found:
        current.append(group)
        n += len(group[2])
        if n >= size:
            shards.append(current)
            current, n = [], 0
    return shards + ([current] if current else [])


def fetch_heads(
    client: httpx.Client,
    p: "PerceptionDataConfig",
    cfg: "PreferencePoolConfig",
    entries: list[tuple[str, str]],
    update: Callable,
) -> list[dict]:
    """(H): the heads of (basedir, relpath) entries on head_workers threads, `update()`
    after each; a failure cancels the queued ones, raising the first failing entry's."""
    threads = ThreadPoolExecutor(cfg.head_workers)
    url_stems = [basedir + r.removesuffix(".mp4") for basedir, r in entries]
    jobs = [threads.submit(head, client, p, u, cfg.head_bytes) for u in url_stems]
    try:
        for job in as_completed(jobs):
            if job.exception() is not None:
                break
            update()
    finally:
        threads.shutdown(cancel_futures=True)
    failed = [j for j in jobs if not j.cancelled() and j.exception() is not None]
    if failed:
        raise failed[0].exception()
    return [job.result() for job in jobs]


def sessions_json(
    shard: list[Group], heads: list[dict], cfg: "PreferencePoolConfig"
) -> dict:
    """Datasets sessions/: a head shard's sessions.json, the sessions of its groups in
    group order, each group's by (S); the counts of its entries' statuses."""
    found, rest = [], iter(heads)
    for s, _, relpaths in shard:
        pairs = [(r, next(rest)) for r in relpaths]
        found += [{"subset": s, "pieces": pieces} for pieces in sessions(pairs, cfg)]
    statuses = [h["status"] for h in heads]
    return {"sessions": found, "entries": {k: statuses.count(k) for k in STATUSES}}


def store_sessions(
    cfg: "PreferencePoolConfig",
    p: "PerceptionDataConfig",
    client: httpx.Client,
    d: Path,
    pools: Pools,
) -> None:
    """(H): the head shards h_H, ... of d, each written by write_shard once its heads
    are fetched (the cut follows from the indices); bar `sessions`."""
    shards = head_shards(groups(pools, p.min_version), cfg.entries_per_shard)
    entries = [[(b, r) for _, b, rs in shard for r in rs] for shard in shards]
    h = complete_shards(d)
    total, initial = sum(map(len, entries)), sum(map(len, entries[:h]))
    with shard_bar("sessions", len(shards), h, total, initial, "episode") as update:
        for eta in range(h, len(shards)):
            heads = fetch_heads(client, p, cfg, entries[eta], update)
            data = json_bytes(sessions_json(shards[eta], heads, cfg))
            with write_shard(d, eta) as staged:
                (staged / "sessions.json").write_bytes(data)
