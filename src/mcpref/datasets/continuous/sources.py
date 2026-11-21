"""ETag-pinned sources with durable receipts and bounded video staging."""

import gzip
import json
import shutil
from pathlib import Path

import httpx

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.vpt.downloads import download, fetch_to_path
from mcpref.datasets.vpt.episodes import options_ok
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import publish, write_file


def reserve(path: Path, required: int, minimum: int) -> None:
    if shutil.disk_usage(path).free < required + minimum:
        raise RuntimeError(
            "Disk reserve reached; collection remains resumable and incomplete"
        )


def response_check(expected: dict):
    def check(response: httpx.Response):
        if response.status_code == 412:
            raise RuntimeError(f"Source ETag changed: {expected['url']}")
        if response.status_code not in (200, 206):
            return
        if response.headers.get("ETag") != expected["etag"]:
            raise RuntimeError(f"Source ETag differs: {expected['url']}")
        if response.request.method == "HEAD":
            if int(response.headers["Content-Length"]) != expected["bytes"]:
                raise RuntimeError("Source size differs from frozen manifest")
        else:
            requested = response.request.headers["Range"].removeprefix("bytes=")
            if (
                response.headers.get("Content-Range")
                != f"bytes {requested}/{expected['bytes']}"
            ):
                raise RuntimeError("Source returned an incorrect Content-Range")

    return check


def fetch_file(pipe, piece: dict, kind: str, target: Path) -> bool:
    expected = piece[kind]
    headers = {"x-ms-version": pipe.cfg.x_ms_version, "If-Match": expected["etag"]}
    with httpx.Client(
        headers=headers, event_hooks={"response": [response_check(expected)]}
    ) as client:
        ok = fetch_to_path(client, pipe.cfg, expected["url"], target, pipe.ranges)
    if ok and target.stat().st_size != expected["bytes"]:
        raise RuntimeError("Downloaded source size differs from manifest")
    return ok


def pack(path: Path) -> Path:
    output = path.with_suffix(path.suffix + ".gz")
    staged = output.with_suffix(output.suffix + ".part")
    with (
        path.open("rb") as src,
        staged.open("wb") as dst,
        gzip.GzipFile(fileobj=dst, mode="wb", mtime=0) as stream,
    ):
        shutil.copyfileobj(src, stream)
    publish(staged, output)
    path.unlink()
    return output


def reusable(target: Path, piece: dict) -> dict | None:
    path = target / "receipt.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt["piece"] != piece:
        raise ValueError(f"Cached source manifest differs: {target}")
    for name, expected in receipt["files"].items():
        if (
            not (target / name).exists()
            and name == "source.mp4"
            and (target / "published.json").exists()
        ):
            continue
        if digest(target / name) != expected:
            raise ValueError(f"Cached source checksum differs: {target / name}")
    return receipt


def acquire(pipe, cfg, piece: dict) -> tuple[Path, dict]:
    target = REPO / cfg.source_cache / piece["piece_id"]
    target.mkdir(parents=True, exist_ok=True)
    if cached := reusable(target, piece):
        if cached.get("reason") is None and not (target / "source.mp4").exists():
            reserve(target, piece["mp4"]["bytes"], cfg.disk_reserve_bytes)
            if not fetch_file(pipe, piece, "mp4", target / "source.mp4"):
                raise RuntimeError(
                    "Previously published source video is no longer available"
                )
            if digest(target / "source.mp4") != cached["files"]["source.mp4"]:
                raise ValueError("Reacquired source video checksum differs")
        return target, cached
    needed = piece["mp4"]["bytes"] + 2 * piece["jsonl"]["bytes"]
    reserve(target, needed, cfg.disk_reserve_bytes)
    files, reason, options_etag = {}, None, None
    with httpx.Client(headers={"x-ms-version": pipe.cfg.x_ms_version}) as client:
        url = piece["unprobed_options_url"]
        try:
            response = download(client, pipe.cfg, url)
            options_etag = response.headers.get("ETag")
            write_file(target / "options.json", response.content)
            files["options.json"] = digest(target / "options.json")
            options = json.loads(response.content)
            if not isinstance(options, dict):
                reason = "malformed_options"
            elif not options_ok(options, pipe.cfg.required_options):
                reason = "options"
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            reason = "missing_options"
        except (ValueError, TypeError):
            reason = "malformed_options"
    if reason is None:
        for kind in ("jsonl", "mp4"):
            path = target / f"source.{kind}"
            if not fetch_file(pipe, piece, kind, path):
                reason = f"missing_{kind}"
                break
            if kind == "jsonl":
                path = pack(path)
            files[path.name] = digest(path)
    receipt = {
        "piece": piece,
        "files": files,
        "reason": reason,
        "options_etag": options_etag,
        "state": "sources_downloaded_video_decode_pending",
    }
    write_json(target / "receipt.json", receipt)
    return target, receipt
