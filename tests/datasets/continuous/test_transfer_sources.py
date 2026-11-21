"""Source receipts remain reusable across the acquisition transport replacement."""

import gzip
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from mcpref.datasets.continuous import sources
from mcpref.datasets.continuous.acquisition import http as transfer


def checksum(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def collection(monkeypatch, tmp_path):
    payloads = {
        "/options": b'{"guiScale":1}',
        "/jsonl": b'{"tick":0}\n{"tick":1}\n',
        "/mp4": b"original-video-bytes",
    }
    statuses, etags, calls = {}, {}, []

    def handler(request):
        path = request.url.path
        calls.append(path)
        if path in statuses:
            return httpx.Response(statuses[path])
        body = payloads[path]
        if path == "/options":
            return httpx.Response(200, content=body, headers={"ETag": '"options"'})
        first, last = map(int, request.headers["Range"][6:].split("-"))
        return httpx.Response(
            206,
            headers={
                "ETag": etags.get(path, '"frozen"'),
                "Content-Range": f"bytes {first}-{last}/{len(body)}",
                "Content-Length": str(last - first + 1),
            },
            stream=httpx.ByteStream(body[first : last + 1]),
        )

    constructor = httpx.Client
    monkeypatch.setattr(
        transfer.httpx,
        "Client",
        lambda **kwargs: constructor(**kwargs, transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(sources, "REPO", tmp_path)
    access = SimpleNamespace(
        x_ms_version="test",
        http_timeout=2,
        http_retries=0,
        retry_delay=0,
        required_options={"guiScale": 1},
    )
    settings = SimpleNamespace(
        connections=2,
        file_connections=2,
        block_bytes=8,
        read_bytes=2,
        block_timeout=5,
        keepalive_expiry=10,
    )
    cfg = SimpleNamespace(source_cache="cache", disk_reserve_bytes=0)
    piece = {
        "piece_id": "piece",
        "unprobed_options_url": "https://source.test/options",
        **{
            kind: {
                "url": f"https://source.test/{kind}",
                "etag": '"frozen"',
                "bytes": len(payloads[f"/{kind}"]),
            }
            for kind in ("jsonl", "mp4")
        },
    }
    with transfer.Transport(access, settings) as transport:
        yield SimpleNamespace(
            root=tmp_path / "cache/piece",
            pipe=SimpleNamespace(cfg=access),
            cfg=cfg,
            piece=piece,
            transport=transport,
            payloads=payloads,
            statuses=statuses,
            etags=etags,
            calls=calls,
        )


def acquire(case):
    return sources.acquire(case.pipe, case.cfg, case.piece, case.transport)


def seed_receipt(case):
    case.root.mkdir(parents=True)
    files = {
        "options.json": case.payloads["/options"],
        "source.jsonl.gz": gzip.compress(case.payloads["/jsonl"], mtime=0),
        "source.mp4": case.payloads["/mp4"],
    }
    for name, body in files.items():
        (case.root / name).write_bytes(body)
    receipt = {
        "piece": case.piece,
        "files": {name: checksum(body) for name, body in files.items()},
        "reason": None,
        "options_etag": '"options"',
        "state": "sources_downloaded_video_decode_pending",
    }
    (case.root / "receipt.json").write_text(json.dumps(receipt))
    return receipt


def published(case):
    receipt = seed_receipt(case)
    (case.root / "published.json").write_text('{"fragment_shard":0}')
    (case.root / "source.mp4").unlink()
    return receipt


def test_legacy_complete_receipt_reused_without_network(collection):
    old = seed_receipt(collection)
    root, receipt = acquire(collection)
    assert root == collection.root and receipt == old
    assert collection.calls == []


def test_published_video_is_reacquired_and_receipt_preserved(collection):
    old = published(collection)
    _, receipt = acquire(collection)
    assert receipt == old
    assert collection.calls and set(collection.calls) == {"/mp4"}
    assert (
        checksum((collection.root / "source.mp4").read_bytes())
        == old["files"]["source.mp4"]
    )
    assert json.loads((collection.root / "receipt.json").read_text()) == old


def test_reacquired_video_with_same_etag_but_wrong_hash_is_rejected(collection):
    old = published(collection)
    collection.payloads["/mp4"] = b"x" * len(collection.payloads["/mp4"])
    with pytest.raises(ValueError, match="Reacquired source video checksum"):
        acquire(collection)
    assert json.loads((collection.root / "receipt.json").read_text()) == old


def test_missing_published_video_aborts_without_reclassifying_receipt(collection):
    old = published(collection)
    collection.statuses["/mp4"] = 404
    with pytest.raises(RuntimeError, match="no longer available"):
        acquire(collection)
    assert json.loads((collection.root / "receipt.json").read_text()) == old
    assert not (collection.root / "source.mp4").exists()
    assert not list(collection.root.glob("*.part"))


@pytest.mark.parametrize("name", ["options.json", "source.jsonl.gz", "source.mp4"])
def test_cached_hash_mismatch_is_rejected_before_network(collection, name):
    seed_receipt(collection)
    (collection.root / name).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Cached source checksum"):
        acquire(collection)
    assert collection.calls == []


def test_changed_frozen_manifest_is_rejected_before_network(collection):
    seed_receipt(collection)
    collection.piece["mp4"]["etag"] = '"new-version"'
    with pytest.raises(ValueError, match="Cached source manifest"):
        acquire(collection)
    assert collection.calls == []


@pytest.mark.parametrize("kind", ["jsonl", "mp4"])
def test_network_etag_mismatch_never_creates_success_receipt(collection, kind):
    collection.etags[f"/{kind}"] = '"new-version"'
    with pytest.raises(RuntimeError, match="ETag"):
        acquire(collection)
    assert not (collection.root / "receipt.json").exists()
    assert not list(collection.root.glob("*.part"))


@pytest.mark.parametrize(
    "missing,expected_files",
    [
        ("options", set()),
        ("jsonl", {"options.json"}),
        ("mp4", {"options.json", "source.jsonl.gz"}),
    ],
)
def test_missing_source_has_durable_reusable_rejection_receipt(
    collection, missing, expected_files
):
    collection.statuses[f"/{missing}"] = 404
    root, receipt = acquire(collection)
    assert receipt["reason"] == f"missing_{missing}"
    assert set(receipt["files"]) == expected_files
    assert receipt["options_etag"] == (None if missing == "options" else '"options"')
    assert json.loads((root / "receipt.json").read_text()) == receipt
    for name, digest in receipt["files"].items():
        assert checksum((root / name).read_bytes()) == digest
    assert not list(root.glob("*.part"))
    collection.calls.clear()
    assert acquire(collection) == (root, receipt)
    assert collection.calls == []


def test_new_download_packs_jsonl_and_hashes_all_published_files(collection):
    root, receipt = acquire(collection)
    assert receipt["reason"] is None
    assert set(receipt["files"]) == {"options.json", "source.jsonl.gz", "source.mp4"}
    assert (
        gzip.decompress((root / "source.jsonl.gz").read_bytes())
        == collection.payloads["/jsonl"]
    )
    assert not (root / "source.jsonl").exists()
    for name, digest in receipt["files"].items():
        assert checksum((root / name).read_bytes()) == digest
