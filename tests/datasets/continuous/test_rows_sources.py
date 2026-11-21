"""Whole-source rollback, bounded row views and pinned range responses."""

import gzip
import json

import httpx
import pytest
from datasets.vpt.action_lines import episode

from mcpref.datasets.continuous.acquisition.http import check_response
from mcpref.datasets.continuous.rows import Rows
from mcpref.datasets.vpt.video.sequence import RowView


def source(tmp_path, lines):
    path = tmp_path / "source.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        for row in lines:
            stream.write(json.dumps(row) + "\n")
    return path, {
        "nominal_steps": len(lines),
        "observed_boundary": {
            "first_milli": lines[0]["milli"],
            "last_milli": lines[-1]["milli"],
            "first_server_tick": lines[0]["serverTick"],
            "last_server_tick": lines[-1]["serverTick"],
        },
    }


def test_disk_rows_preserve_real_world_start_and_rollback_bad_source(tmp_path):
    lines = episode("w w w w")
    path, piece = source(tmp_path, lines)
    rows = Rows(tmp_path / "rows.sqlite", 2)
    try:
        assert rows.append_source(path, piece) == 0
        assert rows[0]["stats"] == {}
        assert not rows[0]["starts_session"]
        assert [r["serverTick"] for r in RowView(rows, [3, 0, 2])] == [3, 0, 2]
        lines[2]["tick"] = 0
        path, piece = source(tmp_path, lines)
        with pytest.raises(ValueError, match="tick"):
            rows.append_source(path, piece)
        assert len(rows) == 4
    finally:
        rows.close()


@pytest.mark.parametrize(
    "field,value", [("ETag", "other"), ("Content-Range", "bytes 2-4/10")]
)
def test_wrong_source_version_or_range_is_rejected(field, value):
    expected = {"etag": "version", "bytes": 10, "url": "https://source.test/a"}
    headers = {"ETag": "version", "Content-Range": "bytes 1-3/10"} | {field: value}
    request = httpx.Request("GET", expected["url"], headers={"Range": "bytes=1-3"})
    response = httpx.Response(206, headers=headers, request=request)
    with pytest.raises(RuntimeError):
        check_response(response, expected, 1, 3)
