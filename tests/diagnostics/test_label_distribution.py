"""Offline exact distribution and histogram of corrected preference labels."""

import json

from PIL import Image

from mcpref.diagnostics.label_distribution import histogram, main
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import write_shard


def write_labels(directory, shards, complete=True):
    write_json(directory / "run.json", {"complete": complete})
    for k, pairs in enumerate(shards):
        with write_shard(directory / "shards", k) as staged:
            (staged / "pairs.json").write_text(json.dumps(pairs))


def pair(p, inc):
    return {"labels": {"p": p, "inc": inc}}


def test_distribution_and_histogram_across_shards(tmp_path, capsys):
    labels, out = tmp_path / "labels", tmp_path / "out"
    write_labels(
        labels,
        [
            [pair(0.5, 0), pair(None, 1)],
            [pair(0.75, 0), pair(0.5, 0), pair(1.0, 0)],
        ],
    )
    assert main([str(labels), str(out)]) == 0
    report = json.loads((out / "distribution.json").read_text())
    assert report == {
        "pairs": 5,
        "comparable": 4,
        "incomparable": 1,
        "counts": {"0.5": 2, "0.75": 1, "1.0": 1, "incomparable": 1},
        "shares": {"0.5": 0.4, "0.75": 0.2, "1.0": 0.2, "incomparable": 0.2},
    }
    image = Image.open(out / "histogram.png")
    assert image.format == "PNG" and image.width >= 1000
    result = histogram(report)
    assert [label.get_text() for label in result.axes[0].get_yticklabels()] == [
        "p=0.5",
        "p=0.75",
        "p=1.0",
        "incomparable",
    ]
    assert [bar.get_width() for bar in result.axes[0].patches] == [0.4, 0.2, 0.2, 0.2]
    result.clear()
    assert not list(out.rglob("*.part"))
    assert capsys.readouterr().out == ""


def test_replot_is_deterministic(tmp_path):
    labels, out = tmp_path / "labels", tmp_path / "out"
    write_labels(labels, [[pair(0.6, 0)]])
    main([str(labels), str(out)])
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    assert main([str(labels), str(out)]) == 0
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_empty_complete_dataset(tmp_path, capsys):
    labels, out = tmp_path / "labels", tmp_path / "out"
    write_labels(labels, [])
    assert main([str(labels), str(out)]) == 0
    assert json.loads((out / "distribution.json").read_text()) == {
        "pairs": 0,
        "comparable": 0,
        "incomparable": 0,
        "counts": {"incomparable": 0},
        "shares": {},
    }
    assert Image.open(out / "histogram.png").format == "PNG"
    result = histogram({"pairs": 0, "counts": {}, "shares": {}})
    assert result.axes[0].texts[0].get_text() == "No pairs"
    result.clear()
    assert "label distribution: nothing to do" in capsys.readouterr().err


def test_incomplete_dataset_is_rejected(tmp_path, capsys):
    labels, out = tmp_path / "labels", tmp_path / "out"
    write_labels(labels, [[pair(0.6, 0)]], complete=False)
    assert main([str(labels), str(out)]) == 1
    assert not out.exists()
    assert f"{labels} is not complete" in capsys.readouterr().err
