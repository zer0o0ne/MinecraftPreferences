"""Exact distribution and SVG histogram of corrected preference labels."""

import argparse
import html
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.progress import shard_bar
from mcpref.infrastructure.runs import files
from mcpref.infrastructure.shards.store import complete_shards, shard_path, write_file

PROG = "mcpref.diagnostics.label_distribution"


def read_distribution(directory: Path) -> dict:
    files.finished_run(directory, str(directory), "complete")
    shards = complete_shards(directory / "shards")
    probabilities = Counter()
    incomparable = 0
    with shard_bar("label distribution", shards, 0, shards, 0, "shard") as update:
        for k in range(shards):
            path = shard_path(directory / "shards", k) / "pairs.json"
            for pair in json.loads(path.read_text()):
                labels = pair["labels"]
                if labels["inc"]:
                    incomparable += 1
                else:
                    probabilities[labels["p"]] += 1
            update()
    total = sum(probabilities.values()) + incomparable
    counts = {str(p): probabilities[p] for p in sorted(probabilities)}
    counts["incomparable"] = incomparable
    shares = {label: count / total for label, count in counts.items()} if total else {}
    return {
        "pairs": total,
        "comparable": sum(probabilities.values()),
        "incomparable": incomparable,
        "counts": counts,
        "shares": shares,
    }


def bar_svg(label, count, share, x, y, width, height) -> list[str]:
    text = html.escape(f"p={label}" if label != "incomparable" else label)
    return [
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="#2563eb"/>',
        f'<text x="{x + width / 2:.3f}" y="{y - 6:.3f}" text-anchor="middle" font-family="sans-serif" font-size="11">{count} ({share:.1%})</text>',
        f'<text x="{x + width / 2:.3f}" y="{y + height + 22:.3f}" text-anchor="middle" font-family="sans-serif" font-size="12">{text}</text>',
    ]


def histogram(report: dict) -> str:
    width, height, left, right, top, bottom = 960, 540, 70, 30, 60, 100
    plot_width, plot_height = width - left - right, height - top - bottom
    labels = [label for label in report["counts"] if label != "incomparable"]
    labels.sort(key=float)
    labels.append("incomparable")
    labels = labels if report["pairs"] else []
    peak = max(report["shares"].values(), default=1.0) or 1.0
    slot = plot_width / max(len(labels), 1)
    body = [
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="#94a3b8"/>',
        '<text x="70" y="32" font-family="sans-serif" font-size="20">Final preference labels</text>',
        f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">share of all pairs</text>',
    ]
    for index, label in enumerate(labels):
        share = report["shares"][label]
        bar_height = share / peak * plot_height
        x = left + (index + 0.15) * slot
        y = top + plot_height - bar_height
        body.extend(
            bar_svg(
                label,
                report["counts"][label],
                share,
                x,
                y,
                slot * 0.7,
                bar_height,
            )
        )
    if not report["pairs"]:
        body.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="16">no pairs</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(body)
        + "</svg>\n"
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = read_distribution(args.labels_dir)
        files.write_json(args.out_dir / "distribution.json", report)
        write_file(args.out_dir / "histogram.svg", histogram(report).encode())
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
