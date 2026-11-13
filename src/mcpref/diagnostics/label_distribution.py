"""Exact distribution and Matplotlib histogram of corrected preference labels."""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from matplotlib.ticker import PercentFormatter

from mcpref.diagnostics.charts import figure, save_png
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.progress import shard_bar
from mcpref.infrastructure.runs import files
from mcpref.infrastructure.shards.store import complete_shards, shard_path

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


def histogram(report: dict):
    labels = sorted(
        (key for key in report["counts"] if key != "incomparable"), key=float
    )
    labels = labels + ["incomparable"] if report["pairs"] else []
    result = figure(f"Final preference labels | {report['pairs']:,} pairs", 1)
    result.set_size_inches(10, max(4, 0.35 * len(labels) + 1.5))
    axis = result.axes[0]
    axis.set_xlabel("Share of all retained pairs")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    if not labels:
        axis.text(0.5, 0.5, "No pairs", ha="center", transform=axis.transAxes)
        return result
    display = [f"p={key}" if key != "incomparable" else key for key in labels]
    bars = axis.barh(
        display, [report["shares"][key] for key in labels], color="#2463a7"
    )
    axis.bar_label(
        bars,
        labels=[
            f"{report['counts'][key]:,} ({report['shares'][key]:.1%})" for key in labels
        ],
        padding=5,
        fontsize=10,
    )
    axis.invert_yaxis()
    axis.margins(x=0.28)
    return result


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = read_distribution(args.labels_dir)
        files.write_json(args.out_dir / "distribution.json", report)
        save_png(histogram(report), args.out_dir / "histogram.png")
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
