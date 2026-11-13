"""A local report linking the plots produced by this invocation."""

import html
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from mcpref.infrastructure.shards.store import write_file

GUIDE = """
<h1>Training diagnostics</h1>
<p>Saved observations only. No model evaluation, smoothing or metric recomputation.</p>
<h2>Where to start</h2>
<ol>
<li><b>Pretrain:</b> open saved reference losses on validation, then train.
Compare joint, P_only, phi_only and initial on the same fixed population.
Training-window losses describe changing batches; they do not measure generalisation.</li>
<li><b>Training health:</b> inspect loss components, learning rates, gradient norms,
clip coefficients and update ratios. Read missing observations as missing, never zero.</li>
<li><b>RL:</b> inspect each cycle and retrain phase separately. Start with losses,
PPO KL / clipping / ESS, then MCTS outcomes and actual milestone coverage.</li>
</ol>
<details><summary>Known limitations of existing producer records</summary>
<p>Pretrain block-33 prior_ce currently uses the current batch's target prior;
topk_support currently uses k=1. PPO eligible currently counts only rows already
selected as eligible. These fields cannot establish performance against a fixed
global prior, configured top-k quality, or batch eligibility coverage.
Their stored numbers are shown without correction.</p>
<p>Reliability and survival curves show the latest snapshot for each source.
Old plots from earlier invocations may remain on disk; this page links only the
current selection. Source population, cycle and local step are preserved.</p>
</details>
"""


def write_gallery(directory: Path, entries: list[dict]) -> None:
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["group"]].append(entry)
    sections, navigation = [], []
    for i, (group, plots) in enumerate(groups.items()):
        title = html.escape(group.replace("-", " ").title())
        navigation.append(f'<li><a href="#group-{i}">{title}</a> ({len(plots)})</li>')
        cards = []
        for entry in plots:
            path = quote(entry["file"], safe="/")
            label = html.escape(entry["title"])
            cards.append(
                f'<figure><figcaption>{label}</figcaption><a href="{path}">'
                f'<img loading="lazy" src="{path}" alt="{label}"></a></figure>'
            )
        sections.append(
            f'<section id="group-{i}"><h2>{title}</h2>{"".join(cards)}</section>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Training diagnostics</title><style>"
        "body{font:16px system-ui,sans-serif;max-width:1250px;margin:32px auto;"
        "padding:0 20px;color:#243448;background:#f5f7fa;line-height:1.6}"
        "figure{margin:24px 0;padding:16px;background:white;border:1px solid #dfe5ec;"
        "border-radius:8px}img{width:100%;height:auto}figcaption{overflow-wrap:anywhere;"
        "font-weight:600}a{color:#2463a7}details{padding:12px;background:#fff4d8}"
        "</style>"
        + GUIDE
        + "<nav><ul>"
        + "".join(navigation)
        + "</ul></nav>"
        + ("".join(sections) if entries else "<p>No completed observations found.</p>")
        + "</html>\n"
    )
    write_file(directory / "index.html", page.encode())
