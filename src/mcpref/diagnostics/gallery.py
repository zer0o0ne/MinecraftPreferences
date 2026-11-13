"""Compact local navigation for twelve thematic diagnostic PNGs."""

import html
import re
from pathlib import Path
from urllib.parse import quote, unquote

from mcpref.infrastructure.shards.store import write_file

GUIDE = """
<h1>Training diagnostics</h1>
<p>Twelve dashboards. Click an image to open the full-resolution PNG.
Saved values only; no model evaluation, smoothing or metric recomputation.</p>
<details><summary>Collection cadence and missing data</summary>
<p>Repository defaults (a run can override these):</p>
<table><tr><th>Observations</th><th>Collected / published</th></tr>
<tr><td>Pretrain and RL losses, basic metrics</td><td>Every optimizer step / every 100 steps + final</td></tr>
<tr><td>Pretrain pooled windows</td><td>Every 100 steps + final partial window</td></tr>
<tr><td>Rich pretrain policy / masks</td><td>Checkpoint-boundary batch, normally every 100 steps</td></tr>
<tr><td>Fixed reference</td><td>Step 0, then every 500; 64 pairs per split</td></tr>
<tr><td>Gradient flow</td><td>First / middle / last labelled step, at most three</td></tr>
<tr><td>MCTS / milestones</td><td>Each real action / each sampling block; published every 1000 actions</td></tr>
<tr><td>Memory / labels</td><td>Once per RL cycle / once per produced dataset shard</td></tr></table>
<p>Null-only series do not occupy axes. Missing values break lines; isolated points
remain visible. A missing population is not zero: succession needs comparable pairs,
clipping by sign needs advantages of that sign, first times need reached milestones.
Old runs are not backfilled. Pretrain retrain has no reference or selected gradient-flow;
RL retrain currently saves losses and learning rates only. Bootstrap bounds need 20 clusters.</p>
</details>
<details><summary>Known producer limitations</summary>
<p>Pretrain block-33 prior_ce uses the current batch prior; top-k uses k=1.
PPO eligible sees only already-eligible rows. Affected legends say [producer issue];
the viewer does not correct historical numbers. Fixed reference, ordinary train,
windows, RL cycles and retrain keep distinct curves and local steps.
Reliability and survival show the latest available snapshot.</p></details>
"""


def previous_images(directory: Path) -> set[str]:
    path = directory / "index.html"
    if not path.is_file():
        return set()
    page = path.read_text()
    if "<h1>Training diagnostics</h1>" not in page:
        return set()
    names = {unquote(name) for name in re.findall(r'<img[^>]* src="([^"]+)"', page)}
    return {
        name
        for name in names
        if Path(name).name == name
        and re.fullmatch(r"[a-zA-Z0-9_.-]+__[a-f0-9]{12}(?:__[a-z0-9-]+)?\.png", name)
    }


def write_gallery(directory: Path, entries: list[dict]) -> None:
    old_images = previous_images(directory)
    cards = []
    for entry in entries:
        path, title = quote(entry["file"]), html.escape(entry["title"])
        status = (
            f"{entry['panels']} panels"
            if entry["panels"]
            else "No recorded observations"
        )
        sources = "".join(
            f"<li>{html.escape(source)}</li>" for source in entry["sources"]
        )
        reason = html.escape(entry["empty_reason"])
        cards.append(
            f'<article><h2><a href="{path}">{title}</a></h2><p>{status}</p>'
            f'<a href="{path}"><img loading="lazy" src="{path}" alt="{title}"></a>'
            f"<details><summary>Sources and availability</summary><p>{reason}</p>"
            f"<p>{entry['missing']} null-only series omitted.</p><ul>{sources}</ul></details></article>"
        )
    empty = (
        "<p>No completed observations found.</p>"
        if not any(e["panels"] for e in entries)
        else ""
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Training diagnostics</title><style>"
        "body{font:16px system-ui,sans-serif;max-width:1600px;margin:28px auto;padding:0 20px;"
        "color:#243448;background:#f5f7fa;line-height:1.5}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}"
        "article{padding:16px;background:white;border:1px solid #dfe5ec;border-radius:8px}"
        "img{width:100%;height:260px;object-fit:contain}h2{font-size:18px}a{color:#2463a7}"
        "details{margin:12px 0;overflow-wrap:anywhere}summary{cursor:pointer}"
        "td,th{padding:6px 12px;text-align:left;border-bottom:1px solid #dfe5ec}</style>"
        + GUIDE
        + empty
        + "<main>"
        + "".join(cards)
        + "</main></html>\n"
    )
    write_file(directory / "index.html", page.encode())
    # Only remove prior viewer outputs referenced by its own report, after success.
    for name in sorted(old_images - {entry["file"] for entry in entries}):
        (directory / name).unlink(missing_ok=True)
