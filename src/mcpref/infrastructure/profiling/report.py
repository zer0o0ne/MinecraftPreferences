"""Host spans and attributed device work are inclusive, overlapping quantities."""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

from mcpref.infrastructure.profiling.scopes import PREFIX
from mcpref.infrastructure.runs.files import write_json


def publish(trace, directory: Path, metadata: dict, cuda: bool, top_ops: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "trace.tmp.json.gz"
    trace.export_chrome_trace(str(temporary))
    temporary.replace(directory / "trace.json.gz")
    events = trace.key_averages()
    device_events = sum(
        event.count
        for event in events
        if event.device_type == torch.profiler.DeviceType.CUDA
    )
    measured_cuda = cuda and device_events > 0
    regions = [
        {
            "name": event.key.removeprefix(PREFIX),
            "calls": event.count,
            "host_s": event.cpu_time_total / 1e6,
            "device_s": event.device_time_total / 1e6 if measured_cuda else None,
        }
        for event in events
        if event.key.startswith(PREFIX)
    ]
    regions.sort(key=lambda row: row["host_s"], reverse=True)
    note = (
        "Inclusive host spans include GPU waits; device time is attributed kernel work. "
        "Nested regions and CPU/GPU overlap: do not add rows or host and device times. "
        "Backward includes checkpoint recomputation. Wall time includes profiling overhead "
        "but excludes profiler export."
    )
    if cuda and not measured_cuda:
        note += " WARNING: no CUDA device events captured; GPU breakdown is unavailable (check CUPTI)."
        tqdm.write(
            "profile: WARNING: no CUDA device events captured; check CUPTI",
            file=sys.stderr,
        )
    write_json(
        directory / "summary.json",
        metadata
        | {"regions": regions, "note": note, "cuda_device_events": device_events},
    )
    lines = [
        f"{metadata['stage']} step {metadata['step']}: {metadata['wall_s']:.3f}s wall",
        f"batch={metadata['batch']} micro_batch={metadata['micro_batch']} items={metadata['items']}",
        f"CUDA peak memory (bytes): {metadata['memory']}",
        f"Qwen calls={len(metadata['qwen_batches'])}; rows per call: {[row['rows'] for row in metadata['qwen_batches']]}",
        note,
        "",
        f"{'Region':40s} {'Calls':>8s} {'Host (s)':>12s} {'Device (s)':>12s}",
    ]
    for row in regions:
        device = f"{row['device_s']:.3f}" if measured_cuda else "n/a"
        lines.append(
            f"{row['name']:40s} {row['calls']:8d} {row['host_s']:12.3f} {device:>12s}"
        )
    lines += [
        "",
        "Operators by self CPU time",
        events.table(sort_by="self_cpu_time_total", row_limit=top_ops),
    ]
    if measured_cuda:
        lines += [
            "Operators by self device time",
            events.table(sort_by="self_device_time_total", row_limit=top_ops),
        ]
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    tqdm.write(f"profile: {lines[0]} -> {directory / 'summary.txt'}", file=sys.stderr)
