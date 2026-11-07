"""Producer outboxes, per-channel identities/watermarks and crash recovery."""

import copy
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from mcpref import REPO
from mcpref.diagnostics.artifacts import (
    complete_records,
    input_identity,
    iter_records,
    write_record,
)
from mcpref.diagnostics.config import DiagnosticsConfig, load_config
from mcpref.diagnostics.state import load_state, save_state
from mcpref.diagnostics.types import MetricRecord
from mcpref.runs.files import write_json
from mcpref.shards.store import name, sync

CHANNELS = ("vote", "label", "graph", "pretrain-step", "pretrain-reference")
Join = Callable[[MetricRecord, Sequence[str]], MetricRecord]


@contextmanager
def diagnostic_bar(
    desc: str, total: int, initial: int, unit: str
) -> Iterator[tqdm | None]:
    if total == initial:
        line = "already done" if initial else "nothing to do"
        print(f"{desc}: {line}", file=sys.stderr)
        yield None
    else:
        with tqdm(
            total=total, initial=initial, desc=desc, unit=unit, file=sys.stderr
        ) as bar:
            yield bar


class DiagnosticRun:
    def __init__(self, cfg: DiagnosticsConfig, state: dict):
        self.cfg = cfg
        self.out_dir = REPO / cfg.out_dir
        self.state = state

    @classmethod
    def load(cls, path: Path) -> "DiagnosticRun":
        cfg = load_config(path)
        return cls(cfg, load_state(path, cfg))

    def save(self) -> None:
        save_state(self.out_dir, self.state)

    def identity(self, channel: str) -> str:
        paths = [
            source
            for event in self.state["channels"][channel]["events"]
            for source in event["source"]
        ]
        return input_identity(paths) if paths else ""

    def register(self, channel: str, *, watermark: int, next: int) -> None:
        channels = self.state["channels"]
        if channel not in channels:
            channels[channel] = {
                "watermark": watermark,
                "next": next,
                "events": [],
                "records": 0,
                "complete": False,
            }
            self.state["sha256"]["inputs"][channel] = ""
        elif channels[channel]["watermark"] != watermark:
            raise RuntimeError(f"{channel} diagnostics watermark changed")
        elif self.identity(channel) != self.state["sha256"]["inputs"][channel]:
            raise RuntimeError(f"{channel} diagnostics source inputs changed")
        self.save()

    def stage(
        self,
        channel: str,
        ordinal: int,
        event_id: str,
        next: int,
        sources: Sequence[str],
        record: MetricRecord,
    ) -> None:
        pending = dict(record) | {"index": None}
        value = {
            "version": 1,
            "channel": channel,
            "event": {"id": event_id, "next": next},
            "source": list(sources),
            "record": pending,
        }
        path = self.out_dir / "outbox" / channel / f"{name(ordinal)}.json"
        write_json(path, value)

    def outbox(self, channel: str) -> Iterator[tuple[Path, dict]]:
        directory = self.out_dir / "outbox" / channel
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json")):
            yield path, json.loads(path.read_bytes())

    def entries(self, channel: str, event_id: str) -> list[tuple[Path, dict]]:
        return [
            (path, value)
            for path, value in self.outbox(channel)
            if value["event"]["id"] == event_id
        ]

    def has_entries(self, channel: str) -> bool:
        directory = self.out_dir / "outbox" / channel
        return directory.exists() and any(directory.glob("*.json"))

    def event_ids(self, channel: str) -> list[str]:
        return list(
            dict.fromkeys(value["event"]["id"] for _, value in self.outbox(channel))
        )

    def accepted(self, kind: str, event_id: str) -> MetricRecord | None:
        for record in iter_records(self.out_dir, kind):
            if record["id"] == event_id:
                return record
        return None

    def remove(self, paths: Sequence[Path]) -> None:
        for path in paths:
            path.unlink()
        if paths:
            sync(paths[0].parent)

    def accept(self, channel: str, event_id: str, *, join: Join | None = None) -> bool:
        entries = self.entries(channel, event_id)
        if not entries:
            return False
        sources = entries[0][1]["source"]
        if any(entry["source"] != sources for _, entry in entries):
            raise RuntimeError(f"{event_id} outbox sources differ")
        if any(not (REPO / source).is_file() for source in sources):
            self.remove([path for path, _ in entries])
            return False
        state = self.state["channels"][channel]
        already = any(event["id"] == event_id for event in state["events"])
        new_records = 0
        for _, entry in entries:
            pending = copy.deepcopy(entry["record"])
            pending = join(pending, sources) if join is not None else pending
            accepted = self.accepted(pending["kind"], pending["id"])
            if accepted is not None:
                if dict(accepted) | {"index": None} != pending:
                    raise RuntimeError(f"diagnostic event {pending['id']} differs")
                new_records += int(not already)
                continue
            index = complete_records(self.out_dir, pending["kind"])
            write_record(self.out_dir, pending["kind"], index, pending)
            new_records += 1
        if not already:
            state["events"].append({"id": event_id, "source": sources})
        state["records"] += new_records
        state["next"] = max(state["next"], entries[0][1]["event"]["next"])
        state["complete"] = False
        self.state["complete"] = False
        self.state["sha256"]["inputs"][channel] = self.identity(channel)
        self.save()
        self.remove([path for path, _ in entries])
        return not already

    def recover(self, channel: str, *, join: Join | None = None) -> None:
        for event_id in self.event_ids(channel):
            self.accept(channel, event_id, join=join)

    def finish(self, channel: str, *, source_complete: bool, done: bool) -> None:
        state = self.state["channels"][channel]
        state["complete"] = source_complete and done and not self.has_entries(channel)
        channels = self.state["channels"]
        self.state["complete"] = all(
            channel in channels and channels[channel]["complete"]
            for channel in CHANNELS
        )
        self.save()
