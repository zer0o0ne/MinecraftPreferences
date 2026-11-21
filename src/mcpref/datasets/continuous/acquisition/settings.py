"""Transfer tuning does not change the continuous dataset's identity."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, positive_int, read_yaml


@dataclass(frozen=True)
class TransferConfig:
    workers: int
    prefetch_sources: int
    prefetch_bytes: int
    processing_reserve_bytes: int
    connections: int
    file_connections: int
    block_bytes: int
    read_bytes: int
    block_timeout: int
    keepalive_expiry: int


def load_transfer(path: Path) -> TransferConfig:
    values = config_mapping(read_yaml(path), path, TransferConfig)
    for key, value in values.items():
        positive_int(key, value)
    if values["file_connections"] > values["connections"]:
        raise ValueError("file_connections exceeds connections")
    if values["workers"] > values["prefetch_sources"]:
        raise ValueError("workers exceeds prefetch_sources")
    if values["read_bytes"] > values["block_bytes"]:
        raise ValueError("read_bytes exceeds block_bytes")
    return TransferConfig(**values)
