"""Validate torch.save ZIP records before accepting persisted tensor state."""

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import torch


def load_payload(path: Path, *, weights_only: bool):
    """torch.load ignores record CRCs; check them using bounded reads first."""
    try:
        with path.open("rb") as stream:
            with ZipFile(stream) as archive:
                damaged = archive.testzip()
                if damaged is not None:
                    raise BadZipFile(f"CRC mismatch in {damaged}")
            stream.seek(0)
            return torch.load(stream, map_location="cpu", weights_only=weights_only)
    except (OSError, BadZipFile, EOFError) as error:
        raise RuntimeError(f"Cannot read tensor payload {path}: {error}") from error
