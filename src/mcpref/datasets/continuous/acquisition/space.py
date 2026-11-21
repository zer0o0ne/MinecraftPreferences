"""One scheduler accounts for all admitted source writes and the publisher."""

import shutil
from pathlib import Path


def source_bytes(piece: dict) -> int:
    return piece["mp4"]["bytes"] + 2 * piece["jsonl"]["bytes"]


class Space:
    def __init__(self, root: Path, minimum: int):
        self.root, self.minimum = root, minimum
        self.downloads = {}

    @property
    def pending(self):
        self.downloads = {
            job: size for job, size in self.downloads.items() if not job.done()
        }
        return sum(self.downloads.values())

    def available(self, required: int) -> bool:
        pending = self.pending
        return shutil.disk_usage(self.root).free >= self.minimum + pending + required

    def check(self, required: int) -> None:
        if not self.available(required):
            raise RuntimeError(
                "Disk reserve reached; collection remains resumable and incomplete"
            )

    def track(self, job, size: int) -> None:
        self.downloads[job] = size
