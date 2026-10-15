"""Readers of the outputs of `collect` (specs/perception-data.md Datasets) under
`self.out`, and the check of a whole out_dir against the expected outputs."""

import json
import os

from vpt_world import SUBSETS

from .collect_expected import JSON_FILES, Expected


class OutDir:
    """Readers of out_dir = `self.out`."""

    def run_json(self) -> dict:
        return json.loads((self.out / "run.json").read_text())

    def completed(self) -> bool:
        """Whether run.json has the counts written at the end of a run that completes."""
        return {"episodes", "clips"} <= set(self.run_json())

    def raw(self, *parts: str) -> dict:
        """Every file under out_dir/<parts>: path relative to out_dir -> bytes."""
        root = self.out.joinpath(*parts)
        files = (
            sorted(x for x in root.rglob("*") if x.is_file()) if root.exists() else []
        )
        return {x.relative_to(self.out).as_posix(): x.read_bytes() for x in files}

    def tree(self, *parts: str) -> dict:
        """`raw` with the files of JSON_FILES parsed."""
        return {p: json.loads(x) if p.rsplit("/", 1)[-1] in JSON_FILES else x
                for p, x in self.raw(*parts).items()}  # fmt: skip

    def records(self) -> list[dict]:
        """The records of the complete shards, in shard order."""
        root = self.out / "shards"
        return [r for x in self.complete() for r in json.loads((root / x / "records.json").read_text())]  # fmt: skip

    def rows(self) -> list[dict]:
        """clip.json of each clip of the complete shards, in shard order, then as taken."""
        root = self.out / "shards"
        return [json.loads((root / x / "clips" / i / "clip.json").read_text())
                for x in self.complete()
                for r in json.loads((root / x / "records.json").read_text())
                for i in r["clips"] or []]  # fmt: skip

    def complete(self) -> list[str]:
        """The complete shards' names (shards spec (2): name(k), no `.part`)."""
        return [x for x in self.shard_names() if not x.endswith(".part")]

    def shard_names(self) -> list[str]:
        """The entries of out_dir/shards, sorted."""
        root = self.out / "shards"
        return sorted(os.listdir(root)) if root.exists() else []


def assert_outputs(site, ex: Expected) -> None:
    """out_dir of the Site `site` holds run.json with the counts, the indices as served,
    the shards of `ex` and nothing else (no staging)."""
    files = site.tree()
    assert json.loads(files.pop("run.json")) == site.expected_run(ex)
    indices = {f"indices/{s}.json": site.index_body(s) for s in SUBSETS}
    assert {p: files.pop(p) for p in list(files) if p.startswith("indices/")} == indices
    assert files == ex.tree()
    assert {"indices", "run.json"} <= set(os.listdir(site.out)) <= {"indices", "run.json", "shards"}  # fmt: skip
