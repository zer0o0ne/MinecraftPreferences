"""Durability of the outputs of `main run` in specs/vpt-gui-frames.md (Library mapping:
specs/shards.md): `make_dirs` of indices/ and frames/ in every run, `write_file` of each
index and of run.json, `write_shard` of each plan shard, `publish` of each frame. `os.fsync`
is recorded with the entries under out_dir at each call, as in specs/shards.md Tests."""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "shards"))
from store_disk import ancestors, fd_path

from mcpref.vpt.gui_frames import main


def record_syncs(monkeypatch, root: Path) -> list[tuple[Path, frozenset]]:
    """Replace `os.fsync` by a recorder of (path of the fd, entries under `root`)."""
    log, real = [], os.fsync

    def fsync(fd):
        walk = os.walk(root)
        log.append(
            (fd_path(fd), frozenset(Path(t, n) for t, d, f in walk for n in d + f))
        )
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return log


def synced(
    log, path: Path, present: Path | None = None, absent: Path | None = None
) -> bool:
    """Whether `path` was synced while `present` existed and `absent` did not."""
    return any(
        p == path and (present is None or present in e) and absent not in e
        for p, e in log
    )


def published(log, path: Path) -> bool:
    """New `path` written as `<path>.part`, synced before the rename; its parent after."""
    part = path.with_name(path.name + ".part")
    return synced(log, part, present=part, absent=path) and synced(
        log, path.parent, present=path, absent=part
    )


def test_fresh_run_syncs_every_output(site, monkeypatch):
    out = site.out.resolve()
    log = record_syncs(monkeypatch, out)
    assert main(["run", site.config()]) == 0
    for d in ("indices", "frames", "plan"):
        assert all(synced(log, a) for a in ancestors(out / d)), d
    assert all(
        published(log, out / "indices" / f"{s}.json") for s in site.world.index_urls
    )
    assert published(log, out / "run.json")
    for k in range(len(site.shards())):
        final = out / "plan" / f"{k:06d}"
        staged = final.with_name(final.name + ".part")
        assert synced(log, staged / "shard.json", absent=final)
        assert synced(log, staged, absent=final)
        assert synced(log, out / "plan", present=final)
    rows, _ = site.expected()
    assert all(published(log, out / "frames" / f"{r['id']}.yuv") for r in rows)


def test_rerun_makes_its_directories_durable_again(site, monkeypatch):
    """make_dirs of indices/ and frames/ in every run, also with nothing to do."""
    assert main(["run", site.config()]) == 0
    out = site.out.resolve()
    log = record_syncs(monkeypatch, out)
    assert main(["run", site.config()]) == 0
    made = ancestors(out / "indices") | ancestors(out / "frames")
    assert all(synced(log, a) for a in made)
