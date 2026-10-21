# Shards
Status: approved (2026-10-15; clarified after the refactor the same day: no `..` in paths, approved); 2026-10-20: `part_path` and `sync` in the interface, for block pretrain (`specs/pretrain.md`), approved
Stage: every block that builds or reads a dataset (AGENTS.md Code, Data); `specs/00-overview.md` block 27; first users: blocks vpt-gui-frames and perception-data

## Sources
- [U] the user, 2026-10-15: every dataset, when built and when used, is dumped to disk in shards and loaded from there; holding data in memory is unacceptable for safety and resources (AGENTS.md Code, Data); one shared mechanism, a block of its own, before the blocks that use it.
- [U2] the user, 2026-10-15, decision on the audited draft: shards survive an OS crash or power loss, not only an interrupted process (every file and directory entry synced before it is trusted); shard names, the in-order check and the error texts as proposed in the draft.
- [POSIX] `rename(2)`: renaming to a name that does not exist, or replacing a file, is atomic within one file system; `fsync(2)` on a file or a directory (Linux): its data, or its entries, reach the disk; a new entry survives a crash only once its parent directory is synced.
- [CPy] CPython 3.10.20 (`configs/spark-env.yaml`), audit of 2026-10-15: `os.rename` = `rename(2)` (`Modules/posixmodule.c:4682`); `os.fsync` = `fsync(2)` retried on `EINTR`, raising `OSError` without a file name (`:3539`, `:1761`); `os.open` adds `O_CLOEXEC` (`:9166`); `open(2)` with `O_RDONLY` and `fsync(2)` work on a directory on Linux (checked with the pinned interpreter); `shutil.rmtree` (`Lib/shutil.py:627-681`, `688-750`): errors raise, a missing path raises `FileNotFoundError`, symlinks inside are deleted, not followed; `pathlib.Path.rename` binds `os.rename` at import (`Lib/pathlib.py:306`).

## Notation
- $D$ — a sharded dataset: a directory.
- $k = 0, 1, \dots$ — shard index; $\mathrm{name}(k)$ = `f"{k:06d}"` (six digits, more when $k \ge 10^6$).
- Complete shard $k$: the directory $D/\mathrm{name}(k)$; staged shard $k$: the directory $D/\mathrm{name}(k)$`.part`.
- $n$ — the number of complete shards of $D$; $j$ — a missing shard index in (2).
- `part`, `path` — a file under a temporary name and its final name, in one directory; `data` — the bytes of a file.
- $\mathrm{sync}(p)$: `os.fsync` of `fd = os.open(p, os.O_RDONLY)`, `os.close(fd)` in a `finally`; `os.fsync` is called as `os.fsync`.
- $\mathrm{make}(p)$ [U2]: every missing directory from the topmost missing ancestor down to $p$ is created; then $p$ and every ancestor of $p$ up to `/` are synced, whether created now or earlier (a directory created by an interrupted run is synced too).

## Domain
- One writer of $D$ at a time; readers of $D$ run while no writer of $D$ runs.
- Paths given to the block have no `..` component.
- Shards are written in order: shard $k$ only after shards $0, \dots, k - 1$.
- A caller puts into a staged shard only regular files and directories, written with plain calls or renamed in from the same file system, each file closed before the caller returns; no symlinks.

## Behaviour
(1) Writing shard $k$ of $D$ [U2]:
  1. Checks, in this order, before anything is written: $k > 0$ and $D/\mathrm{name}(k-1)$ not a directory → `ValueError(f"{D}: shard {name(k - 1)} missing before {name(k)}")`; an entry $D/\mathrm{name}(k)$ of any kind → `FileExistsError(f"{D / name(k)} exists")`.
  2. $\mathrm{make}(D)$.
  3. An entry $D/\mathrm{name}(k)$`.part` left over → removed: `shutil.rmtree` for a directory (symlinks not followed), `os.unlink` for anything else; the staged shard is created (`mkdir`) and handed to the caller, who writes the shard into it.
  4. The caller returns normally: every regular file and every directory under the staged shard, at any depth, and the staged shard itself are synced, in any order; the staged shard is renamed to $D/\mathrm{name}(k)$; $D$ is synced.
  5. The caller raises: the staged shard stays as it is (removed by the next (1) for $k$); the exception propagates.
(2) $n$ of $D$ [U2], changing nothing on disk: $D$ missing → 0. Otherwise the names of $D$'s entries, sorted as strings: names $\mathrm{name}(k)$`.part` are ignored; a name that is no $\mathrm{name}(k)$, or whose entry is not a directory (symlinks not followed) → `ValueError(f"{D}: foreign entry {name}")` for the smallest such name; then the $k$ found must be $0, \dots, n - 1$, else `ValueError(f"{D}: shard {name(j)} missing")` for the smallest missing $j$; $\mathrm{sync}(D)$ (a shard renamed by an interrupted writer whose sync of $D$ did not happen is then trusted durably); the result is $n$.
(3) Publishing a file written under a temporary name `part` as `path` (same directory) [U2]: $\mathrm{sync}$(`part`); `os.rename(part, path)` (an existing file replaced); $\mathrm{sync}$ of the parent.
(4) Writing a file atomically [U2]: $\mathrm{make}$ of the parent of `path`; `data` written to `<path>.part` (a leftover one truncated), closed; published as `path` by (3).

## Interface
- Module `src/mcpref/shards/store.py`:
  - `def shard_path(D: Path, k: int) -> Path` — $D/\mathrm{name}(k)$.
  - `@contextlib.contextmanager def write_shard(D: Path, k: int) -> Iterator[Path]` — (1); yields the staged shard.
  - `def complete_shards(D: Path) -> int` — (2).
  - `def publish(part: Path, path: Path) -> None` — (3); for files that a program other than Python, or a library writing to a path (e.g. `torch.save` of block pretrain), writes under a temporary name (e.g. ffmpeg's `<id>.yuv.part` of vpt-gui-frames (c)).
  - `def write_file(path: Path, data: bytes) -> None` — (4).
  - `def make_dirs(p: Path) -> None` — $\mathrm{make}(p)$; for a caller's directory of single files (e.g. `frames/` and `indices/` of vpt-gui-frames), called before the caller trusts or publishes files there.
  - `def part_path(path: Path) -> Path` — the temporary name `<path>.part` of (1) and (4).
  - `def sync(p: Path) -> None` — $\mathrm{sync}(p)$ of Notation.
- Callers read shard $k$ through `shard_path(D, k)`, one shard at a time, for $k < n$; each caller's spec names its shards' files, items per shard and the state stored with a shard (skill `spec-format`, Datasets).
- Datasets: none of its own.
- Progress bars: none; each loop of the block (the syncs and `rmtree` of one shard, the listing of (2)) covers one shard or one listing and ends well under a couple of seconds.
- Tests:
  - (1): a shard's files and subdirectories present after the rename, no staged shard left; a caller raising → no complete shard, the staged shard left as it is, then replaced by the next (1) for $k$; the checks with their exact messages and nothing changed on disk (a missing previous shard, an existing directory or file named $\mathrm{name}(k)$, both at once → the missing previous shard); a leftover staged directory and a leftover staged file removed; $D$ created with its parents; `os.fsync` monkeypatched to record `os.readlink(f"/proc/self/fd/{fd}")` and whether $D/\mathrm{name}(k)$ exists at each call: every file and directory of the shard synced before the rename, $D$ after it, $D$ and all its ancestors synced by $\mathrm{make}$ also when $D$ existed.
  - (2): a missing $D$; staged shards ignored and left untouched; non-directories and non-canonical names (`0000123`, `12345`, `-00001`, a Unicode digit) as foreign entries, the smallest one named; gaps, the smallest named; foreign entries reported before gaps; $\mathrm{sync}(D)$ when $D$ exists.
  - Names: `shard_path(D, 10**6)` is `D/1000000`; (1) at $k = 10^6$ after $10^6 - 1$ (only those two directories present, the check of the previous shard satisfied); (2) on a $D$ holding only `999999` and `1000000` reports `000000` missing.
  - (3), (4), `make_dirs`: a new and an existing `path`; a leftover `<path>.part`; a missing parent created; the syncs (recorded as above): the file before the rename, the parent after it, the parent and all its ancestors by $\mathrm{make}$ also when they existed; `os.close` called when `os.fsync` raises.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1), (3) | `os.rename(src, dst)` | CPython 3.10.20 | — | [CPy] |
| (1)–(4) | `os.fsync(fd)` on `os.open(p, os.O_RDONLY)`, `os.close(fd)` | CPython 3.10.20 | `os.O_RDONLY` | [CPy] |
| (1) | `shutil.rmtree(p)` for a leftover staged directory, `os.unlink(p)` for any other leftover staged entry | CPython 3.10.20 | defaults (errors raise) | [CPy] |
| (1), (4), $\mathrm{make}$ | `Path.mkdir(parents=True, exist_ok=True)` | CPython 3.10.20 | `parents=True`, `exist_ok=True` | `Lib/pathlib.py` `mkdir` |

## Out of scope
- Several writers of one $D$ at once; reading $D$ while it is written; file systems without atomic `rename` (network file systems); a full disk (the exception propagates); symlinks.
- Deleting complete shards; shard formats (each caller's spec).
