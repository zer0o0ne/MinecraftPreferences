# Runs
Status: approved (2026-10-20); 2026-10-20: pretrain a producer and a reader (`specs/pretrain.md`), approved; 2026-10-20: `shard_files`, `new_hash`, `fingerprint`, `clip_fields` moved here from `pretrain/` (`specs/preference-pool.md`), approved; 2026-10-21: `json_bytes` moved here (`specs/preference-pool.md`), approved
Stage: every block that writes or reads a `run.json` (`specs/00-overview.md`, block 30)

## Sources
- [U] the user, 2026-10-20: the checks that another block's output is complete become one function in a subpackage of their own, named after what it holds (`runs/`: the run files of the blocks); one message, `{dir} is not complete`, everywhere; the shared helpers `sha256`, `write_json`, `check_fingerprints` move there unchanged; the dead code of `data/vpt/config.py` (`check_rerun`, `write_run`, no caller since the perception-data revision of 2026-10-15) is removed; the other rerun checks (perception-data, vpt-gui-frames, gui-labels) stay in their blocks.
- [U] the user, 2026-10-20, on the audited draft: a missing frames `run.json` also gives `{frames_dir} is not complete` (was `FileNotFoundError`); the removals are tested; `specs/perception-data.md` (its line on the vpt helpers no longer used) is left as it is.
- The code at commit `d877ff0`: `complete_run` (`gui/recognition/stored.py`), `counted_run` (`training/pretrain/data/index.py`), `run_counts`, `frame_counts`, `sha256`, `write_json`, `check_fingerprints`, `write_run` (`gui/label_files.py`), `check_rerun`, `write_run`, `check_same`, `check_outputs` (`data/vpt/config.py`).
- `specs/shards.md` (approved): `write_file` (4).

## Notation
- Run file of an output directory $D$: `D/run.json`, a JSON object written atomically by its producer; its marker, written once the output is complete:

  | Marker | Producers | Readers | Value written |
  |---|---|---|---|
  | `complete` | gui-masks; train-index; acyclic (the producer of $Q_\ell$); pretrain; vlm-labels | gui-stack, gui-classifier (of gui-masks); train-index (R) 4 (of $Q_\ell$); pretrain (R) 1 (of train-index); acyclic (R) (of vlm-labels) | `true` |
  | `clips` | perception-data; preference-pool (in perception-data's format) | train-index (R) 1; pretrain (R) 5; vlm-labels (R) | an integer ≥ 0 |
  | `frames` | vpt-gui-frames | gui-labels (4), gui-masks (R) | an object, possibly `{}` |
- $D$ — a directory resolved by the caller; `written` — $D$ as written in the caller's config, for messages; `marker` — a top-level key of a run file.

## Math
(1) Finished: `{D}/run.json` exists and its parsed object holds `marker` with a value that is neither JSON `false` nor `null` (identity: not `False`, not `None`; `0` and `{}` count as present).
(2) File hash: the lowercase hex SHA-256 of a file's bytes.
(3) JSON file: `json.dumps(value, indent=2) + "\n"` (UTF-8), written by `specs/shards.md` (4).
(4) Fingerprints of an existing run file at `path` (no file → nothing): the file holds a `sha256` object (else `KeyError`); for each fingerprint `name` in the order given, its stored value is the one under `name` in `sha256`, else at the top level (`sha256` wins when both exist; a name in neither → `KeyError`), compared with Python `!=` on the parsed values; the first that differs → `RuntimeError(f"{name} changed since the run started; delete {out_dir}")`.

## Domain
- A run file is a JSON object; its marker is absent or holds the value its producer writes (table above). Other values, a file that is not JSON or not an object: out of domain (errors of `json.loads` propagate).

## Interface
- Subpackage `src/mcpref/infrastructure/runs/` (new), module `infrastructure/runs/files.py`:
  - `def finished_run(directory: Path, written: str, marker: str) -> dict` — the parsed `{directory}/run.json` when (1) holds, else `RuntimeError(f"{written} is not complete")` (a missing file included).
  - `def sha256(path: Path) -> str` — (2).
  - `def json_bytes(value: object) -> bytes` — the bytes of (3), moved from `data/clips/store.py` (`specs/preference-pool.md`); `def write_json(path: Path, value: dict) -> None` — (3), writing `json_bytes(value)`.
  - `def check_fingerprints(path: Path, fingerprints: dict, out_dir: str) -> None` — (4).
  - `HASHED`, `CLIP_FIELDS`, `ABSENT` and `def clip_fields(run: dict) -> dict` — the clip-defining fields of `specs/train-index.md` Notation present in a `run.json`, {dotted name: value}.
  - `def shard_files(directory: str, file: str, n: int, fingerprint: hashlib._Hash, update: Callable) -> Iterator[tuple[int, bytes]]`, `def new_hash(fields: dict | None) -> hashlib._Hash`, `def fingerprint(directory: str, file: str, fields: dict | None, update: Callable[[], object]) -> str` — (4) of `specs/train-index.md` of one dataset; `file` — any file name of its shards (`specs/pretrain.md` Interface).
- Callers, each importing these names from `infrastructure/runs/files.py`; behaviour unchanged except the messages named:
  - gui-masks, `gui/recognition/frame_set.py`, (R) first check: `counts = run_counts(finished_run(REPO / frames_dir, frames_dir, "frames"))`; message `{frames_dir} is not complete` (was `{frames_dir}/run.json lacks the counts: the frames are not complete`, and `FileNotFoundError` for a missing file).
  - `gui/label_files.py`: `def run_counts(run: dict) -> dict[str, int]` — `{status: run["frames"].get(status, 0)}` for `ok`, `failed`; `def frame_counts(frames_dir: Path, written: str) -> dict[str, int]` — `run_counts(finished_run(frames_dir, written, "frames"))`, then `RuntimeError(f"no ok frame in {path}")` when `ok` < 1, `path` the resolved `{frames_dir}/run.json` (as before); `sha256`, `write_json`, `check_fingerprints` removed; `write_run` stays, calling `runs.files.write_json`.
  - gui-labels, `gui/labels.py`: `frame_counts(REPO / cfg.frames_dir, cfg.frames_dir)`; message `{frames_dir} is not complete` (was `{path} lacks the counts: the frames are not complete`, `path` the resolved `run.json`, and `FileNotFoundError` for a missing file).
  - gui-stack, `gui/stacking/run.py`: `finished_run(cfg.masks_dir, raw["masks_dir"], "complete")`.
  - gui-classifier, `gui/recognition/classifier.py`: `finished_run(REPO / masks_dir, masks_dir, "complete")`; `complete_run` removed from `gui/recognition/stored.py`.
  - train-index, `training/pretrain/data/index.py`: (R) 1 `finished_run(REPO / d, d, "clips")` for $C$ and $Q_c$; (R) 4 `finished_run(REPO / pair_labels_dir, pair_labels_dir, "complete")`; `counted_run` removed.
  - The other importers of `sha256`, `write_json`, `check_fingerprints` (`gui/recognition/{recognise,evaluate,stored}.py`, `data/clips/collect.py`) import them from `infrastructure/runs/files.py`.
  - `data/vpt/config.py`: `check_rerun` and `write_run` removed; its module docstring and `check_same`'s narrowed to vpt-gui-frames (d).
  - The importers of `shard_files`, `new_hash`, `fingerprint`, `clip_fields` and their constants (`training/pretrain/data/items.py`, `training/pretrain/data/index.py` and the pretrain loop's modules; blocks vlm-labels and acyclic) import them from `infrastructure/runs/files.py`.
- Revisions of other specs in this cycle ([U]), exact text:
  - `specs/gui-masks.md`: Status gains `; 2026-10-20: run files through \`runs/\` (\`specs/runs.md\`), the frames message, approved`; (R): `` `{frames_dir}/run.json` holds its counts (`{frames_dir}/run.json lacks the counts: the frames are not complete`) `` becomes `` `{frames_dir}/run.json` holds its counts, by `finished_run` of `specs/runs.md` with marker `frames` (`{frames_dir} is not complete`, also when the file is absent) ``; Interface, the `stored.py` sentence: `` ; `def complete_run(…)` … (used by blocks gui-stack and gui-classifier) `` removed, and a sentence added after it: `` The run files go through `infrastructure/runs/files.py` (`specs/runs.md`): `finished_run`, `sha256`, `write_json`, `check_fingerprints`. ``
  - `specs/gui-labels.md`: Status gains `; 2026-10-20: the frames check through \`runs/\` (\`specs/runs.md\`), approved`; (4): `holds its counts (the frames are complete)` becomes `` holds its counts (the frames are complete; `finished_run` of `specs/runs.md` with marker `frames`, message `{frames_dir} is not complete` as written, also when the file is absent) ``; Consumes gains `` `finished_run`, `write_json`, `sha256` (`specs/runs.md`) ``.
  - `specs/gui-stack.md`: Status gains `; 2026-10-20: \`finished_run\` of \`specs/runs.md\`, approved`; Consumes: `checked by \`mcpref.gui.recognition.stored.complete_run\`` becomes `` checked by `mcpref.infrastructure.runs.files.finished_run`, marker `complete` ``.
  - `specs/gui-classifier.md`: Status gains `; 2026-10-20: \`finished_run\` of \`specs/runs.md\`, approved`; (L) 1: `by \`complete_run\`, which also gives its contents` becomes `` by `finished_run` (`specs/runs.md`, marker `complete`), which also gives its contents ``; Reuses: `` `read_thresholds` and `complete_run` of `gui/recognition/stored.py` `` becomes `` `read_thresholds` of `gui/recognition/stored.py`, `finished_run` and `sha256` of `infrastructure/runs/files.py` ``, and `` `sha256` of `gui/label_files.py`, `` is removed from the list; Library mapping: `` `mcpref.gui.recognition.stored.read_thresholds`, `complete_run` `` becomes `` `mcpref.gui.recognition.stored.read_thresholds`; `mcpref.infrastructure.runs.files.finished_run` `` with Source read `` `specs/gui-masks.md` Interface; `specs/runs.md` ``.
  - `specs/train-index.md`: Status gains `; 2026-10-20: run files through \`runs/\` (\`specs/runs.md\`), approved`; Sources, the helpers entry: the `complete_run` clause becomes `` `finished_run(directory, written, marker)` (`specs/runs.md` (1)) `` and `check_fingerprints`, `write_json`, `sha256` are named as of `infrastructure/runs/files.py` (`specs/runs.md` (2)–(4)); (R) 1: `` exists and holds `clips` `` becomes `` exists and holds `clips` (`finished_run`, marker `clips`) ``; (R) 4: `` `complete_run` of $Q_\ell$ `` becomes `` `finished_run` of $Q_\ell$, marker `complete` ``; Reuses: `` `check_fingerprints`, `write_json`, `sha256` (`gui/label_files.py`), `complete_run`, `shard_bar` (`infrastructure/progress.py`) `` becomes `` `finished_run`, `check_fingerprints`, `write_json`, `sha256` (`infrastructure/runs/files.py`), `shard_bar` (`infrastructure/progress.py`) ``; Library mapping row (R), (W): `` `write_json`, `check_fingerprints`, `complete_run`, `shard_bar` `` becomes `` `finished_run`, `write_json`, `check_fingerprints` (`infrastructure/runs/files.py`), `shard_bar` ``, Source read gains `specs/runs.md`.
  - `specs/00-overview.md`: after row 29, `` | 30 | runs — the run files of the blocks: completeness of another block's output, fingerprints, JSON files, file hashes | all | 27 | — | ``; package map, a last line `` - `runs` → every block with a run.json → `specs/runs.md` ``.
- Datasets, checkpoints, progress bars: none (single small files).
- Tests:
  - `tests/infrastructure/runs/test_files.py`: (1) a missing file, a missing key, `false`, `null`, `0`, `{}`, `true`, an object; the message with `written`; (2) on known bytes; (3) the bytes written, atomically (a leftover `.part` replaced); (4) the order, `sha256` before the top level, `KeyError` without `sha256` or for a name in neither, `!=` on parsed values, no file.
  - The removals: `sha256`, `write_json`, `check_fingerprints` absent from `gui.label_files`, `complete_run` from `recognition.stored`, `counted_run` from `pretrain.index`, `check_rerun` and `write_run` from `vpt.config` (as `tests/gui/recognition/test_frame_set.py::test_label_files_helpers_removed`).
  - `tests/gui/recognition/test_stored.py`: the two `complete_run` tests, its import and its docstring entry removed (covered by (1)).
  - `tests/gui/recognition/test_evaluate_aborts.py`, `CHECKS[0]`: the message becomes `"{f} is not complete"`; a new case: `{frames_dir}/run.json` missing → the same message.
  - gui-labels: `tests/gui/test_label_files.py::test_abort_while_frames_lack_counts` and `tests/gui/test_label_files_checks.py::test_rerun_aborts_while_frames_lack_counts` assert `{frames_dir} is not complete` (as written); the `frames/run.json` case of `test_abort_on_unreadable_file` asserts it too.
  - Every other existing test keeps passing unchanged.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1), (4) | `json.loads`, `Path.read_text`, `Path.exists` | CPython 3.10.20 | — | — |
| (2) | `hashlib.sha256(...).hexdigest()` | CPython 3.10.20 | — | — |
| (3) | `json.dumps`; `mcpref.infrastructure.shards.store.write_file` | CPython 3.10.20; this repo | `indent=2` | `specs/shards.md` |

## Out of scope
- Unifying the rerun checks of perception-data, vpt-gui-frames and gui-labels; changing what the producers write.
