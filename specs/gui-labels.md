# GUI labels
Status: approved (2026-10-10; clarified after the prototype the same day: paths, run.json form, no ok frame; 2026-10-14: progress bar, approved); revision of 2026-10-15 (AGENTS.md Data: plan shards in, label shards out), approved the same day
Stage: GUI-recognition experiment (Q GX; `specs/00-overview.md`, block 17)

## Sources
- [U] the user, 2026-10-10: the screen type of every sampled frame is labelled by the VLM; the labels are the reference for comparing mask variants (Q GX); VPT's `isGuiInventory` is an independent check (Q GX-3); every Spark action is a bash script.
- [U] the user, 2026-10-14: a progress bar after AGENTS.md Code (`tqdm`), advanced once per batch; an error message is the last line of stderr.
- [U] the user, 2026-10-15: rule AGENTS.md Code, Data (the VLM's answers are its exception: a rerun repeats requests and seeds, not answers); decisions: label shard $k$ = the labels of the `ok` frames of plan shard $k$, one `ask_many` call per shard (`batch` dropped); `no_answer` is final; each label record carries $g_f$; a rerun aborts when the frames changed since labelling started (fingerprint).
- `specs/vpt-gui-frames.md` (approved, revision of 2026-10-15): plan shards, frames, `run.json` counts as the completeness mark, RGB view (5); `specs/shards.md` (approved).
- `specs/gui-screens.md` (approved): classes, their one-line `description`.
- `specs/vlm-client.md` (approved): `ask_many`, `load_config`.

## Notation
- Plan shard $k$ — `{frames_dir}/plan/<name(k)>/shard.json` (`specs/vpt-gui-frames.md`); its frames in their order; $H$ — the number of plan shards.
- $h$ — the complete label shards (`specs/shards.md` (2)).
- $g_f$ — `line.isGuiInventory` of the plan record of an `ok` frame $f$; $K$ — `plan_shard_draws` of vpt-gui-frames.
- Fingerprint of the frames: $H$ and the `frames` counts of `{frames_dir}/run.json` (`ok`, `failed`; 0 when absent).
- Position of a frame — its 0-based index over the frames of all plan shards concatenated in shard order (`failed` frames included).
- $f$ — a frame with status `ok` (its `{frames_dir}/frames/<id>.yuv` exists); $x_f$ — its RGB view (vpt-gui-frames (5)).
- $\mathcal{K}$ — the catalogue's classes in catalogue order, plus `other`.
- $q_f$ — the prompt of $f$; $a_f$ — the VLM's answer (string or `None`); $y_f$ — the label.
- Config `configs/gui-labels.yaml` → `@dataclass(frozen=True) class GuiLabelsConfig` (paths relative to the repository root, kept as written in the config and in `run.json`, resolved against the root where used): `frames_dir`, `screens` (path of `configs/gui-screens.yaml`), `vlm` (path of `configs/vlm.yaml`), `instruction` (text before the image), `question` (text after the image; contains the placeholder `{classes}`), `class_line` (format of one class line, placeholders `{name}`, `{description}`), `other_description`, `seed`, `out_dir`.

## Domain
- `{out_dir}` holds only what this block writes.
- During a run, no vpt-gui-frames `run` on `{frames_dir}` and no other labelling run on `{out_dir}` (the fingerprint is checked once, at the start).

## Behaviour
(1) Prompt: $q_f$ = [`instruction`, $x_f$, `question` with `{classes}` replaced (`str.replace`) by one `class_line` per class of $\mathcal{K}$ in order, joined by `"\n"`, each with `{name}`, then `{description}` replaced (`str.replace`); `other` uses `other_description`]. $x_f$ = `yuv420p_to_rgb` with `width`, `height` from the config stored in `{frames_dir}/run.json`.
(2) Seed of $f$: `seed` + the position of $f$.
(3) Label: `no_answer` if $a_f$ is `None`; else $s$ = $a_f$ stripped of whitespace; if $s$ has length ≥ 2 and starts and ends with the same character among `` ` ``, `'`, `"`, those two characters are removed and $s$ is stripped again; $y_f = s$ if $s$ is a name of $\mathcal{K}$ (exact, case-sensitive), else `invalid`.
(4) Labelling, after these checks in this order (each failing one aborts): `{frames_dir}/run.json` holds its counts (the frames are complete); its `frames.ok` count is present and ≥ 1 (an `ok` frame exists); if `{out_dir}/run.json` exists, the SHA-256 of the config, of `configs/vlm.yaml` and of the catalogue file and the fingerprint of the frames equal its values (the hashes compared first, the first difference aborts; message for the fingerprint: `{frames_dir} changed since labelling started; delete {out_dir}`, paths as written in the config, prefixed like every abort by `mcpref.cli.failure.failed`); then `{out_dir}/run.json` is written by `specs/shards.md` (4) (no `run.json` but label shards present → no check, the shards are kept). Then for $k = h, \dots, H - 1$: plan shard $k$ is read; its `ok` frames, in their order, are asked in one `ask_many` call with the seeds of (2) (no call when it has none); label shard $k$ is written by `specs/shards.md` (1) with their records and `end`, the position after the plan shard's last frame (the state of the next shard; positions continue from `end` of label shard $h - 1$, from 0 when $h = 0$). `no_answer` is final: a rerun never asks again. A rerun continues at label shard $h$; its calls equal those of an uninterrupted run, their answers may differ (AGENTS.md Data).
(4') At the end of every run that exits 0 (also when nothing was left to label), `summary.json` is computed by reading the label shards one at a time and written with a plain write.
(5) Check against VPT (a summary only, never used to change $y_f$): over frames with a class label (not `invalid`, `no_answer`), $g_f$ as stored in its label record; counts of the four pairs ($y_f \in$ {`inventory`, `inventory_book`}, $g_f$).

## Interface
- Module `src/mcpref/gui/labels.py`; `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.gui.labels <config path>`; exit 0, or 1 with a one-line message as the last line of stderr when the frames are not complete, there is no `ok` frame, a file is unreadable, a rerun's hashes or frames differ, `complete_shards` raises (a foreign entry or a gap), or `ask_many` raises.
  - `def load_config(path: Path) -> GuiLabelsConfig`.
  - `def prompt(cfg: GuiLabelsConfig, classes: Sequence[ScreenClass], rgb: np.ndarray) -> list[str | np.ndarray]` — (1).
  - `def parse(answer: str | None, names: Sequence[str]) -> str` — (3); `names` = the names of $\mathcal{K}$, `other` included.
- Consumes: `{frames_dir}/plan/` (read one shard at a time), `{frames_dir}/run.json` (`config.width`, `config.height`, the counts), `{frames_dir}/frames/<id>.yuv` (`specs/vpt-gui-frames.md`); `complete_shards`, `write_shard`, `shard_path`, `write_file` (`specs/shards.md`); `load_screens` (`specs/gui-screens.md`); `load_config`, `ask_many` (`specs/vlm-client.md`).
- Produces (its reader gui-masks is revised in its own cycle after this one; until then `eval_gui_masks.sh` does not run on these outputs) `{out_dir}/labels/<name(k)>/labels.json` for $k = 0, \dots, H - 1$ (shards of `specs/shards.md`): `{"labels": [one object per `ok` frame of plan shard $k$, in its order, with keys `id`, `seed`, `answer` (string or null), `label`, `gui_inventory` ($g_f$)], "end": <the position after plan shard $k$'s last frame>}`; the labels are complete when there are $H$ label shards; `{out_dir}/summary.json`: `{"labels": {label: count}, "vpt_check": {"inventory_flag_true": n, "inventory_flag_false": n, "other_flag_true": n, "other_flag_false": n}}`; `{out_dir}/run.json`: `{"config": <the config>, "sha256": {"config", "vlm", "screens"} (lowercase hex), "frames": {"shards": H, "ok": n, "failed": n}, "prompt": <the prompt (1) with the image replaced by "<image>">}`, written by (4) at the start of every run; `labels` in `summary.json` lists the labels that occurred.
- Datasets (AGENTS.md Code): read: `{frames_dir}/plan/` (one shard at a time, by (4)) and `{frames_dir}/frames/` (one file at a time) of `specs/vpt-gui-frames.md`; built: `labels/` — item: the labels of one plan shard's `ok` frames; one label shard per plan shard, file `labels.json`; written after its `ask_many` call, at once when the plan shard has no `ok` frame; state: `end`; a rerun continues at the first missing label shard; read one shard at a time by (4) (`initial`, `end`), (4') and block gui-masks.
- `src/mcpref/gui/label_files.py`: its helpers used only by block gui-masks (`ok_frames`, `read_jsonl`) stay until gui-masks is revised.
- Script `scripts/label_gui_frames.sh` (preamble of `specs/spark-env.md`): `uv run --locked python -m mcpref.gui.labels configs/gui-labels.yaml`. The server must be running (`scripts/serve_vlm.sh`).
- Progress bar (AGENTS.md Code): `label` (unit `frame`) — the loop of (4) over plan shards; total = the `ok` count in `{frames_dir}/run.json`, `initial` = the records of the complete label shards; advanced by a shard's records once it is written (one step per shard, [U] 2026-10-14, 2026-10-15); no postfix; $h = H$ → the line `label: already done` instead. Reading the complete label shards (for `initial`, `end`) and (4') draw no bar: they read small files and end well under a couple of seconds.
- `pyproject.toml`: adds `tqdm==4.70.1` if absent (shared with blocks vpt-gui-frames, gui-masks).
- Starting config values: `frames_dir` = `data/gui_frames`; `out_dir` = `data/gui_labels`; `seed` = 0; `instruction` = "This is a screenshot of Minecraft 1.16 with a GUI screen open."; `question` = "Which of these screens is open? Answer with the name only, exactly as written.\n{classes}"; `class_line` = "{name}: {description}"; `other_description` = "any other screen, or no screen".
- Tests: (1) on a tiny catalogue (order, placeholders, image position); (2); (3) hand cases (exact, padded, backticks, quotes, case mismatch → `invalid`, `None`); `main` with a fake `ask_many` and fake plan shards and frames (written as by `specs/vpt-gui-frames.md`): one call per plan shard with exactly its `ok` frames and their seeds (positions over all shards, `failed` frames included), no call for a shard without `ok` frames; label shard contents and `end`, a plan shard without frames (`end` unchanged) and one with only `failed` frames; the calls (frames and seeds) of a run interrupted after some shards and continued equal those of an uninterrupted run; an interruption (`ask_many` raising) after some shards, then a rerun continuing at the next shard and never asking again for `no_answer`; abort while `{frames_dir}/run.json` lacks the counts, without an `ok` frame, on changed hashes and on a changed fingerprint (a frame turned `ok`, another $K$ with the same frames); `{out_dir}/run.json` contents, written at the start of every run, also before an interruption; summary counts and (5) table recomputed on every run; the bar's total, `initial` and final count on stderr, and `label: already done`.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1), (4) | `mcpref.vlm.client.ask_many` | this repo | `prompts`, `seeds`, `cfg` | `specs/vlm-client.md` |
| (4), (4') | `mcpref.shards.store` `complete_shards`, `write_shard`, `shard_path`, `write_file` | this repo | — | `specs/shards.md` |
| RGB view | `mcpref.vpt.gui_frames.yuv420p_to_rgb` | this repo | `width`, `height` from the frames' config | `specs/vpt-gui-frames.md` |
| progress bars | `tqdm.tqdm` as a context manager; `update(n)`; `set_postfix_str(s)` | 4.70.1 | `total`, `desc`, `unit`, `initial`; defaults otherwise: `file=None` → `sys.stderr`, `leave=True` (the final state stays, followed by a newline, also when an error leaves the `with` block), `disable=False` (drawn also when stderr is not a terminal); `TQDM_*` environment variables, which could change these defaults, are ignored until they cause a problem ([U] 2026-10-14) | `tqdm/std.py:960-979`, `1144-1151`, `1270-1310` |

## Out of scope
- Several answers per frame, answer probabilities, constrained decoding (Q VLMC-8).
- Using VPT's flag to correct labels.
- Asking again for `no_answer` frames; a rerun after deleting `{out_dir}/run.json` alone (the label shards are then trusted); an `{out_dir}` written before the 2026-10-15 revision (deleted before its first run).
- Choosing between prompt wordings (the prompt is config; the user may change it).


## Decisions of the user (2026-10-10)
- The prompt wording above is accepted.
