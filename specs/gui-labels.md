# GUI labels
Status: approved (2026-10-10)
Stage: GUI-recognition experiment (Q GX; `specs/00-overview.md`, block 17)

## Sources
- [U] the user, 2026-10-10: the screen type of every sampled frame is labelled by the VLM; the labels are the reference for comparing mask variants (Q GX); VPT's `isGuiInventory` is an independent check (Q GX-3); every Spark action is a bash script.
- `specs/vpt-gui-frames.md` (approved): frames, `frames.jsonl` records, RGB view (5).
- `specs/gui-screens.md` (approved): classes, their one-line `description`.
- `specs/vlm-client.md` (approved): `ask_many`, `load_config`.

## Notation
- $f$ — a frame with status `ok` in `{frames_dir}/frames.jsonl`; $x_f$ — its RGB view (vpt-gui-frames (5)).
- $\mathcal{K}$ — the catalogue's classes in catalogue order, plus `other`.
- $q_f$ — the prompt of $f$; $a_f$ — the VLM's answer (string or `None`); $y_f$ — the label.
- Config `configs/gui-labels.yaml` → `@dataclass(frozen=True) class GuiLabelsConfig`: `frames_dir`, `screens` (path of `configs/gui-screens.yaml`), `vlm` (path of `configs/vlm.yaml`), `instruction` (text before the image), `question` (text after the image; contains the placeholder `{classes}`), `class_line` (format of one class line, placeholders `{name}`, `{description}`), `other_description`, `seed`, `batch` (prompts per `ask_many` call), `out_dir`.

## Behaviour
(1) Prompt: $q_f$ = [`instruction`, $x_f$, `question` with `{classes}` replaced (`str.replace`) by one `class_line` per class of $\mathcal{K}$ in order, joined by `"\n"`, each with `{name}` and `{description}` replaced (`str.replace`); `other` uses `other_description`]. $x_f$ = `yuv420p_to_rgb` with `width`, `height` from the config stored in `{frames_dir}/run.json`.
(2) Seed of $f$: `seed` + the position of $f$ in `frames.jsonl` (0-based, over all records, `failed` ones included).
(3) Label: `no_answer` if $a_f$ is `None`; else $s$ = $a_f$ stripped of whitespace; if $s$ has length ≥ 2 and starts and ends with the same character among `` ` ``, `'`, `"`, those two characters are removed and $s$ is stripped again; $y_f = s$ if $s$ is a name of $\mathcal{K}$ (exact, case-sensitive), else `invalid`.
(4) Frames still to label (no record yet, or `no_answer`) in `frames.jsonl` order, in batches of `batch` (the last may be smaller), one `ask_many` call per batch with the seeds of (2) (a retried frame keeps its seed); after every batch the output files are rewritten atomically (temporary file, then rename) and hold only finished frames. A rerun aborts (exit 1) if the SHA-256 of the config, of `configs/vlm.yaml` or of the catalogue file differs from the values in `run.json`.
(5) Check against VPT (a summary only, never used to change $y_f$): over frames with a class label (not `invalid`, `no_answer`), $g_f$ = `line.isGuiInventory` of its record; counts of the four pairs ($y_f \in$ {`inventory`, `inventory_book`}, $g_f$).

## Interface
- Module `src/mcpref/gui/labels.py`; `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.gui.labels <config path>`; exit 0, or 1 with a one-line message on stderr when a file is unreadable, a rerun's hashes differ, or `ask_many` raises.
  - `def load_config(path: Path) -> GuiLabelsConfig`.
  - `def prompt(cfg: GuiLabelsConfig, classes: Sequence[ScreenClass], rgb: np.ndarray) -> list[str | np.ndarray]` — (1).
  - `def parse(answer: str | None, names: Sequence[str]) -> str` — (3).
- Consumes: `{frames_dir}/frames.jsonl`, `{frames_dir}/run.json`, `frames/<id>.yuv` (`specs/vpt-gui-frames.md`); `load_screens` (`specs/gui-screens.md`); `load_config`, `ask_many` (`specs/vlm-client.md`).
- Produces `{out_dir}/labels.jsonl`, one record per finished `ok` frame in `frames.jsonl` order: `id`, `seed`, `answer`, `label`; `{out_dir}/summary.json`: `{"labels": {label: count}, "vpt_check": {"inventory_flag_true": n, "inventory_flag_false": n, "other_flag_true": n, "other_flag_false": n}}`; `{out_dir}/run.json`: the config, the three SHA-256 values of (4), and the prompt of the first `ok` frame of `frames.jsonl` with the image replaced by `"<image>"`.
- Script `scripts/label_gui_frames.sh` (preamble of `specs/spark-env.md`): `uv run --locked python -m mcpref.gui.labels configs/gui-labels.yaml`. The server must be running (`scripts/serve_vlm.sh`).
- Starting config values: `frames_dir` = `data/gui_frames`; `out_dir` = `data/gui_labels`; `seed` = 0; `batch` = 256; `instruction` = "This is a screenshot of Minecraft 1.16 with a GUI screen open."; `question` = "Which of these screens is open? Answer with the name only, exactly as written.\n{classes}"; `class_line` = "{name}: {description}"; `other_description` = "any other screen, or no screen".
- Tests: (1) on a tiny catalogue (order, placeholders, image position); (2); (3) hand cases (exact, padded, backticks, quotes, case mismatch → `invalid`, `None`); `main` with a fake `ask_many` (rerun skips, retries `no_answer`, summary counts, (5) table) and fake frames.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1), (4) | `mcpref.vlm.client.ask_many` | this repo | `prompts`, `seeds`, `cfg` | `specs/vlm-client.md` |
| RGB view | `mcpref.vpt.gui_frames.yuv420p_to_rgb` | this repo | `width`, `height` from the frames' config | `specs/vpt-gui-frames.md` |

## Out of scope
- Several answers per frame, answer probabilities, constrained decoding (Q VLMC-8).
- Using VPT's flag to correct labels.
- Choosing between prompt wordings (the prompt is config; the user may change it).


## Decisions of the user (2026-10-10)
- The prompt wording above is accepted.
