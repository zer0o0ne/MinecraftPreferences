# GUI masks
Status: approved (2026-10-10)
Stage: GUI-recognition experiment (Q GX; `specs/00-overview.md`, block 18); the best variant's construction is later rerun on engine frames

## Sources
- [U] the user, 2026-10-10: recognise known GUIs by comparison with templates under masks; three mask variants — (a) template from the game textures with only the outline kept, so items in slots do not spoil it, (b) pixels of low spread over real frames, (c) discriminative patches — compared by agreement with the VLM labels on held-out frames; per-class metrics with counts; an agreement matrix between all labellers per class; the construction of the best variant is rebuilt later on engine frames (Q GX, GX-4, GX-5; decisions below).
- `specs/gui-screens.md` (approved): classes, members, window, `book_panel`, elements, blits, dynamic and label rectangles, slot-content rule; textures config.
- `specs/vpt-gui-frames.md` (approved): `frames.jsonl` (`id`, `relpath`, `line`, `status`), `run.json` (config with `width`, `height`), RGB view (5).
- `specs/gui-labels.md`: `labels.jsonl` (`id`, `label`).
- [N] numpy 1.26.4: `quantile(…, method="inverted_cdf")` returns the $\lceil nq \rceil$-th order statistic in the input dtype (checked by audit, 2026-10-10).

## Notation
- Frames $F$: records of `{frames_dir}/frames.jsonl` with status `ok`, joined by `id` with `{labels_dir}/labels.jsonl`; $x_f \in \{0..255\}^{H \times W \times 3}$ the RGB view ($W, H$ from `{frames_dir}/run.json`); $y_f$ its label; $e_f$ its episode (`relpath`); $g_f$ = `line.isGuiInventory`.
- $\mathcal{K}$ — catalogue classes in catalogue order; `other` — the reject label. Frames with $y_f \in$ {`invalid`, `no_answer`} are left out of building and evaluation, not of the split.
- $d_T(f, p) = \max_{ch} |x_f(p, ch) - T(p, ch)|$, computed in a signed integer type.
- Split: episodes in order of first appearance in `frames.jsonl` (all `ok` records); each gets one draw `rng.choice(3, p=split)` of `numpy.random.Generator(PCG64(seed))`: 0 = train, 1 = validation, 2 = test. $F^{tr}, F^{va}, F^{te}$ — labelled frames of each part; $F^{tr}_k = \{ f \in F^{tr} : y_f = k \}$.
- Qualified classes $\mathcal{Q}$: $k \in \mathcal{K}$ whose $F^{tr}_k$ spans ≥ `min_episodes` distinct episodes.
- Variant $V \in \{a, b, c\}$; for class $k$ a list of (template $T$, mask $M$) pairs — one per member in (a), one in (b), (c).
- $c^V_k(f)$, $|M|$ — the count of matching mask pixels and the mask size of the best pair (below); $S^V_k(f) = c / |M|$ as an exact fraction.
- Config `configs/gui-masks.yaml` → `@dataclass(frozen=True) class GuiMasksConfig`: `frames_dir`, `labels_dir`, `screens` (path of `configs/gui-screens.yaml`), `split` (three decimals summing to 1), `seed`, `min_episodes`, `spread_quantile` ($q$), `spread_max` ($s_{max}$), `patch` ($P$, divides 40), `patches_per_class` ($n_P$, ≤ number of patches), `eps` (> 0), `tau_grid` (integers), `theta_grid`, `mu_grid` (decimal strings, read as exact fractions), `chunk_rows` (image rows processed at once in (2)), `out_dir`. Violated constraints → `ValueError` at load.

## Math
(1) Variant (a), per member $m$ of $k$ ([U]: outline only): $T$ = the member's blits composited in draw order (each texture converted to RGBA, source-over) onto a transparent $H \times W$ canvas; $M$ = pixels with canvas alpha 255, inside $k$'s window or `book_panel`, outside every element hit rectangle, every slot-content rectangle $(x_s, y_s, 17, 18)$, and every dynamic and label rectangle of any member of $k$. Members with an empty $M$ are skipped; a class with no pair left has no template.
(2) Variant (b), for $k \in \mathcal{Q}$ ([U]: low spread; quantile chosen 2026-10-10): $T(p, ch)$ = `quantile` over $F^{tr}_k$ of $x_f(p, ch)$ at 0.5, `inverted_cdf` (the lower median, `uint8`) [N]; $\sigma_k(p)$ = `quantile` over $F^{tr}_k$ of $d_T(f, p)$ at $q$, `inverted_cdf`; $M = \{ p : \sigma_k(p) \le s_{max} \}$. Computed over row blocks of `chunk_rows` rows.
(3) Variant (c), for $k \in \mathcal{Q}$, $|\mathcal{Q}| \ge 2$: patches $Q$ = the non-overlapping $P \times P$ squares tiling the frame from (0, 0); $\bar\sigma_k(Q)$ = mean of $\sigma_k$ over $Q$; $\delta_k(Q) = \min_{j \in \mathcal{Q}, j \ne k}$ mean over $Q$ of $\max_{ch} |T_k - T_j|$; $\rho_k(Q) = \delta_k(Q) / (\bar\sigma_k(Q) + \varepsilon)$ in float64; $M$ = union of the $n_P$ patches with the largest $\rho_k$ (ties: smaller top row, then smaller left column); $T$ = (b)'s template.
(4) Score: for each pair $(T, M)$ of $k$, $c = |\{ p \in M : d_T(f, p) \le \tau \}|$; the pair with the largest $c / |M|$ (exact) is the best pair; $S^V_k(f) = c / |M|$. Classes without a template have no score.
(5) Prediction: if no class has a score, $\hat y_f$ = `other`; else $k^*$ = the class with the largest $S_k$ (ties: catalogue order), $S_{(2)}$ = the second-largest score (0 if only one); $\hat y_f = k^*$ if $S_{k^*} \ge \theta$ and $S_{k^*} - S_{(2)} \ge \mu$ (exact fractions), else `other`.
(6) Thresholds per variant, two choices, each the grid point of `tau_grid` × `theta_grid` × `mu_grid` (τ outermost, μ innermost) maximising a criterion on $F^{va}$ (ties: the first in that order): `agreement` $= \frac{1}{|F^{va}|}\sum_f \mathbb{1}[\hat y_f = y_f]$; `macro_recall` = mean, over the labels present in $F^{va}$, of the share of their frames predicted correctly.
(7) Evaluation on $F^{te}$, per variant and threshold choice: agreement and macro recall; per label $k \in \mathcal{K} \cup$ {`other`}: count with $y_f = k$, count with $\hat y_f = k$, precision, recall, F1 (`null` when the denominator is 0), the ids of misclassified frames; the confusion matrix ($y$ × $\hat y$); the same split by $g_f$ true / false.
(8) Agreement between labellers, per threshold choice: labellers VLM, (a), (b), (c); for each $k \in \mathcal{K}$: $F_k$ = frames of $F^{te}$ to which at least one labeller assigns $k$; $A_k[i, j] = |\{ f \in F_k : L_i(f) = L_j(f) \}| / |F_k|$, reported with $|F_k|$ (no matrix when $F_k$ is empty).

## Interface
- `src/mcpref/gui/masks.py` — (1)–(3):
  - `def texture_pairs(cls: ScreenClass, textures_dir: Path, width: int, height: int) -> list[tuple[np.ndarray, np.ndarray]]` — (1).
  - `def variance_pair(frames: Sequence[np.ndarray], q: float, s_max: int, chunk_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — (2): template (`uint8`), spread, mask (`bool`).
  - `def patch_masks(templates: Mapping[str, np.ndarray], spreads: Mapping[str, np.ndarray], patch: int, n_p: int, eps: float) -> dict[str, np.ndarray]` — (3).
- `src/mcpref/gui/recognise.py` — (4)–(5): `def scores(x: np.ndarray, pairs: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]], tau: int) -> dict[str, Fraction]`; `def predict(scores: Mapping[str, Fraction], order: Sequence[str], theta: Fraction, mu: Fraction) -> str`.
- `src/mcpref/gui/evaluate.py` — config, split, (6)–(8), outputs: `GuiMasksConfig`; `def load_config(path: Path) -> GuiMasksConfig`; `def split_episodes(relpaths: Sequence[str], probs: Sequence[float], seed: int) -> dict[str, int]`; `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.gui.evaluate <config path>`; exit 0, or 1 with a one-line message on stderr.
- Consumes: `load_screens` and the textures directory (`textures.out_dir`) of `configs/gui-screens.yaml`; `yuv420p_to_rgb`; the two `.jsonl` files and `run.json` above.
- Produces `{out_dir}/`:
  - `templates/<variant>/<class>[__<member>]_template.png` (RGB; canvas pixels with alpha < 255 written black) and `…_mask.png` (8-bit grey, 255 = in mask) — for inspection; the format for env is decided when env is specified;
  - `report.json`: `split` (per part: episodes, frames, frames per label), `qualified` (classes of $\mathcal{Q}$), and per variant: `pairs` (per class: number of pairs, mask sizes), and per threshold choice `agreement` / `macro_recall`: `thresholds` {tau, theta, mu}, `test` {agreement, macro_recall, per_label {n_true, n_pred, precision, recall, f1, errors}, confusion, by_flag {true, false} with the same keys}; `agreement_matrices` per threshold choice: per class {n, labellers, matrix};
  - `report.md`: the same as tables.
- Script `scripts/eval_gui_masks.sh` (preamble of `specs/spark-env.md`): `uv run --locked python -m mcpref.gui.evaluate configs/gui-masks.yaml`. The VLM server need not run; stopping it frees its memory.
- Starting config values: `frames_dir` = `data/gui_frames`, `labels_dir` = `data/gui_labels`, `screens` = `configs/gui-screens.yaml`, `split` = ["0.6", "0.2", "0.2"], `seed` = 0, `min_episodes` = 10, `spread_quantile` = 0.9, `spread_max` = 24, `patch` = 8, `patches_per_class` = 64, `eps` = 1, `tau_grid` = [8, 16, 24, 32, 48], `theta_grid` = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95"], `mu_grid` = ["0", "0.05", "0.1", "0.2"], `chunk_rows` = 40, `out_dir` = `data/gui_masks`.
- Tests: (1) on a synthetic class with tiny RGBA and palette textures (compositing, every subtraction, union over members); (2), (3) against naive float64 loops on small synthetic frames, with values near 0 and 255 (no `uint8` wrap-around), even and odd counts; (4)–(5) hand cases (exact-fraction boundaries such as 3/10 − 1/10 vs μ = 1/5, ties, skipped pairs, no score); (6) on a tiny grid with ties; the split (an episode never in two parts, stable when labels change); `main` end-to-end on synthetic frames, labels and catalogue, checking `report.json` keys and values.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1) | `PIL.Image.open(…).convert("RGBA")`, `Image.alpha_composite` | 12.3.0 | RGBA in both; textures' alpha is 0 or 255 except two pixels inside a hit rectangle (audit, 2026-10-10), so the blend mode cannot change a masked pixel | to vet at implementation |
| (2) | `numpy.quantile` | 1.26.4 | `axis=0`, `method="inverted_cdf"` | [N] |
| split | `numpy.random.Generator(PCG64(seed)).choice` | 1.26.4 | `a=3`, `p=split` | to vet at implementation |
| (4)–(6) | `fractions.Fraction` | Python 3.10 stdlib | — | — |
| PNG output | `PIL.Image.fromarray(…).save(…, format="PNG")` | 12.3.0 | — | — |

## Out of scope
- Rebuilding on engine frames (the same code later run on other frames and labels).
- Templates for `other`; recognising windows at other offsets or scales.
- Correcting VLM labels.

## Decisions of the user (2026-10-10)
- (a) uses the textures' own colours; the tolerance $\tau$ absorbs compression and colour conversion; a colour-corrected template only if (a) classifies badly.
- (2) spread is a quantile of the per-pixel distance (robust to tooltips and held stacks).
- (6) both threshold choices are reported with per-class metrics; the choice is made from the report.
- VLM labels are the reference; (8) shows where the methods agree among themselves against the VLM.
