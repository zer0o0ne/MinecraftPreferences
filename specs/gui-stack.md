# GUI stack
Status: approved (2026-10-14; clarified after the prototype the same day: library rows, `seed` range, `VARIANTS` / `CRITERIA` of gui-masks, approved)
Stage: GUI-recognition experiment (Q GX; `specs/00-overview.md`, block 19), on top of block gui-masks

## Sources
- [U] the user, 2026-10-14: a classifier trained on top of all mask methods, taking one number per template (the quantile of its deviation over its mask) and giving the probability of each class; an MLP with one hidden layer, in torch; the loss weighted by class; no thresholds or calibration: the class is the argmax of the softmax; trained on the (b), (c) features cross-fitted over episode folds, so that the rare classes keep all their frames; run from `scripts/eval_gui_masks.sh`. The remaining choices: Decisions below.
- `specs/gui-masks.md` (revision of 2026-10-14): `features.npz` (14), `evaluation` (7), `matrices` (8), `write_report`.
- [T] torch 2.14.1 (PyPI, released 2026-09-30, not yanked; `cp310` wheels for `manylinux_2_28` x86_64 and aarch64, CUDA 13 on Linux), read at the tag `v2.14.1`: `torch/optim/adam.py:34-123` (arguments and defaults), `:215-345` (`step`), `:347-552` (`_single_tensor_adam`), `:553-801` (`_multi_tensor_adam`, the same update), `:902-990` (`adam`: choice of implementation through `torch/optim/optimizer.py:170-195`; fused never by default); in both, the L2 term enters the gradient only when `weight_decay` ≠ 0, $m \leftarrow \beta_1 m + (1 - \beta_1) \nabla$, $v \leftarrow \beta_2 v + (1 - \beta_2) \nabla^2$, $\theta \leftarrow \theta - \eta \frac{m / (1 - \beta_1^t)}{\sqrt{v} / \sqrt{1 - \beta_2^t} + \varepsilon}$; `torch/nn/functional.py:2293-2324` (`log_softmax` → `Tensor.log_softmax(dim)`); `torch/serialization.py` (`torch.load` defaults to `weights_only=True`, which loads tensors, lists, dicts and strings).

## Notation
- From `{masks_dir}/features.npz` of gui-masks: columns $i = 1..C$; labels $\mathcal{Y}$ = $\mathcal{K}$ in catalogue order, then `other` (`labels`); per frame $f$: $x_f \in (\mathbb{R} \cup \{\text{NaN}\})^C$, $y_f \in \mathcal{Y}$, $g_f$, id.
- $S$ — the classifier's training frames: the `train` frames, then the `validation` frames, in file order; $F^{te}$ — the `test` frames.
- $n_y = |\{ f \in S : y_f = y \}|$; $w_f = 1 / n_{y_f}$ for $f \in S$.
- $\mu_i$ — the mean of $x_{f,i}$ over the $f \in S$ where it is not NaN; $s_i$ — 0 if those values are all equal (exact comparison), else their population standard deviation (divisor = count); $z_{f,i} = (x_{f,i} - \mu_i) / s_i$ if $x_{f,i}$ is not NaN and $s_i > 0$, else 0 (a NaN takes the mean).
- $\theta = (W_1 \in \mathbb{R}^{H \times C}, b_1 \in \mathbb{R}^H, W_2 \in \mathbb{R}^{|\mathcal{Y}| \times H}, b_2 \in \mathbb{R}^{|\mathcal{Y}|})$; $h$ — the hidden layer, $\ell$ — the logits (1); $u$ — a uniform draw and fan_in — the input size of a layer (2); $t = 1..T$ — the step, $\nabla_t$ — the gradient of (3) at step $t$, $m_t$, $v_t$ — Adam's moments (4).
- Config `configs/gui-stack.yaml` → `@dataclass(frozen=True) class GuiStackConfig`: `masks_dir` (out_dir of gui-masks), `hidden` ($H$, integer ≥ 1), `l2` ($\lambda$, ≥ 0), `lr` ($\eta$, > 0), `betas` (exactly two: $\beta_1, \beta_2$, each in [0, 1)), `adam_eps` ($\varepsilon$, > 0), `steps` ($T$, integer ≥ 1), `seed` (integer, $0 \le$ `seed` $< 2^{64}$, the range of `torch.Generator.manual_seed`), `device` (a string accepted by `torch.device`), `out_dir`. Paths relative to the repository root, resolved against it. Integers must be YAML integers (not booleans); `l2`, `lr`, `adam_eps` and the betas are read with `float()`. Violated constraints → `ValueError` at load.

## Math
(1) Network ([U]: one hidden layer): $h = \max(0, W_1 z + b_1)$; $\ell = W_2 h + b_2$; $p_\theta(\cdot \mid z) = \text{softmax}(\ell)$ over $\mathcal{Y}$.
(2) Initialisation: one `torch.Generator` on the CPU seeded with `seed`; $W_1, b_1, W_2, b_2$ drawn in this order, each $(2u - 1) / \sqrt{\text{fan\_in}}$ with $u$ = `torch.rand` of its shape (float32), fan_in $= C$ for $W_1, b_1$ and $H$ for $W_2, b_2$; then moved to `device`.
(3) Loss ([U]: weighted by class): $L(\theta) = \frac{\sum_{f \in S} w_f \,(-\log p_\theta(y_f \mid z_f))}{\sum_{f \in S} w_f} + \lambda \left( \|W_1\|_F^2 + \|W_2\|_F^2 \right)$, $\log p$ by `log_softmax` over $\mathcal{Y}$.
(4) Optimisation: $\theta_0$ of (2), $m_0 = v_0 = 0$; for $t = 1..T$, on all of $S$ (full batch): $\nabla_t = \nabla_\theta L(\theta_{t-1})$, $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_t$, $v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla_t^2$, $\theta_t = \theta_{t-1} - \eta \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t} / \sqrt{1 - \beta_2^t} + \varepsilon}$ [T]; no schedule, no early stopping; the result is $\theta_T$. float32.
(5) Prediction for $f \in F^{te}$ ([U]: no thresholds): $p_f = p_{\theta_T}(\cdot \mid z_f)$; $\hat y_f$ = the first label of $\mathcal{Y}$ with the largest $p_f$.
(6) Evaluation on $F^{te}$: gui-masks (7) with $\hat y_f$; gui-masks (8) per threshold choice (`agreement`, `macro_recall`) with labellers VLM, (a)–(f) (their `test_pred_<variant>_<choice>`) and `stack` (the same $\hat y$ under both choices).

## Interface
- `src/mcpref/stacking/config.py`: `GuiStackConfig`; `def load_config(path: Path) -> GuiStackConfig`.
- `src/mcpref/stacking/model.py` — (1)–(5):
  - `def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — $\mu$, $s$ over the rows of $x$ ($|S|$ × $C$).
  - `def standardised(x: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray` — $z$.
  - `def init_params(c: int, h: int, n_labels: int, seed: int, device: torch.device) -> list[torch.Tensor]` — (2): $[W_1, b_1, W_2, b_2]$, requiring grad.
  - `def logits(params: Sequence[torch.Tensor], z: torch.Tensor) -> torch.Tensor` — (1): $\ell$, frames × $|\mathcal{Y}|$.
  - `def loss(params: Sequence[torch.Tensor], z: torch.Tensor, y: torch.Tensor, w: torch.Tensor, l2: float) -> torch.Tensor` — (3); `y` label indices in $\mathcal{Y}$ order, `w` the $w_f$.
  - `def fit(params: Sequence[torch.Tensor], z: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cfg: GuiStackConfig, step: Callable[[], None]) -> tuple[float, float]` — (4) in place; returns $L(\theta_0)$, $L(\theta_T)$; calls `step()` after each Adam update (bar `stack fit`).
  - `def probabilities(params: Sequence[torch.Tensor], z: torch.Tensor) -> np.ndarray` — (5): $p_f$, frames × $|\mathcal{Y}|$, softmax computed in float64.
- `src/mcpref/stacking/run.py` — (5) labels, (6), `main`: `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.stacking.run <config path>`; exit 0, or 1 with a one-line message as the last line of stderr for the failures gui-masks turns into one line (`mcpref.cli.failure.FAILURES`), a features file missing or without a key this block reads (`columns`, `labels`, `train_x`, `train_y`, `validation_x`, `validation_y`, `test_ids`, `test_y`, `test_g`, `test_x`, the twelve `test_pred_*`), $C = 0$, or $S$ empty.
- Consumes: `{masks_dir}/features.npz` (gui-masks Produces); `mcpref.recognition.metrics.evaluation`, `mcpref.recognition.metrics.matrices`, `mcpref.recognition.metrics.CRITERIA`, `mcpref.recognition.pairs.VARIANTS`, `mcpref.recognition.report.write_report` (gui-masks Interface); the keys `test_pred_<variant>_<choice>` are built from `VARIANTS` × `CRITERIA`.
- Produces `{out_dir}/`:
  - `model.pt` (`torch.save`, loadable by `torch.load(path)` with its defaults): `{"params": [W1, b1, W2, b2]` (float32, detached, on the CPU), `"mu": μ, "s": s` (float64 tensors), `"columns": [...], "labels": [...]` (lists of str)`}`;
  - `predictions.jsonl`: one record per frame of $F^{te}$ in file order: `id`, `label` ($y_f$), `predicted` ($\hat y_f$), `p` ({label: probability} over $\mathcal{Y}$);
  - `report.json` (top-level keys `train`, `test`, `agreement_matrices`; fractions as JSON floats): `train` {frames: $|S|$, per_label: $n_y$ over $\mathcal{Y}$ (zeros included), loss_initial: $L(\theta_0)$, loss_final: $L(\theta_T)$}, `test` = `evaluation` (gui-masks `test` keys), `agreement_matrices` = `matrices`; `report.md`: the same as tables, titled `GUI stack report`.
- Script `scripts/eval_gui_masks.sh` (block gui-masks): this block's cycle appends `uv run --locked python -m mcpref.stacking.run configs/gui-stack.yaml` after the gui-masks line.
- `pyproject.toml`: adds `torch==2.14.1`.
- Progress bar (AGENTS.md Code): `stack fit` (unit `step`) — the loop of (4); total $T$ (≥ 1, so the bar is always drawn); advanced after each Adam update; no postfix.
- `specs/00-overview.md`: block 19 gui-stack; package map `stacking` → GUI-recognition experiment → this spec.
- Starting config values: `masks_dir` = `data/gui_masks`, `hidden` = 64, `l2` = 0.0001, `lr` = 0.01, `betas` = [0.9, 0.999], `adam_eps` = 1.0e-8, `steps` = 2000, `seed` = 0, `device` = `cpu`, `out_dir` = `data/gui_stack`.
- Tests: $\mu$, $s$, $z$ on hand arrays (NaN; a constant column of non-integers such as 0.1, where `nanstd` is not exactly 0); the config loader (types, `1e-8` as a string, two betas, a bad device); (2) shapes, bounds $\pm 1/\sqrt{\text{fan\_in}}$, the same values for the same seed; (3) against a float64 numpy oracle on tiny data, with a label absent from $S$; one step of (4) against a numpy oracle of the Adam update; (4) on a tiny separable set reaching no training error; (5) ties; `main` end-to-end on a synthetic `features.npz` (`report.json` keys and values, `predictions.jsonl`, `model.pt` loaded by `torch.load(path)`, NaN columns, $C = 0$, the bar's total and final count); `scripts/eval_gui_masks.sh` runs gui-masks, then gui-stack, and stops when gui-masks fails (fake `uv`).

## Library mapping
| Eq. | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (2) | `torch.Generator(device="cpu").manual_seed`, `torch.rand` | 2.14.1 | `generator`, `dtype=torch.float32` | to vet at implementation (uniform on [0, 1)) |
| (1) | `@`, `torch.relu` | 2.14.1 | — | — |
| (3) | `torch.nn.functional.log_softmax` | 2.14.1 | `dim=1` | [T] `functional.py:2293-2324` |
| (4) | `torch.optim.Adam` | 2.14.1 | `lr`, `betas` (floats), `eps`, `weight_decay=0`, `amsgrad=False`, `maximize=False`, `decoupled_weight_decay=False`; `foreach` / `fused` left to the library (the same update) | [T] `adam.py` |
| (4) | `Tensor.backward`, `Optimizer.zero_grad` | 2.14.1 | — | to vet at implementation |
| (5) | `torch.softmax` on float64 logits | 2.14.1 | `dim=1` | to vet at implementation |
| $\mu$, $s$ | `numpy.nanmean`, `numpy.nanstd` | 1.26.4 | `axis=0`, `ddof=0` | `numpy/lib/nanfunctions.py:1040-1053`, `1720-1770` (divisor = count − ddof) |
| $s_i = 0$ test | `numpy.nanmax`, `numpy.nanmin` | 1.26.4 | `axis=0` | `numpy/lib/nanfunctions.py:340-366` (NaN ignored) |
| $z$ | `numpy.divide` | 1.26.4 | `out` = zeros, `where` = (not NaN and $s_i > 0$) | to vet at implementation |
| (3) $\log p_\theta(y_f \mid z_f)$ | `Tensor.gather` | 2.14.1 | `dim=1`, `index=y[:, None]` | to vet at implementation |
| (5) $\hat y_f$ | `numpy.argmax` | 1.26.4 | `axis=1` (the first maximum) | to vet at implementation |
| numpy ↔ torch | `torch.from_numpy`, `Tensor.cpu().numpy()` | 2.14.1 | — | to vet at implementation (the wheel's numpy bridge with numpy 1.26.4) |
| model file | `torch.save` | 2.14.1 | — | [T] `serialization.py` |
| features | `numpy.load` | 1.26.4 | `allow_pickle=False` | — |
| progress bar | `tqdm.tqdm` as in `specs/gui-masks.md` | 4.70.1 | `total`, `desc`, `unit` | as there |

## Out of scope
- Choosing hyperparameters on held-out data; thresholds, calibration or a correction of the class prior of $p$ (with the weights of (3), $p$ reflects equal class priors).
- Mini-batches, dropout, normalisation layers, other architectures.
- GPU-specific code paths beyond `device`; GPU tests.

## Decisions of the user (2026-10-14)
- MLP with one hidden layer in torch; class weights; argmax without thresholds ([U]).
- $S$ = cross-fitted `train` plus `validation`; a NaN takes the column mean; the initialisation of (2); L2 on $W_1, W_2$ only, and the weighted mean normalised by $\sum w_f$; Adam on the full batch for a fixed $T$; hyperparameters fixed in the config, none chosen on held-out data.
