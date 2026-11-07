# Pretrain diagnostics
Status: approved (2026-11-07); 2026-11-05: complete diagnostic signatures and shared `global-fit.json` initialization/resume lifecycle, approved; 2026-11-05: required keyword-only `PretrainConfig` in `step_gradients`, approved; 2026-11-06: required ordered `Groups` in `reduce_work`, approved; 2026-11-07: legacy/new fixed-reference pair layouts and stage-33a outbox handoff, approved for drafting
Stage: revision of block 7 of `specs/00-overview.md` — diagnostics of the existing pretraining loop; no change to its training mathematics

## Sources
- [U] the user, 2026-11-07: approve `specs/rl-diagnostics.md` stage-33a re-clarification. Existing reference runs without shard-0 `pairs.json` remain metrics-only without backfill; new runs store typed pairs in every reference shard. Producer outboxes allow crash recovery from accepted checkpoints/reference shards without another model pass.
- [U] the user, 2026-11-05: approve the full diagnostic block: low-overhead per-step and 100-step logging plus fixed-reference diagnostics every 500 steps with 64 train and 64 validation pairs and `joint` / `P_only` / `phi_only` / `initial` evaluation. Do not run another full training job for diagnosis.
- [U] the user, 2026-11-05, on the analysis: separate the two preference heads and the components of $\phi$ and $\pi$; show distributions, soft-target calibration, gradient/update activity and data counts; compare preference losses with constants fitted once on the global train population, never refitted on each window; distinguish movement of $P$ from movement of $\phi$ without changing training.
- [U] the user, 2026-11-05: approve all 17 audit resolutions, including comparable-only conditional `succ` suites, the persisted strata catalogue, two-pass shard-bounded selection, typed observation/reference interfaces, exact aggregate/coverage schemas, configured-device RNG handling and `checkpoint_steps=100`.
- [U] the user, 2026-11-05, expanding the work order through `data/analysis/preference-gradient-flow/analytics-handoff.md`: instrument selected real pretraining steps with per-micro-batch component gradients, step-summed dot/cosines, actual AdamW-update alignment, and post-step counterfactual preference losses on exactly the same labelled pairs. Preserve training updates, schedule, RNG consumption and stored training results. Synthetic probes cannot close the dynamic hypotheses.
- [U] the user, 2026-11-05: approve all eight remaining audit resolutions: nullable infinite `copy_current`, shared post-step evaluation state, zero-based pair indices, nullable undefined update tensors, bundle-local alignment reasons, canonical module-qualified parameter names, `out_dir`-relative evidence paths and null `GlobalFit.sum_p` without comparable train pairs.
- [U] the user, 2026-11-05: approve complete `preference_suite` and `WindowAccumulator` signatures and the shared `{out_dir}/global-fit.json` initialization artifact, with no partial reference shard 0 and byte-identical `reference/000000/baselines.json` on publication.
- [U] the user, 2026-11-05: approve `step_gradients(..., micro_batch: int, *, cfg: PretrainConfig)` with all previous positional parameters preserved, no default, overload or compatibility mode, and `TypeError` when `cfg` is omitted.
- [U] the user, 2026-11-06: approve `reduce_work(directory: Path, *, groups: Groups) -> dict`, with the ordered groups from `check_groups`, no JSON schema change and no spec-level progress argument.
- `specs/pretrain.md` (approved): the loop, its mini-batch and micro-batch reductions, clipping, AdamW, checkpoints and `losses.json`; this block adds observations to (T), not another objective or update.
- `specs/preference.md` (approved): $(s,t)$ and $\ell^P$; `specs/policy.md` (approved): $\ell^\pi$, its targets and outputs; `specs/perception.md` (approved): $\ell^\phi$, $d_\alpha$, $q_s$, $\chi_s$ and the norm hinge.
- `specs/train-index.md` (approved): train and validation index shards, labels and episode-linked split. This block is the first consumer of validation by the pretraining loop.
- `specs/shards.md` (approved): atomic shards and files; `specs/runs.md` (approved): JSON and fingerprint conventions.
- [T] torch 2.14.1 and numpy 1.26.4, as vetted in the consumed specs; additional calls are listed in Library mapping.

## Notation
- All notation of `specs/pretrain.md` is retained. In particular, $t$ is the optimizer-step number, $T$ the total number of steps, $K=K(\mathcal B)$ the clips of a mini-batch, $R=R(\mathcal B)$ its pairs, $L$ the states per clip, $D$ the latent size, $G$ the parameter groups and $g_{max}$ `grad_clip`.
- $W_d$ — `diagnostic_window` = 100; $T_r$ — `reference_steps` = 500; $R_r$ — `reference_pairs` = 64 per part; $B_c$ — `calibration_bins` = 10; $\tau_s$ — `saturation_threshold` = 0.01; $\tau_i$ — `inc_classification_threshold` = 0.5; $Q_d$ — `quantiles` = (0.05, 0.5, 0.95); $\epsilon_c$ — `action_copy_eps` = 0.1.
- A **pair population** is a finite sequence of pairs with labels $y_r=(p_r,i_r)$ and logits $(s_r,t_r)$. Here $i_r=$ `inc` and $C=\{r:i_r=0\}$. Loss/baseline weights $w_r$ sum to 1 over all pairs: per step $w_r=1/|R|$, for a fixed-reference part $w_r=1/R_r$. Output-suite weights are defined separately in (5). A step with $|R|=0$ has no pair population.
- A **clip population** is a finite sequence of clip occurrences with weights summing to 1. A pair contributes two clip occurrences, an unlabeled item one; repetitions are retained. Per step each clip has weight $1/|K|$.
- $\sigma(x)=1/(1+e^{-x})$; $\mathrm{BCE}(x,y)=\max(x,0)-xy+\log(1+e^{-|x|})$. $\log$ is natural. The convention $0\log0=0$ is used in every entropy and baseline below.
- For normalized weights $w_r$, $\bar x=\sum_rw_rx_r$ and $\mathrm{Var}(x)=\sum_rw_r(x_r-\bar x)^2$; `std` is $\sqrt{\mathrm{Var}(x)}$, with no Bessel correction. `mean`/`std` over an empty population are null.
- $N_c=|C|$, $N_i=|R|-|C|$ in an unweighted population; `count` is an integer count before weighting. For the six current comparable strata and any future soft target, the comparable stratum key is the decimal JSON value of $p_r$; the incomparable stratum key is `"inc"`.
- $q^\pi_{i,s,a}$, $\chi^\pi_{i,s}$ and $C_{i,s,a}$ — the policy target, shift validity and executed-action input weights of `specs/policy.md` (1), (8), for clip $i$, state $s=0..L-3$ and action $a$. $\lambda^\pi_{i,s,a}$ are the policy logits and $\pi_{i,s,a}=\mathrm{softmax}_a(\lambda^\pi_{i,s})$.
- For $\phi$, $d_{i,s}$, $q_{i,s}$, $\chi_{i,s}$, $h_{i,s}$ and $\log\lambda$ have the meanings of `specs/perception.md` (10)–(11). $z\_norm_{i,s}=\|z_{i,s}\|_2=\sqrt{\sum_dz_{i,s,d}^2}$ and $h_{i,s}=\max(0,z\_norm_{i,s}^2-c^2)$. All four loss components below retain the stop-gradients of that equation.
- `null` is JSON null. JSON NaN and infinities are forbidden: a metric whose mathematical value is undefined or infinite is null. Counts and `active` flags are never null. `mean_nullable` returns the mean over non-null values and their count, or `(null,0)` when there are none.
- `train` and `validation` are the two index parts. Reference pair ordinal $j=0..2R_r-1$ follows the deterministic selection order: all train pairs, then all validation pairs, each in the order of (17).

## Math

### Loss terms and scalar observations

(1) **Training invariance [U].** The forward calls, their order, random draws, loss values and dtypes, backward calls, gradient accumulation, clipping, learning rates, AdamW updates, checkpoint schedule and training schedule remain exactly those of `specs/pretrain.md`. Diagnostics use detached values from the same forward and backward passes. They add no training forward/backward, stop-gradient, loss weight or optimizer operation. Component extraction replaces, rather than duplicates, the body of each existing loss `forward`. Each `terms` result also carries `total`, computed by the original reduction and operation order; `forward` returns `total`, so separate diagnostic reductions cannot alter even floating-point training results.

(2) **Preference terms.** For pair $r$:
$$\ell^{P}_{r,inc}=2\mathrm{BCE}(t_r,i_r),\qquad
\ell^{P}_{r,succ}=2[i_r=0]\mathrm{BCE}(s_r,p_r),\qquad
\ell^P_r=\ell^{P}_{r,inc}+\ell^{P}_{r,succ}.$$
The logged step values are $\sum_rw_r\ell^P_{r,inc}$ and $\sum_rw_r\ell^P_{r,succ}$. With pairs but $C=\emptyset$, `P.succ=0`; with no pairs, both and `P` are null. This additive loss reduction is independent of the conditional output-suite reduction in (5).

(3) **Policy terms.** For clip $i$:
$$\ell^\pi_{i,action}=\frac1{L-2}\sum_{s=0}^{L-3}\mathrm{CE}(\lambda^\pi_{i,s},q^\pi_{i,s}),\qquad
\ell^\pi_{i,shift}=\frac1{L-2}\sum_{s=0}^{L-3}w_\Delta\chi^\pi_{i,s}d_\alpha(\widehat{\Delta z}_{i,s}-\mathrm{sg}(z_{i,s+1}-z_{i,s}))^2,$$
with $\ell^\pi_i=\ell^\pi_{i,action}+\ell^\pi_{i,shift}$. A row with $q^\pi_{i,s}=0$ has CE 0 and is not an action-valid row. A row with $\chi^\pi_{i,s}=0$ has shift contribution 0 and is not shift-valid.

(4) **Perception terms.** For clip $i$, with every mean over $s=0..L-2$:
$$\ell^\phi_{i,push}=\frac1{L-1}\sum_s(-\chi_{i,s}d_{i,s}),\quad
\ell^\phi_{i,constraint}=\frac1{L-1}\sum_s(-\chi_{i,s}\mathrm{sg}(\lambda)q_{i,s}),$$
$$\ell^\phi_{i,dual}=\frac1{L-1}\sum_s\chi_{i,s}\log\lambda\,\mathrm{sg}(q_{i,s}),\quad
\ell^\phi_{i,norm}=\frac1{L-1}\sum_s\beta(h_{i,s}+h_{i,s+1}),$$
and $\ell^\phi_i$ is their sum. `phi_stats` contains, over all clips and states with equal weight: `d_alpha_mean` over all $(i,s)$; `d_alpha_changed_mean` over $\chi_{i,s}=1$, null if none; `z_norm_mean` = the mean of $\|z_{i,s}\|_2$ over all $i,s=0..L-1$, not of the squared norm; `hinge_active` = fraction of states with $z\_norm_{i,s}>c$; `changed` and `state_pairs` counts; and scalar `dual_lambda` = $\lambda$.

(5) **Preference output suite [U].** Define $u_{r,inc}=\sigma(t_r)$ on all pairs and $u_{r,succ}=\sigma(s_r)$ on $C$ only, with targets $y_{r,inc}=i_r$ and $y_{r,succ}=p_r$. The `inc` suite uses $w_r=1/|R|$ on every population. The `succ` suite is always conditional on comparable pairs and uses $w_r=1/|C|$ for per-step, pooled-window and reference populations. These weights apply to every field below; quantiles and calibration are unweighted over the same conditional population because all its weights are equal.
Each head has:
- `count`, `mean` = $\bar u$, `std`, `brier` = $\sum_rw_r(u_r-y_r)^2$;
- `low` = $\sum_rw_r[u_r<\tau_s]$, `high` = $\sum_rw_r[u_r>1-\tau_s]$;
- for each $q\in Q_d$, `quantiles[str(q)]` by (6);
- `calibration` and `ece` by (7), with soft targets retained for `succ`.
The `succ` suite is null when $C=\emptyset$. A step head suite has exactly `count`, `mean`, `std`, `brier`, `low`, `high`; a full aggregate/reference head suite adds exactly `quantiles`, `calibration`, `ece`.

(6) **Quantile.** Sort the $N$ unweighted values increasingly as $x_{(1)}\le\dots\le x_{(N)}$. For $h=1+q(N-1)$, $j=\lfloor h\rfloor$, $\alpha=h-j$: the result is $x_{(j)}$ when $\alpha=0$, otherwise $(1-\alpha)x_{(j)}+\alpha x_{(j+1)}$. For $N=1$ it is the sole value; for $N=0$, null. No library interpolation convention is implicit.

(7) **Calibration.** Bin $b=0..B_c-1$ contains $u\in[b/B_c,(b+1)/B_c)$, except that $u=1$ belongs to $b=B_c-1$. `calibration` is always a list of exactly $B_c$ entries in increasing bin order, each `{lower, upper, count, mean_prediction, mean_target}`; means in an empty bin are null, and no bin index or object-key encoding is added. `ece` = $\sum_b(n_b/N)|\bar u_b-\bar y_b|$ over nonempty bins, $N=\sum_bn_b$, and is null for $N=0$. Action calibration uses the same list with `mean_target` equal to the fraction of correct rows.

(8) **Incomparability classification.** Predict positive iff $u_{r,inc}\ge\tau_i$. Log TP, TN, FP, FN and `accuracy` = $(TP+TN)/N$, null for $N=0$. `tpr` = $TP/(TP+FN)$, null when $TP+FN=0$; `tnr` = $TN/(TN+FP)$, null when $TN+FP=0$. `balanced_accuracy` = $(tpr+tnr)/2$ only when both recalls are non-null, otherwise null. Raw accuracy never replaces balanced accuracy.

(9) **Preference strata.** The persisted catalogue of (10) defines every stratum. On each population, let $S_k$ be the rows in stratum $k$ and $n_k=|S_k|$. Empty strata have `count=0` and null `mean_prediction`, `mean_target`, `bce`, `brier`. For a nonempty comparable stratum $k$:
$$mean\_prediction=\frac1{n_k}\sum_{r\in S_k}\sigma(s_r),\qquad mean\_target=\frac1{n_k}\sum_{r\in S_k}p_r,$$
$$brier=\frac1{n_k}\sum_{r\in S_k}(\sigma(s_r)-p_r)^2,\qquad bce=\frac2{n_k}\sum_{r\in S_k}\mathrm{BCE}(s_r,p_r).$$
For `"inc"`:
$$mean\_prediction=\frac1{n_k}\sum_{r\in S_k}\sigma(t_r),\qquad mean\_target=1,$$
$$brier=\frac1{n_k}\sum_{r\in S_k}(\sigma(t_r)-1)^2,\qquad bce=\frac2{n_k}\sum_{r\in S_k}\mathrm{BCE}(t_r,1).$$
Window strata use pooled pairs; reference strata use equal-weight pairs of the part. Ordinary direction accuracy of $p_\succ$ is not defined or logged.

### Fixed global train baselines

(10) **Fit once and share initialization state [U].** During the counts pass of (17), read every train pair item exactly once in index order, without reading clips. Let $N=N_R$, $N_i=\sum_ri_r$, $N_c=N-N_i$ and $S_p=\sum_{r\in C}p_r$. Fit
$$u=N_i/N,\qquad v=S_p/N_c.$$
The shared initialization artifact `{out_dir}/global-fit.json` stores `pairs`, `incomparable`, `comparable`, `sum_p`, `inc_probability=u`, `succ_probability=v`, `inc_logit=log(u/(1-u))`, `succ_logit=log(v/(1-v))`, and `strata`. A logit is null when its probability is 0 or 1. When $N_c=0$, `sum_p`, `succ_probability` and `succ_logit` are null, signifying no comparable population rather than a numeric zero sum; when $N_c>0$, `sum_p` is the numeric $S_p$.
`strata` is the complete union of train and validation strata in the order of (17), including zero counts in either part. Each entry is `{key, kind, p, train, validation}`: for comparable, `kind="comparable"`, `key` is the decimal JSON value of $p$ and `p` is its float; for incomparable, `kind="inc"`, `key="inc"`, `p=null`. The counts are those of the corresponding part and are sufficient to reconstruct $N_x$ and largest-remainder allocations without another counts pass.
Initialization and resume use this exact precedence: with complete `reference/000000/`, load its `baselines.json` and, if `global-fit.json` also exists, require the typed fit and full catalogue to be identical, otherwise `RuntimeError`; with no complete shard 0 but existing `global-fit.json`, load it without rescanning; with neither, execute the counts pass once and atomically write `global-fit.json`. Stage 1 never creates `reference/` or a partial shard 0. Stage 2 reuses the shared fit and writes `reference/000000/baselines.json` byte-identical to `global-fit.json` when that file is the source; when only shard 0 is the source, its existing bytes are retained. A crash before `global-fit.json` exists repeats the counts pass; a crash afterward but before complete shard 0 reuses it, and a staged shard 0 is handled by ordinary shard rules. Fit and catalogue never move with a window, checkpoint or validation evaluation.

(11) **Evaluate fixed constants.** On any pair population with normalized loss/baseline weights $w_r$:
$$B_{inc}=2\sum_rw_r[-i_r\log u-(1-i_r)\log(1-u)],$$
$$B_{zero}=2\log2\,(1+\sum_{r\in C}w_r),\qquad B_{inc\_only}=B_{inc}+2\log2\sum_{r\in C}w_r,$$
$$B_{both}=B_{inc}+2\sum_{r\in C}w_r[-p_r\log v-(1-p_r)\log(1-v)],$$
$$B_{target\_entropy}=2\sum_{r\in C}w_r[-p_r\log p_r-(1-p_r)\log(1-p_r)].$$
If a population has no comparable pairs, the sums over $C$ are 0 and the last three metrics are still defined. If a term contains $\log0$ multiplied by a nonzero coefficient, the affected metric is null (its mathematical value is $+\infty$). If $N_c=0$ in the global fit, $B_{both}$ and the constant $s$ contribution are null. Per-step values use $w_r=1/|R|$; a 100-step aggregate averages the non-null per-step values with equal step weight; a reference part uses $w_r=1/R_r$. There is no per-window preference-constant refit.

### Gradient and update observations

(12) **Preclip norms.** After all micro-batch backwards of step $t$ and before clipping, for group $q$ with parameters $\theta\in q$:
$$n_q=\sqrt{\sum_{\theta\in q:g_\theta\ne None}\|g_\theta\|_2^2},\qquad
n_{all}=\sqrt{\sum_{q\in G}n_q^2},\qquad c_{clip}=\min(1,g_{max}/(n_{all}+10^{-6})).$$
A group is `active=true` iff at least one of its gradients is not None; its norm is null when inactive. Zero gradients that are not None make it active with norm 0. `n_all` and `c_clip` are null only if every parameter is inactive. Also log the same norms and activity for `P.pref.weight`, `P.inc.weight`, `P.inc.bias`; these three observations are null/inactive when $P$ is absent or unreached. These are norms before clipping, not component-wise gradient cosines.

(13) **Update norms without a second parameter copy.** Let $\theta^+$ be a parameter after AdamW step $t$, $\eta$ its group LR at $t$, $\lambda_w$ its weight decay, and $d=1-\eta\lambda_w$. If its gradient was not None, let $m,v,n$ be its optimizer state after that update and define
$$u=\frac{m/(1-\beta_1^n)}{\sqrt{v}/\sqrt{1-\beta_2^n}+\varepsilon},\qquad
\Delta\theta=-\frac{\eta}{d}(\lambda_w\theta^+ + u),\qquad \theta^-=\theta^+-\Delta\theta.$$
This is the algebraic rearrangement of `specs/pretrain.md` (6), evaluated on detached tensors one parameter at a time. For a parameter whose gradient was None, $\Delta\theta=0$ and $\theta^-=\theta^+$ even when $d=0$. If $d=0$ for a parameter with a non-None gradient, reconstruction is undefined: its `UpdateTerm.before` and `.delta` are null, never placeholder tensors, and `.unavailable="zero_decay_factor"`. For each group, `updates.groups[q].norm` = $\sqrt{\sum_{\theta\in q}\|\Delta\theta\|_2^2}$ and `.relative` = that norm divided by $\sqrt{\sum_{\theta\in q}\|\theta^-\|_2^2}$, null when the denominator is 0. If any parameter has unavailable update terms, its group's norm and relative are null and `.unavailable` = `"zero_decay_factor"`; otherwise `.unavailable` is null. Computation is after `optimizer.step`, changes no state and needs temporary storage for only one parameter at a time.

### Step and 100-step reductions

(14) **Step record.** Every optimizer step produces the JSON object specified in Interface. Existing keys `step`, `phi`, `pi`, `P`, `loss`, `lr` retain their meanings. All new loss components are detached raw per-clip/per-pair means, not the dataset-normalized $\mathcal L$ factors. `counts` contains `items`, `clips`, `pairs`, `comparable`, `incomparable`, `sum_p`, `action_targets`, `shift_targets`, `phi_changed`, `phi_state_pairs`; `sum_p` is null when there are no comparable pairs. The action counts are valid state rows, not nonzero elements of the distributions:
$$counts.action\_targets=\sum_{i,s}\left[\sum_aq^\pi_{i,s,a}>0\right],\qquad counts.shift\_targets=\sum_{i,s}\chi^\pi_{i,s}.$$
`outputs.pi_action.count` equals `counts.action_targets` for the step; window/reference counts are their pooled sums.

(15) **Window.** A full window ends at $t$ divisible by $W_d$ and covers steps $t-W_d+1..t$. If $T$ is not divisible by $W_d$, the final checkpoint also writes one partial window $T-(T\bmod W_d)+1..T$ with `partial=true`; all others have `partial=false`. Losses and per-step baselines are means over their non-null step values with equal step weight, using `mean_nullable`; their nested coverage is defined in Interface. Counts are sums over steps. Output distributions, Brier scores, calibration, classification and action row metrics are recomputed from pooled observations, not averaged from per-step means: every pooled pair has weight $1/N$ for `inc`, every comparable pair has weight $1/|C|$ for `succ`, and every valid action row has equal weight. `uniform`, `copy_current` and `zero_shift` are averaged with equal step weight; the action empirical oracle uses (16).
Gradient and update aggregates are equal-step means over non-null values, zeros included, with their own counts rather than the loss coverage. `gradients.global` has `mean_norm`, `mean_clip_coefficient`, `active_steps`; each `gradients.groups[q]` and `P_heads` entry has `mean_norm`, `active_steps`. Gradient means use only active steps; inactive norms are null and excluded. `updates.groups[q]` has `mean_norm`, `mean_relative`, `norm_steps`, `relative_steps`, `unavailable.zero_decay_factor`; the first two use their respective non-null counts, and the last counts unavailable steps. When $P$ is absent, aggregate `P_heads=null`; when present but never active, its entries have null means and zero active steps.
`uniform` and `zero_shift` are non-null on every valid training step, including all-zero-target steps where they are 0, so their coverage is the window's step count. `copy_current` is null on a step when its exact CE is $+\infty$ by (16); its coverage counts only steps with a finite value, and its aggregate is their `mean_nullable` result. `empirical_constant_oracle` is a whole-window metric and has no coverage entry. Pooled suites and gradient/update aggregates carry their own counts and do not appear in `coverage`.

(16) **Policy window/reference observations.** On each valid action row, $a^*=\arg\max_a\lambda^\pi_{i,s,a}$ (ties take the smallest action index), confidence $c=\max_a\pi_{i,s,a}$ and correctness $z=[q^\pi_{i,s,a^*}>0]$. The index is computed without a library-dependent `argmax`: take the maximum logit, replace indices that do not attain it by $|A|$, and take the minimum index. Log pooled `top1_support` = mean $z$, `mean_confidence`, the $B_c$-element confidence/correctness calibration list of (7), and `ece`; all are null with no valid action rows. `action_entropy` = the mean over valid rows of $-\sum_a\pi_{i,s,a}\log\pi_{i,s,a}$. `zero_shift` uses the same per-clip reduction as $\ell^\pi_{i,shift}$ but replaces the predicted shift with 0; it is a moving target diagnostic because $z$ moves with $\phi$.

Three action CE baselines use the same per-clip $(L-2)^{-1}$ reduction and then the population's clip weights:
- `uniform`: each valid row has CE $\log|A|$;
- `copy_current`: predicted probabilities $r_a=(1-\epsilon_c)C_{i,s,a}+\epsilon_c/|A|$ on each row, with exact CE $-\sum_aq^\pi_{i,s,a}\log r_a$. If any $q^\pi_{i,s,a}>0$ has $r_a=0$, that clip's CE is $+\infty$ and the corresponding step/window/reference metric is null; JSON Infinity is forbidden. The domain $0\le\epsilon_c\le1$ is unchanged, so this includes possible $\epsilon_c=0$ cases.
- `empirical_constant_oracle`: fit $r_a=\sum_{i,s}w_{i,s}q^\pi_{i,s,a}$, normalize $r$ by $\sum_ar_a$, then evaluate $-\sum_{i,s,a}w_{i,s}q^\pi_{i,s,a}\log r_a$. For one step $w_{i,s}=1/(|K|(L-2))$; for a window of $W$ steps $w_{i,s}=1/(W|K_{step(i)}|(L-2))$; for a reference part $w_{i,s}=1/(2R_r(L-2))$. It is null when every target row is zero. This is explicitly a same-population oracle under the stated equal-step/equal-pair reduction, not a fixed global action baseline and not a cross-window fixed reference.

### Deterministic fixed reference

(17) **Two-pass selection [U].** Selection is built once, only when the train part has at least one pair. Strata are all distinct comparable $p$ values in train or validation, sorted numerically ascending, followed by `"inc"`.
Pass 1 (`pretrain reference counts`) runs only when neither complete shard 0 nor `global-fit.json` exists. It reads train then validation, one index shard at a time, retains only the global fit of (10) and `(part,stratum)` counts, and atomically writes `global-fit.json`. An existing shared fit makes this pass unnecessary. For part $x$ and stratum $k$, let $n_{x,k}$ be its pair count and $N_x$ its total pair count. If $N_x<R_r$, abort with `RuntimeError(f"{x} has only {N_x} pairs, reference needs {R_r}")` before pass 2.
Allocate $R_r$ selections proportionally by largest remainder: $a_{x,k}=\lfloor R_rn_{x,k}/N_x\rfloor$, then distribute the remaining $R_r-\sum_ka_{x,k}$ one each to the strata with largest fractional part $R_rn_{x,k}/N_x-a_{x,k}$, ties by the stratum order above. A stratum with zero allocation is absent from the reference and has null per-stratum metrics.
Use one local generator `numpy.random.Generator(PCG64([seed, 4]))`. For $x=$ train then validation and strata in the order above, draw `permutation(n_{x,k})`, take its first $a_{x,k}$ positions, and sort those selected source positions. These ordinals are positions within the stratum across its index shards, not positions within a shard.
Pass 2 (`pretrain reference select`) rereads the same shards in the same order. For each stratum, retain `offset`, the number of its items already passed; `local_index` is the item's zero-based ordinal within that stratum in the current shard. An item at stratum ordinal `offset + local_index` is retained iff that ordinal was selected; `offset` advances by the stratum's item count in the shard after it is processed. At most 128 full items are retained, each as the full index item plus `{part, stratum, shard, item_index}`, where `item_index` is its original position in that shard. No source position is repeated, and one pair item cannot occur in two strata or parts. Reference order is part order, stratum order, then ascending stratum ordinal.
Memory holds one index shard, fit/counts, selected ordinals and at most 128 selected items; no all-pair list or intermediate dataset other than `global-fit.json` is written. A crash before `global-fit.json` exists repeats both passes. If it exists but shard 0 is incomplete, only pass 2 is repeated, using the shared fit and no new counts scan. An existing complete shard 0 skips both and supplies the persisted fit, catalogue and selection without a rescan.

(18) **Shard 0.** After module and optimizer initialization but before optimizer step 1, compute and atomically write `reference/000000/`:
- `selection.json` = `{version: 1, seed, pairs_per_part: R_r, items: [...]}` from (17);
- `baselines.json` = the byte-identical content of `{out_dir}/global-fit.json` when that shared initialization file is the source, including the full `strata` catalogue; an already complete shard 0 retains its existing bytes;
- `initial.pt` = `torch.save` of `{Z0, P0}` where `Z0` is a CPU tensor of shape $(2R_r,2,L,D)$ in selection order, argument $a$ before $b$, preserving the head dtype, and `P0` is the CPU-copy `state_dict()` of $P$ at step 0;
- `metrics.json` = the step-0 record of (20);
- `pairs.json` in a new run = one typed `ReferencePair` per selected item, in selection order, with `micro_batch=j`, `pair_in_micro=0`, initial joint loss as `baseline=only_phi_post=joint_post` and both deltas zero.
For each selected pair, read its two clips, run the current step-0 $\phi$ once on the list `[clip_a, clip_b]`, store the detached CPU result in row $j$ of `Z0`, and evaluate `joint` on that result and $P$. At step 0 `joint=P_only=phi_only=initial`; their preference suites are computed once and copied into all four names, not recomputed. `joint` also contains the $\phi/\pi$ suite. If $P$ is absent, no reference dataset is built and all preference diagnostics are null.

(19) **Reference evaluation modes and state.** Every reference evaluation, including initialization, is under `torch.no_grad()`. Let `device = torch.device(cfg.device)`. Before evaluation, save every module's boolean `training` mode and `torch.get_rng_state()`; if `device.type == "cuda"`, also save `torch.cuda.get_rng_state(device)` for the configured training device, not the default CUDA device. Set every given module to eval mode. Use a separate `copy.deepcopy` of the current $P$, load `P0` into it, set it to eval mode and use it only for `phi_only`; never load old weights into the training $P$. Read one pair at a time and do not call `torch.cuda.set_device`.
In a `finally` block, including on exceptions, restore every module mode with `Module.train(saved_mode)`, restore the CPU state with `torch.set_rng_state`, and, when saved, restore CUDA with `torch.cuda.set_rng_state(cuda_state, device)`. Do not zero or change gradients, parameters, optimizer state or the numpy schedule generator. Evaluation reads no global numpy generator. Thus the next training step has exactly the state and random draws it would have had without evaluation.

(20) **The four conditions [U].** At step $t>0$ divisible by $T_r$, for each selected pair $j$:
- `joint`: $Z=\phi_t([clip_a,clip_b])$ and $P_t(Z_a,Z_b)$; also evaluate current $\pi$ and the terms of all three losses on $Z$;
- `P_only`: $P_t(Z0_{j,a},Z0_{j,b})$ on the stored initial latents;
- `phi_only`: the frozen $P_0$ on the same current $Z$ used by `joint`;
- `initial`: the step-0 losses, preference suite and baselines loaded from reference shard 0, repeated unchanged; no forward.
All four produce the loss components of (2) and the full preference suite of (5)–(11) separately for train and validation. Only `joint` adds the $\phi/\pi$ losses and suites of (3), (4) and (16), separately by part. A condition's preference loss components and baselines have pair weight $1/R_r$ within each part. No condition writes gradients or optimizer updates. In a new run, each returned positive-step `ReferencePair` uses the persisted initial pair's `baseline`, the current `phi_only` and `joint` losses, and their respective differences. A legacy run returns no pair records and never reads an initial pairs file.

(21) **Cadence, ordering and resume.** A reference evaluation is due only at $t=0$ and at $t=T_r,2T_r,\dots\le T$; there is no extra evaluation at a nonmultiple final $T$. At a positive due step, first write and prune the ordinary checkpoint (including that step's losses and 100-step aggregates), then evaluate. Before reference publication, persist the stage-33a outbox of per-condition/per-part observations; after source publication, publish those records without another forward. Shard $k$ contains `{version:1, step:k*T_r, conditions:{joint,P_only,phi_only,initial}}`; only shard 0 additionally contains selection, baselines/catalogue and initial tensors.
Shard 0 fixes the layout for the entire run. If it has `pairs.json`, shard 0 and every later shard contain both `metrics.json` and `pairs.json`. If it has no `pairs.json`, the run is legacy metrics-only: every later shard contains only `metrics.json`, no pairs are synthesized, and pair-derived diagnostics use `legacy_reference_pairs_absent`. A missing later pairs file in new mode or any later pairs file in legacy mode is an error. The two layouts are never mixed.
On startup, existing shard 0 is reused verbatim: no reselection, refitting, initial forward or replacement of `Z0/P0`; its persisted strata catalogue defines empty and nonempty strata on resume. All complete shards are checked for the layout above. For restored step $t$, expected complete reference count is $\lfloor t/T_r\rfloor+1$. If exactly one latest shard is missing and $t$ is a positive multiple of $T_r$, evaluate it from the restored state before step $t+1$; otherwise the count must equal the expected count. A later/future shard or any other gap is an error rather than silently recomputing an older state. Checkpoint pruning never deletes reference shards or `global-fit.json`. Global baselines and 100-step aggregates read the same typed fit and catalogue on every resume, using the precedence of (10); they never rescan merely because `global-fit.json` or complete shard 0 already exists.

### Behaviour and Domain

(R) In addition to the checks of `specs/pretrain.md`, config loading requires $W_d,T_r,R_r,B_c$ positive integers, `checkpoint_steps % diagnostic_window == 0`, `reference_steps % checkpoint_steps == 0`, $0<\tau_s<1/2$, $0<\tau_i<1$, $0\le\epsilon_c\le1$, and $Q_d$ a non-empty list of distinct numbers strictly increasing in $(0,1)$. Violations raise `ValueError` naming the field or divisibility condition. These divisibility rules ensure that every complete non-final aggregation window is checkpointed atomically and every reference evaluation immediately follows its checkpoint. The starting `checkpoint_steps` is 100, reconciling the current config with the stale value 500 in the old pretrain starting-values text/test; it is not changed back to 500.

(T) Startup follows the existing checks, module construction and optimizer restore, then (18)/(21), selection/reuse of (22) and deletion of stale gradient-flow work of (25). Each ordinary training step performs the existing backwards, then (12), existing clipping/LR/AdamW, then (13), then appends the step record. On a selected step it instead uses the retained original backward of (23), performs (24), freezes $P_t$ and computes the baseline of (27) before the same update, then performs (25)–(26) after that update and the post-update counterfactual; the original update itself is identical. At a checkpoint it writes all completed window records with `WindowAccumulator.drain(final=(step==T))` and all pending selected gradient-flow records; after that checkpoint is published it regenerates `gradient-flow/summary.json` from accepted records. Reference work follows only the ordering of (21). `model.pt`, the training schedule and the final `complete` marker are unchanged.

Domain additions: model outputs and gradients are finite as in the existing loop; configured index parts used for a reference each contain at least $R_r$ pairs when any train pair exists; no reference files are edited or deleted by hand. A run without train pairs retains the existing unpaired protocol and skips all preference/reference-only behavior. The approved evaluation performs exactly $2R_r=128$ pair iterations per due evaluation and at most $256$ clip occurrences through current $\phi$ per evaluation; there is no additional full-dataset model evaluation or training run.

### Gradient-flow instrumentation of selected real steps

(22) **Selected labelled steps [U].** Before training, independently replay the existing schedule (3) of `specs/pretrain.md` with a new local `numpy.random.Generator(PCG64([seed, 3]))` and the index items, one visited shard at a time; this neither consumes the training schedule generator nor reads frames. Let $\mathcal L$ be the steps with $|R(\mathcal B_t)|>0$. If $\mathcal L=\emptyset$, gradient-flow instrumentation is absent and prints `pretrain gradient flow: nothing to do`. Otherwise select:
- `beginning`: the first $t\in\mathcal L$;
- `middle`: the first $t\in\mathcal L$ with $t\ge\max(1,\lfloor T/2\rfloor)$, or the last $t\in\mathcal L$ before that target if none exists after it;
- `end`: the last $t\in\mathcal L$.
Deduplicate steps and store them in ascending order in `gradient-flow/selection.json` = `{version:1, middle_target:max(1,floor(T/2)), steps:[{step,roles}]}`, where `roles` contains `beginning`, `middle`, `end` in that order when a step has several roles. This gives at most three real labelled steps, including beginning/middle/end of the actual schedule rather than synthetic probes. The selection file is written atomically before step 1; a crash before it exists repeats the replay, and an existing selection file is reused without rescanning.

(23) **Exactly weighted component losses and original backward.** For micro-batch $j$, retain scalar tensors
$$L_{\phi,j}=(M/K_{tot})\sum_i\ell^\phi_{i,j},\qquad L_{\pi,j}=(M/K_{tot})w_\pi\sum_i\ell^\pi_{i,j},$$
and, only for a labelled micro-batch, $L_{P,j}=(Mw_P/R_{tot})\sum_r\ell^P_{r,j}$. The original training scalar remains computed by the original expression and operation order: `k_factor * (l_phi + w_pi * l_pi).sum()` plus `r_factor * l_P.sum()` when present. These separate tensors are diagnostic only and are not substituted into the training loss.
On an ordinary step, `loss.backward()` is unchanged. On a selected step, call `loss.backward(retain_graph=True)`, which computes the same original accumulated gradients while retaining only the graph needed by (24). Save the CPU and configured-device CUDA RNG states immediately after that original backward. All additional component calls occur after it and before the next micro-batch; after them restore RNG to this post-original-backward state, so instrumentation consumes zero effective RNG draws and the next micro-batch sees exactly the uninstrumented state.

(24) **Per-micro component gradients [U].** Parameter set $x=\phi$ comprises every parameter with `requires_grad=True` of modules `phi` and `phi_loss`, including `phi_dual`; $x=P$ comprises every such parameter of module `P`. For $x=\phi$, $c\in\{\phi,\pi,P\}$; for $x=P$, only $c=P$. For each component, call `torch.autograd.grad(L_c, params_x, grad_outputs=torch.ones_like(L_c), retain_graph=True, create_graph=False, allow_unused=True, materialize_grads=False)` once for all parameters of $x$, grouped for output only by the existing optimizer groups. It returns rather than accumulates gradients. `None` for an unused parameter in an existing labelled component is a zero contribution. An unlabelled micro-batch has no $L_P$ and therefore no $g[x,P,j]$ at all; it is not a zero-P observation.
Write one CPU tensor per returned non-None parameter contribution into that micro-batch's work shard, preserving dtype, shape, $x,c$ and optimizer group. The canonical `parameter` identifier is `<module_name>.<named_parameters() key>`, for example `phi.backbone....`, `phi_loss.log_lambda` or `P.pref.weight`. One mapping is built from `modules` in existing module order and `named_parameters()` order; every identifier is globally unique, belongs to exactly one optimizer group and is used verbatim by the work manifest and `UpdateTerm.parameter`. `str(id(parameter))` is forbidden because the identifier must remain stable on resume. An unused None contribution is represented in the manifest with `zero=true` and no tensor file, contributes an exact zero vector to reduction and zero nonfinite counts, and is not counted as a tensor contribution in work bytes or the reduce bar. Collection holds only one component's full returned vector at a time; the original `.grad` values, module parameters and optimizer state are untouched. Record weighted losses and nonfinite counts per micro-batch and group. For each tensor, `nan`, `pos_inf`, `neg_inf` count those entries and `count` is their sum; step nonfinite counts sum these per-micro counts by $x,c,q$.

(25) **Work shards and step-summed gradient metrics.** Work lives only at `gradient-flow/work/<name(t)>/<name(j)>/`, one atomic shard per micro-batch, with `manifest.json` and one `tensors/<id>.pt` per contribution. The manifest maps each id to `{x,component,group,parameter,shape,dtype,zero}`, with `zero=false` for every tensor file and `zero=true` for an unused None contribution. After all micro-batches, reduce the component vectors of one parameter at a time across micro-batches; never materialize all micro-batch vectors or all model gradients together. For parameter $\theta$ in optimizer group $q$, let $S_{q,\theta}^x(c)=\sum_jg_q^x(c,j)_\theta$, summed before every norm or dot below. For $x=\phi$ compute per optimizer group and total over groups in existing group order:
$$N_c=\|S^\phi(c)\|_2,\quad D_{P,\phi}=\langle S^\phi(P),S^\phi(\phi)\rangle,\quad D_{P,\pi}=\langle S^\phi(P),S^\phi(\pi)\rangle,$$
$$D_{P,other}=\langle S^\phi(P),S^\phi(\phi)+S^\phi(\pi)\rangle=D_{P,\phi}+D_{P,\pi},$$
$$D_{P,joint}=\langle S^\phi(P),S^\phi(\phi)+S^\phi(\pi)+S^\phi(P)\rangle.$$
Cosines are $D/(N_PN)$ for `P_phi`, `P_pi`, `P_other`; a cosine is null when either norm is 0. No epsilon is added. $D_{P,joint}$ is the signed first-order preference decrease rate for an unnormalised step in direction $-S^\phi(\phi)-S^\phi(\pi)-S^\phi(P)$: positive predicts decrease, negative predicts increase. Per-micro cosines are neither computed nor averaged. For $x=P$, compute $N_{P,P}=\|S^P(P)\|_2$ per group and total.
After reduction, delete that selected step's work tree; only metrics and per-pair results persist. A crash before ordinary checkpoint publication invalidates the work, and startup deletes the entire `gradient-flow/work/` tree before continuing, since accepted checkpoints never depend on unfinished work.

(26) **Alignment with the actual AdamW update.** After the unchanged clipping/LR/AdamW update, reconstruct $\Delta\theta$ and $\theta^-$ per parameter with the same iterator and formula as (13), never a second approximation. While reducing the step-summed gradients, compute per group and total
$$A_\phi=\langle S^\phi(P),\Delta\theta_\phi\rangle,\qquad A_P=\langle S^P(P),\Delta\theta_P\rangle.$$
For `phi_dual`, the preference gradient is zero wherever the parameter is unused. Negative $A$ predicts a first-order preference decrease from the actual update; positive predicts an increase. If (13) makes a group's update unavailable, its alignment is null with `unavailable="zero_decay_factor"`; the total is null if any contributing group is unavailable. A new `update_and_align` entry point returns both ordinary update statistics and these alignments in one parameter pass; ordinary steps still use `update_stats`, and both share one `iter_update_terms` implementation.

(27) **Counterfactual post-step preference loss on the same pairs [U].** Keep the selected batch's raw labelled items in schedule order, at most `batch` items, and make a frozen deep copy $P_t$ before the optimizer update; set only the copy to eval mode. Immediately before that update, under eval/no_grad and the state preservation of (19), re-read each labelled pair's raw clips and compute baseline per-pair total $\ell^P_0=\ell^P(P_t,\phi_t,labels)$. This evaluation-mode baseline avoids conflating counterfactual changes with train/eval mode differences. Reuse, rather than duplicate, the reference evaluation-state context manager.
After the actual update, enter the same `evaluation_state(cfg, modules)` context as for the baseline: save post-update module modes and CPU/configured-device RNG, set training modules to eval and use `no_grad`, then re-read exactly the same raw items in the same order, encode once with $\phi_{t+1}$, and on that same updated latent evaluate both frozen $P_t$ and current $P_{t+1}$. On exit, including exceptions, restore training-module modes and RNG to their post-update entry values; parameters and optimizer state are never restored. The frozen copy remains in eval mode throughout:
$$\Delta_{only\_phi,r}=\ell^P(P_t,\phi_{t+1},y_r)-\ell^P_{0,r},\qquad
\Delta_{joint,r}=\ell^P(P_{t+1},\phi_{t+1},y_r)-\ell^P_{0,r}.$$
There is one baseline and one post-update $\phi$ call per labelled pair, with two preference calls on the post-update latents. `P_t` is fixed in `Delta_only_phi`. No fixed-reference pair is substituted: every labelled pair of the selected optimizer step appears exactly once in `pairs.json` with `{micro_batch, pair_in_micro, a, b, labels, baseline, only_phi_post, joint_post, delta_only_phi, delta_joint}`. Both indices are zero-based: `micro_batch=j` identifies work shard `gradient-flow/work/<name(t)>/<name(j)>/`, and `pair_in_micro` is the pair's ordinal among labelled items only in $R(\mathcal B_j)$, not among all items or clips; with `micro_batch=1` it is always 0.

(28) **Per-pair and cross-step distributions.** For each counterfactual $d\in\{\Delta_{only\_phi},\Delta_{joint}\}$, store per selected step `count`, `mean`, `median`, every $q\in Q_d$ quantile by (6), `harmful` = fraction $d_r>0$, `helpful` = fraction $d_r<0$, and `zero` = fraction $d_r=0$. `pairs.json` retains every per-pair value. `gradient-flow/summary.json` contains a row per accepted selected step and, over those steps, distributions of every total norm/dot/cosine/update-alignment and of both counterfactual means. Each distribution has `count`, `median`, `quantiles`, `fraction_negative`, `fraction_positive`, `fraction_zero`; negative cosine means conflict, negative update alignment predicts improvement, and positive update alignment predicts harm. It also contains pooled per-pair distributions of both deltas with the per-step fields above. Every accepted selected step has equal weight in `step_distributions`, and every accepted pair has equal weight in `pair_distributions`. Fractions are computed from the stored per-step/per-pair values, not from a run-wide scalar mean. Summary is regenerated after each accepted checkpoint from at most three accepted step records.

(29) **No change to training results and real-evidence requirement.** Component collection adds no forward pass; the baseline and post-step evaluations add only the pair calls of (27). The original forward/backward/update order and every original loss, gradient, parameter, optimizer state, schedule position and effective RNG draw are unchanged. `checkpoint.pt`, `model.pt`, existing `losses.json` and `diagnostics.json` contents are bitwise identical with and without selected instrumentation. Selected metrics link to the ordinary step's `checkpoint_losses_index` for preclip global norm, clipping coefficient and learning rates instead of duplicating them.
Each accepted record has `evidence.kind="real_pretrain_step"`, its checkpoint shard and raw-pair artifact. Graph connectivity is established by the existing code/tests; a nonzero $N_P$ establishes only nonzero preference gradient; dot/cosines establish agreement or conflict; only $A_\phi$ and especially $\Delta_{only\_phi}$ directly test preference improvement caused by the actual perception update. Dynamic conclusions require these real pretraining checkpoint records and their raw labelled items. Synthetic unit tests establish correctness only and leave all dynamic hypotheses open.

## Interface

### Step JSON — `checkpoints/<name(j)>/losses.json`

The file remains a chronological list since the previous checkpoint. Each entry has exactly these top-level keys:

```
step, phi, pi, P, loss, lr, counts, components, outputs, phi_stats,
gradients, updates, baselines
```

- `counts`: the integer fields of (14), with `sum_p: float|null`; `action_targets` and `shift_targets` are valid-row counts by (14).
- `components`: `{"phi":{"push","constraint","dual","norm"}, "pi":{"action","shift"}, "P":{"inc","succ"}|null}`. Existing `phi`, `pi`, `P` are reduced from `terms()["total"]` exactly as before, never from separate diagnostic components.
- `outputs`: `{"P_inc": <step head suite|null>, "P_succ": <step head suite|null>, "pi_action": {"count","top1_support","mean_confidence","action_entropy"}}`. `P_succ` is conditional on comparable pairs by (5).
- `phi_stats`: (4); `z_norm_mean` uses $\|z\|_2$.
- `gradients`: `{"global":{"norm","clip_coefficient","active"}, "groups":{q:{"norm","active"}}, "P_heads":{"pref_weight":{"norm","active"},"inc_weight":{"norm","active"},"inc_bias":{"norm","active"}}|null}`. `P_heads` is null only when $P$ is absent; an unreached head has `norm=null, active=false`.
- `updates`: `{"groups":{q:{"norm","relative","unavailable"}}}` by (13).
- `baselines`: `{"zero","inc_only","both","target_entropy","inc_constant","succ_constant"}` by (11), all null when pairs are absent; `inc_constant` = $B_{inc}$ and `succ_constant` is the comparable constant-$s$ contribution.

### Aggregate JSON — `checkpoints/<name(j)>/diagnostics.json`

A chronological list of windows drained since the previous checkpoint, including the final partial window of (15) when applicable. Each entry has exactly:

```
{version, first_step, last_step, partial, losses, coverage, counts,
 preference, pi, phi, gradients, updates}
```

`version` is 1. `losses` and `coverage.losses` have exactly the following mirrored structure; a numeric leaf in `losses` is `float|null`, and the corresponding leaf in `coverage` is a nonnegative integer step count:

```
losses = {
  phi, pi, P, loss,
  components: {
    phi: {push, constraint, dual, norm},
    pi: {action, shift},
    P: {inc, succ}
  },
  baselines: {zero, inc_only, both, target_entropy, inc_constant, succ_constant}
}
coverage = {
  losses: <the same nested keys with integer leaves>,
  pi: {uniform, copy_current, zero_shift}
}
```

Keys never disappear because a value is null. Every `losses` value is the equal-step `mean_nullable` result over its non-null per-step values; coverage is its non-null count. `coverage.pi` has the three integer counts of (15): `uniform` and `zero_shift` are always the window step count, while `copy_current` is the number of steps with finite exact CE. `lr`, pooled suites, `empirical_constant_oracle`, gradients and updates are not in this coverage.

- `counts`: sums of the step counts in (14); `sum_p` is the sum over all comparable pairs in the window and is null iff the window has no comparable pairs.
- `preference`: `{inc:<full head suite>, succ:<full comparable-only head suite>, inc_classification:<(8)>, strata:<(9)>}`.
- `pi`: `{action:<(16)>, empirical_constant_oracle, uniform, copy_current, zero_shift}`. `action.calibration` is the $B_c$-element list of (7). `uniform`, `copy_current`, `zero_shift` are equal-step means; `empirical_constant_oracle` is fitted once on the whole window with the weights of (16).
- `phi`: equal-step means of `push`, `constraint`, `dual`, `norm`, `dual_lambda`; pooled `d_alpha_mean`, `d_alpha_changed_mean`, `z_norm_mean`, `hinge_active`; summed `changed`, `state_pairs`.
- `gradients`: `{"global":{"mean_norm","mean_clip_coefficient","active_steps"}, "groups":{q:{"mean_norm","active_steps"}}, "P_heads":{"pref_weight","inc_weight","inc_bias"}|null}`, where each head entry has `mean_norm`, `active_steps`. Means follow (15).
- `updates`: `{"groups":{q:{"mean_norm","mean_relative","norm_steps","relative_steps","unavailable":{"zero_decay_factor":int}}}}`, by (15).

No raw per-example outputs are written. During a window memory holds the CPU `StepObservations` contract below for at most $W_d$ steps; drained observations are discarded. Since `checkpoint_steps` is a multiple of $W_d$, a resumed run recomputes no accepted partial window.

### Reference JSON — `reference/<name(k)>/metrics.json`

Every record has `{version:1, step, conditions}`. `conditions` has exactly `joint`, `P_only`, `phi_only`, `initial`. Each condition has exactly `train` and `validation`; each part has `{losses, preference, baselines}`.
- For every condition, `losses` contains `P`, `P_inc`, `P_succ`; `preference` has the aggregate preference schema; `baselines` has the same six keys as a step's baselines. All use equal pair weight within the part.
- Only `conditions.joint.<part>.losses` additionally contains `phi`, `phi_push`, `phi_constraint`, `phi_dual`, `phi_norm`, `pi`, `pi_action`, `pi_shift`. Only `conditions.joint.<part>` additionally has `pi` and `phi`, with the aggregate schemas of (16) and (4), evaluated over that part's $2R_r$ clip occurrences.
- `initial` repeats the step-0 losses/preference/baseline objects and has no `pi` or `phi` keys outside `joint`.
`selection.json`, `baselines.json` and `initial.pt` occur only in shard 0; their formats are (10), (17) and (18). In a new run every shard additionally has `pairs.json`, a list of the typed fixed-reference `ReferencePair` rows of `specs/rl-diagnostics.md`, in selection order. Legacy runs have no `pairs.json` in any shard; the file is never backfilled.

### Gradient-flow JSON and artifact layout

Immutable `gradient-flow/selection.json` has exactly `{version:1, middle_target, steps:[{step,roles}]}` from (22).

Each selected step is stored inside its next ordinary checkpoint as `checkpoints/<name(j)>/gradient_flow/<name(k)>/`, where $k$ is the zero-based index in `selection.steps`. The directory contains:
- `metrics.json` = `{version,step,roles,checkpoint_losses_index,micro_batches,phi,P,nonfinite,counterfactual,evidence}`;
- `pairs.json` = the list of every labelled pair of (27), in selected-step schedule order.

`checkpoint_losses_index` is the zero-based index of the same optimizer step in that checkpoint's `losses.json`. Consumers join it for preclip global norm, clip coefficient and LR; these values are not duplicated in gradient-flow JSON.

Each `micro_batches` entry is:
```
{micro_batch, labelled,
 losses: {phi, pi, P|null},
 norms: {phi:{q:{phi,pi,P|null}}, P:{q:{P}}|null},
 nonfinite: <nested nonfinite counts>}
```
The losses are the dataset-weighted scalars of (23). Micro norms are per optimizer group; an unlabelled micro-batch has null P norms and no P contribution. `phi` in the top-level record is `{groups:{q:<phi bundle>},total:<phi bundle>}`. A phi bundle is:
```
{norms:{phi,pi,P},
 dot:{P_phi,P_pi,P_other,P_joint},
 cosine:{P_phi,P_pi,P_other},
 update_dot,
 update_unavailable}
```
Top-level `P` is `{groups:{q:<P bundle>},total:<P bundle>}`, where a P bundle is `{norm,update_dot,update_unavailable}`. Totals sum squared norms and dot products over groups before the outer square root, not per-group cosines. Every scalar is `float|null`; zero-denominator cosines and unavailable updates are null by (25)–(26). Every group and total bundle also has `update_unavailable`: null when update alignment is available, otherwise `"zero_decay_factor"`; total carries that reason if any contributing group is unavailable. No join to step losses is needed to interpret `update_dot`.

`nonfinite`, both per micro and top-level, is nested `x -> component -> group -> {nan,pos_inf,neg_inf,count}` with integer counts by (24); a missing P component on an unlabelled micro is null rather than zero. Top-level counts are sums over micro-batches. `counterfactual` has exactly `delta_only_phi` and `delta_joint`, each `{count,mean,median,quantiles,harmful,helpful,zero}` by (28). `evidence` = `{kind:"real_pretrain_step",checkpoint_shard,pairs}`. Both path values are strings relative to `out_dir`: `checkpoint_shard="checkpoints/<name(j)>"` and `pairs="checkpoints/<name(j)>/gradient_flow/<name(k)>/pairs.json"`; direct lookup is `out_dir / value`.

`gradient-flow/summary.json`, atomically regenerated after each accepted checkpoint, is:
```
{version:1,
 steps:[{step,roles,metrics,phi_total,P_total,
         delta_only_phi_mean,delta_joint_mean}],
 step_distributions:{<canonical metric path>:{
   count,median,quantiles,fraction_negative,fraction_positive,fraction_zero}},
 pair_distributions:{delta_only_phi:<pair stats>,delta_joint:<pair stats>}}
```
`metrics` is the string `"checkpoints/<name(j)>/gradient_flow/<name(k)>/metrics.json"`, relative to `out_dir`, so direct lookup is `out_dir / metrics` without another path base; `phi_total` and `P_total` are the corresponding bundles. Canonical metric paths are `phi.total.norms.{phi,pi,P}`, `phi.total.dot.{P_phi,P_pi,P_other,P_joint}`, `phi.total.cosine.{P_phi,P_pi,P_other}`, `phi.total.update_dot`, `P.total.norm`, `P.total.update_dot`, `counterfactual.delta_only_phi.mean`, and `counterfactual.delta_joint.mean`. Null values are excluded from each distribution's count; an empty distribution has count 0 and null statistics. Pair stats have the schema of (28) and are pooled from every accepted pair. Summary rows and distributions include only accepted checkpoint records, never pending or synthetic work.

### Observation and fit types

`src/mcpref/pretrain/diagnostics.py` defines these CPU observation contracts. All tensors have `requires_grad=False`.

```
@dataclass(frozen=True)
class PreferenceRows:
    s: Tensor       # CPU float32, (N,)
    t: Tensor       # CPU float32, (N,)
    p: Tensor       # CPU float32, (N,), placeholder 0.5 when inc=1
    inc: Tensor     # CPU uint8, (N,)

@dataclass(frozen=True)
class PolicyObservations:
    q_sum: Tensor                  # CPU float64, (|A|), per-step weights of (16)
    action_valid: int
    shift_valid: int
    confidence_sum: float
    correct_sum: int
    entropy_sum: float
    calibration_count: Tensor      # CPU int64, (B_c,)
    calibration_confidence: Tensor # CPU float64, (B_c,)
    calibration_correct: Tensor    # CPU float64, (B_c,)
    uniform: float
    copy_current: float | None       # null iff exact CE is +infinity
    zero_shift: float

@dataclass(frozen=True)
class PhiObservations:
    d_sum: float
    d_changed_sum: float
    changed: int
    state_pairs: int
    z_norm_sum: float
    z_count: int
    hinge_active: int

@dataclass(frozen=True)
class StepObservations:
    preference: PreferenceRows | None
    policy: PolicyObservations
    phi: PhiObservations

@dataclass(frozen=True)
class Stratum:
    key: str
    kind: Literal["comparable", "inc"]
    p: float | None
    train: int
    validation: int

@dataclass(frozen=True)
class GlobalFit:
    pairs: int
    incomparable: int
    comparable: int
    sum_p: float | None
    inc_probability: float
    succ_probability: float | None
    inc_logit: float | None
    succ_logit: float | None
    strata: tuple[Stratum, ...]
```

After each micro-batch's `loss.backward()` and before the next micro-batch, detached sufficient statistics from the same forward are reduced, transferred to CPU and accumulated into `StepObservations`. Full policy logits, full $q$/current-weight matrices, $Z$ and perception state tensors are used only within that micro-batch and are not retained. `q_sum` is already normalized per step by (16), so windows sum the vectors. Preference rows are the only per-example values retained, bounded by `diagnostic_window * batch` pairs; action/phi state is $O(|A|+B_c)$ sufficient statistics per step. No raw GPU observations enter `WindowAccumulator`.

`src/mcpref/pretrain/reference.py` additionally defines:

```
@dataclass(frozen=True)
class ReferenceData:
    selection: dict
    baselines: GlobalFit
    Z0: Tensor                # CPU, (2*R_r, 2, L, D)
    P0: dict[str, Tensor]     # CPU state_dict
    initial_metrics: dict
    initial_pairs: tuple[ReferencePair, ...] | None  # None only in legacy mode
```

JSON serialization of `GlobalFit` and `Stratum` uses exactly the fields of (10), identical between `{out_dir}/global-fit.json` and `reference/000000/baselines.json`; loading reconstructs the complete typed catalogue and part counts.

### Function and module interface

- `src/mcpref/preferences/loss.py`: `def terms(self, out: tuple[Tensor, Tensor], labels: list[dict]) -> dict[str, Tensor]` returns per-pair `{inc,succ,total}` of (2), each shape $(m,)$; `forward` returns `total`.
- `src/mcpref/policy/loss.py`: `def terms(self, out: tuple[Tensor, Tensor], Z: Tensor, clips: list[Clip]) -> tuple[dict[str, Tensor], dict[str, Tensor]]` returns per-clip `{action,shift,total}`, each shape $(m,)$, and `{action_valid,shift_valid}` boolean tensors of shape $(m,L-2)$. `total` preserves the original per-state addition and reduction, and `forward` returns it. Targets are computed once.
- `src/mcpref/perception/loss.py`: `def terms(self, Z: Tensor, clips: list[Clip]) -> tuple[dict[str, Tensor], dict[str, Tensor]]` returns per-clip `{push,constraint,dual,norm,total}`, each shape $(m,)$, and `{changed,d_alpha}` of shape $(m,L-1)$ plus `{z_norm,hinge}` of shape $(m,L)$; `changed` is boolean, the others have $Z$'s dtype. `z_norm` is $\|z\|_2$ and `hinge` = $\max(0,z\_norm^2-c^2)$. `total` preserves the original per-state addition and reduction, and `forward` returns it. Frames, metas and $\chi$ are computed once.
- `src/mcpref/pretrain/loss.py`: `def step_gradients(items: list[dict], modules: dict[str, torch.nn.Module], read: Callable[[str], Clip], m: int, k_tot: int, r_tot: int, w_pi: float, w_P: float, micro_batch: int, *, cfg: PretrainConfig) -> dict`. All previous positional parameters retain their order and meaning; `cfg` is required, keyword-only, and supplies `calibration_bins` and `action_copy_eps` together with the rest of the shared diagnostic config used later by reference and gradient-flow. There is no default, overload or backward-compatibility mode; omitting `cfg` raises `TypeError`. `micro_loss` additionally returns the separately weighted scalar tensors `{phi,pi,P}` of (23) while computing the original training scalar by its original expression. `step_gradients` produces `StepObservations` and, only for a selected step, uses `loss.backward(retain_graph=True)` and calls `collect_micro`; ordinary steps use `loss.backward()` and no gradient-flow call. There is no additional module forward.
- `src/mcpref/pretrain/diagnostics.py`:
  - `def preference_suite(rows: PreferenceRows | None, *, strata: tuple[Stratum, ...], calibration_bins: int, saturation_threshold: float, inc_classification_threshold: float, quantiles: tuple[float, ...], full: bool) -> dict`: `full=False` returns `{inc:<step head suite>,succ:<step head suite>}`; `full=True` returns `{inc:<full>,succ:<full>,inc_classification:<(8)>,strata:<(9)>}`. Every config parameter is required even when `full=False`; no global catalogue or hidden default is read.
  - `def preference_baselines(rows: PreferenceRows | None, fit: GlobalFit) -> dict[str, float | None]`: returns exactly `zero`, `inc_only`, `both`, `target_entropy`, `inc_constant`, `succ_constant` by (11).
  - `def policy_suite(observations: Sequence[PolicyObservations], *, calibration_bins: int, copy_eps: float, full: bool) -> dict`: `full=False` returns step `pi_action`; `full=True` returns the aggregate `pi` schema of (16).
  - `def phi_suite(observations: Sequence[PhiObservations]) -> dict[str, float | int | None]`: returns exactly `d_alpha_mean`, `d_alpha_changed_mean`, `z_norm_mean`, `hinge_active`, `changed`, `state_pairs`.
  - `def mean_nullable(values: Iterable[float | None]) -> tuple[float | None, int]`: returns the non-null mean and count, or `(None,0)`.
  All are pure scalar/JSON functions with no network calls or hidden fit/strata state.
- `src/mcpref/pretrain/reference.py`:
  - `def load_global_fit(out_dir: Path) -> GlobalFit` — applies the exact precedence of (10), including identity checking when both sources exist.
  - `def write_global_fit(out_dir: Path, fit: GlobalFit) -> None` — atomically writes `{out_dir}/global-fit.json` with the exact bytes later copied to `reference/000000/baselines.json`.
  - `def select_reference(cfg: PretrainConfig, run: dict) -> tuple[dict, GlobalFit]` — uses the shared fit when present, otherwise performs the counts pass and writes it, then performs pass 2 and returns `selection` and typed `baselines`.
  - `def load_initial(directory: Path) -> ReferenceData` — loads shard 0 and its pairs when present, and checks every complete reference shard for the consistent new/legacy layout of (21).
  - `def write_initial(directory: Path, selection: dict, baselines: GlobalFit, initial: dict, metrics: dict, pairs: Sequence[ReferencePair]) -> None` — atomically writes the new shard-0 format of (18)–(21).
  - `def write_metrics(directory: Path, k: int, metrics: dict, pairs: Sequence[ReferencePair] | None) -> None` — atomically writes one later shard; `pairs=None` is valid only for a legacy run.
  In these functions `directory` is `{out_dir}/reference`; shard 0 and later shards live below it.
- `src/mcpref/pretrain/reference_eval.py`:
  - `def evaluate_initial(cfg: PretrainConfig, modules: Modules, selection: dict, baselines: GlobalFit, run: dict) -> tuple[dict[str, Tensor | dict[str, Tensor]], dict, list[ReferencePair]]` — returns `({"Z0":Z0, "P0":P0}, initial_metrics, initial_pairs)`. `run` supplies states, width, height and index paths.
  - `def evaluate_reference(cfg: PretrainConfig, modules: Modules, reference: ReferenceData, run: dict, step: int) -> tuple[dict, list[ReferencePair] | None]` — (19)–(20), using explicit stored `Z0`, `P0`, fit/catalogue, initial metrics and, in new mode, initial pairs. Legacy mode returns `(metrics,None)` without reading pairs. No optimizer argument, mutation, hidden default device/geometry or additional forward.
  - `@contextmanager def evaluation_state(cfg: PretrainConfig, modules: Modules) -> Iterator[None]` — the single shared implementation of eval/no_grad and configured-device state preservation of (19), also used by counterfactual evaluation.
- `src/mcpref/pretrain/aggregate.py`:
  - `class WindowAccumulator`: `__init__(self, window: int, *, strata: tuple[Stratum, ...], calibration_bins: int, saturation_threshold: float, inc_classification_threshold: float, quantiles: tuple[float, ...], action_copy_eps: float) -> None`; `add(self, step_record: dict, observations: StepObservations) -> None`, checking consecutive steps; `drain(self, *, final: bool = False) -> list[dict]`.
  - `drain(False)` emits only full windows. At a checkpoint the loop calls `drain(final=(step==T))`; `final=True` additionally emits one partial window if observations remain and may be called only once. Returned records are chronological and drained observations are discarded.
- `src/mcpref/pretrain/update.py`: existing `update` remains the only clipping/LR/optimizer operation. `def gradient_stats(groups: Groups, grad_clip: float, *, preference: torch.nn.Module | None) -> dict` reads `preference.pref.weight`, `preference.inc.weight`, `preference.inc.bias` and implements (12) immediately before `update`; it is called with `preference=modules.get("P")`.
  - `@dataclass(frozen=True) class UpdateTerm`: fields `group:str`, `parameter:str`, `before:Tensor|None`, `delta:Tensor|None`, `unavailable:str|None`; `parameter` is the canonical module-qualified name of (24), identical to the work manifest. Non-null tensors are detached $\theta^-$ and $\Delta\theta$ on the parameter device, while `d=0` with a non-None gradient gives null tensors and `unavailable="zero_decay_factor"`.
  - `def iter_update_terms(cfg: PretrainConfig, optimizer: torch.optim.AdamW, groups: Groups, lr: dict) -> Iterator[UpdateTerm]` — the one implementation of (13), yielding one parameter's `{group,parameter,before,delta,unavailable}` at a time.
  - `def update_stats(cfg: PretrainConfig, optimizer: torch.optim.AdamW, groups: Groups, lr: dict) -> dict` — ordinary-step update norms through `iter_update_terms`.
  - `def update_and_align(cfg: PretrainConfig, optimizer: torch.optim.AdamW, groups: Groups, lr: dict, work: Path) -> tuple[dict, dict]` — one parameter pass producing both ordinary update stats and the two alignments of (26) from `work`; it does not call the optimizer or recompute updates with another formula.
- `src/mcpref/pretrain/gradient_work.py`:
  - `def collect_micro(weighted: dict[str, Tensor | None], modules: Modules, groups: Groups, directory: Path, micro_batch: int) -> dict` — (23)–(24), returning the micro JSON record and writing its atomic work shard. RNG is restored to immediately after the original backward and original `.grad` values are not mutated.
  - `def reduce_work(directory: Path, *, groups: Groups) -> dict` — (25), returning per-group/total norms, dots, cosines and nonfinite counts while holding the component vectors of one parameter across micro-batches. The required `groups` is the same ordered mapping produced by `check_groups`; it determines output group order and parameter membership, independently of manifest collection order. It does not average per-micro cosines. No `progress` argument or hidden progress state belongs to this interface; the caller owns the tqdm bar.
- `src/mcpref/pretrain/gradient_flow.py`:
  - `@dataclass(frozen=True) class GradientStepRecord`: fields `metrics: dict` and `pairs: list[dict]`, exactly the contents of one accepted `metrics.json` and `pairs.json`; it is the sole pending/checkpoint/summary record type.
  - `def select_gradient_steps(cfg: PretrainConfig, run: dict, data: Part) -> dict` — (22), using a local schedule replay and returning the exact `selection.json` object.
  - `def baseline_pairs(cfg: PretrainConfig, modules: Modules, items: list[dict], run: dict) -> list[float]` — (27) before update through `evaluation_state`, returning total preference loss per labelled pair in item order.
  - `def post_step_pairs(cfg: PretrainConfig, modules: Modules, frozen_P: torch.nn.Module, items: list[dict], run: dict) -> tuple[list[float], list[float]]` — one current-$\phi$ call and frozen/current preference calls per pair after update, returning `(only_phi_post,joint_post)`.
  - `def summarize(records: Sequence[GradientStepRecord]) -> dict` — (28) and the exact `summary.json` schema using only accepted real records and their stored per-pair values.
- `src/mcpref/pretrain/checkpoints.py`: `write_checkpoint(..., losses: list[dict], diagnostics: list[dict], gradient_flow: Sequence[GradientStepRecord])` additionally writes `diagnostics.json` and every pending `gradient_flow/<name(k)>/{metrics.json,pairs.json}` from the supplied accepted-candidate records. `checkpoint.pt` is unchanged. `prune` still affects only `checkpoint.pt`.
- `src/mcpref/pretrain/loop.py`: after `global_fit = load_global_fit(out_dir)`, constructs `WindowAccumulator(cfg.diagnostic_window, strata=global_fit.strata, calibration_bins=cfg.calibration_bins, saturation_threshold=cfg.saturation_threshold, inc_classification_threshold=cfg.inc_classification_threshold, quantiles=cfg.quantiles, action_copy_eps=cfg.action_copy_eps)` once, calls `step_gradients(..., micro_batch=cfg.micro_batch, cfg=cfg)` and `reduce_work(directory, groups=groups)` with the ordered mapping from `check_groups`, and passes the same config values to every `preference_suite`, reference evaluation and gradient-flow evaluation. It loads `DiagnosticsConfig` explicitly from `cfg.diagnostics`, constructs the one-time stage-33a `GlobalFit` handoff, and orchestrates (T), per-step/reference outboxes and accepted records, reference startup/resume, selected gradient-flow work/pending records, deletion of stale work and atomic `gradient-flow/summary.json` regeneration from accepted records. `src/mcpref/pretrain/config.py`: the new fields and checks of (R), plus required relative `diagnostics` config path.

### Consumes, produces, datasets and checkpoints

- Consumes: all existing pretrain inputs; additionally `{index_dir}/validation/<name(s)>/items.json` only for reference selection/evaluation. Reads clip files only for the selected 128 reference pairs and the labelled pairs of at most three gradient-flow steps, never for global baseline fitting or selection. Reuses `write_shard`, `complete_shards`, `shard_path`, `write_file`, `part_path`, `publish`, `cpu_copies` and existing checkpoint restoration.
- Produces: extended `losses.json`; new `diagnostics.json` in existing checkpoint shards; shared initialization `{out_dir}/global-fit.json`; `{out_dir}/reference/`; immutable `{out_dir}/gradient-flow/selection.json`; accepted `gradient_flow/<name(k)>/` directories inside checkpoint shards; atomically regenerated `{out_dir}/gradient-flow/summary.json`; and the stage-33a per-step/reference outboxes and accepted records under `DiagnosticsConfig.out_dir`. `global-fit.json` is written once by the counts pass, is never overwritten, and is copied byte-identically into shard 0. No change to `model.pt` or the pretrain run's own `run.json` completion semantics.
- Dataset `reference/`: item = one due reference evaluation; one evaluation per shard. A new shard 0 contains every file of (18), including `pairs.json`, and every new later shard contains `metrics.json` plus `pairs.json`; every legacy shard contains only the former files except `pairs.json`. Each source shard is written atomically immediately after its outbox and corresponding checkpoint and kept permanently. Readers: this block on resume and subsequent analysis tooling. The two selection passes each read one index shard at a time and retain only the state bounded by (17); evaluation holds one selected pair's clips/current latents, the small $Z0$ tensor, a frozen $P$ copy and the current suite, never the frames of the clip dataset.
- Gradient-flow accepted dataset: item = one selected optimizer step; its `metrics.json` and complete per-pair `pairs.json` are written only inside the next ordinary checkpoint and kept through checkpoint pruning. Pending records hold only metrics plus at most `batch` raw labelled items in memory. `summary.json` is derived from accepted records and rewritten after that checkpoint is published. On every startup before continuing, summary is rebuilt from accepted records and stale `gradient-flow/work/` is removed; pending records and raw work are never trusted across processes.
- Gradient-flow work dataset `gradient-flow/work/<name(t)>/`: item = one selected micro-batch; one atomic shard per micro-batch with manifest and per-contribution CPU tensors. Work is written only for the current selected step and deleted after reduction, and the whole work root is deleted on startup. It is intermediate state, not an accepted metric artifact. Its exact byte count is the sum of `numel * element_size` over every manifest tensor; no dtype compression is allowed. Collection memory holds one component's returned vector at a time, and reduction memory holds the component vectors of one parameter across micro-batches, never the full gradient dataset.
- Checkpoints: existing cadence and pruning unchanged. `diagnostics.json` and all pending gradient-flow records are complete before their checkpoint shard is published. Reference and accepted gradient-flow shards are not optimizer checkpoints and are not pruned; their resume rules are (21) and (25)/(28).
- Exact reference disk size bound beyond JSON metrics: $Z0$ has $2R_r\cdot2\cdot L\cdot D$ scalar values in the head dtype (at the current values $128\cdot2\cdot12\cdot40=122880$ scalars, 491520 bytes in float32), plus one CPU $P$ state dict in `initial.pt` and one frozen $P$ copy in memory during evaluation. Gradient-flow work is bounded to one selected step by the manifest byte rule above rather than to reference size.

### Progress bars

All bars write to stderr, have no postfix and change no result.
- `pretrain reference counts` (unit `shard`): pass 1 of (17), only when neither shared source exists; total = train + validation index shards; advanced after each shard. Existing `global-fit.json` or complete shard 0 → `pretrain reference counts: already done`.
- `pretrain reference select` (unit `shard`): pass 2 of (17); the same total and advance rule. Existing shard 0 → `pretrain reference select: already done`. If pass 1 aborts, pass 2 is not entered.
- `pretrain reference initial` (unit `pair`): (18); total $2R_r=128$, advanced after each pair's initial latents and metrics are complete. Existing shard 0 → `pretrain reference initial: already done`.
- `pretrain reference` (unit `pair`): each positive due evaluation; total $2R_r=128$, advanced after all conditions for a pair. A missing latest reference shard on resume gets a bar with `initial=0`. No due work → no bar; at the corresponding restored due step already complete → `pretrain reference: already done`.
- `pretrain gradient flow selection` (unit `step`): schedule replay of (22); total $T$, advanced after each optimizer-step schedule entry. Existing `gradient-flow/selection.json` → `pretrain gradient flow selection: already done`; no labelled steps → `pretrain gradient flow: nothing to do`.
- `pretrain gradient flow collect` (unit `micro-batch`): (23)–(24) for one selected step; total its number of micro-batches $J$, advanced after each atomic work shard. No selected work → no bar.
- `pretrain gradient flow reduce` (unit `contribution`): (25) and update alignment for the same step; total twice the number of tensor entries in its work manifests, once for each of those two parameter passes, advanced after each contribution tensor is read. An empty contribution list is impossible for a selected labelled step.
- `pretrain gradient flow pairs` (unit `evaluation`): baseline and post-step counterfactual of (27); total $2|R|$ for that selected step, advanced once after each baseline pair and once after each post-update pair.
- With no train pairs, reference setup prints once `pretrain reference: nothing to do` and gradient-flow setup prints once `pretrain gradient flow: nothing to do`. The existing `pretrain inputs` and `pretrain` bars are unchanged. Window aggregation and summary regeneration have no bar because they are bounded by 100 steps and at most three accepted records respectively; work deletion is one directory operation, not a work-item loop.

## Revisions required in existing specs and configs

- `specs/pretrain.md`: add the new config fields and (R) checks; state that validation is read only for this diagnostic reference; replace (7) and the `losses.json` format with the backward-compatible extended schema; add `diagnostics.json`, shared `global-fit.json`, `reference/`, selected gradient-flow instrumentation (22)–(29), work/accepted artifacts, ordering and resume rules to (T), Interface, Datasets, Checkpoints and Progress bars; add the diagnostic tests. Document that selected steps use `backward(retain_graph=True)` only to permit returned component gradients, with post-original-backward RNG restoration and no change to original training results. Include nullable infinite `copy_current`, shared post-step evaluation state, zero-based pair indices, nullable undefined update tensors and bundle-local reasons, canonical parameter names, direct `out_dir`-relative evidence paths, and `step_gradients(..., micro_batch: int, *, cfg: PretrainConfig)` with required keyword-only config and `TypeError` on omission. `reduce_work` requires the ordered `Groups` from `check_groups`, with no spec-level progress argument and `TypeError` on omission. In Starting config values, explicitly replace stale `checkpoint_steps = 500` with `checkpoint_steps = 100`, matching the current config and this block's cadence; update `tests/pretrain/test_config.py::test_starting_values` to 100 and the new fields. Existing (1)–(6), actual checkpoint cadence, schedule and optimizer formulas remain unchanged.
- `specs/preference.md`: add `PreferenceLoss.terms`, including `total`, the comparable-only output-suite distinction and its use for per-pair counterfactual evaluation, to Interface and Tests. Its network, loss value, groups and training protocol are unchanged.
- `specs/policy.md`: add `PolicyLoss.terms`, its validity tensors and single target computation to Interface/Tests. Its loss, targets, dropout and network calls are unchanged.
- `specs/perception.md`: add `PhiLoss.terms`, its diagnostic tensors and single frame/meta computation to Interface/Tests; define diagnostic `z_norm` as $\|z\|_2$ and `hinge` as $\max(0,z\_norm^2-c^2)$; state that gradient-flow parameter set $\phi$ includes `phi_loss`'s `phi_dual` only for component/update accounting, without changing preference evaluation. Its loss and stop-gradients are unchanged.
- `specs/train-index.md`: change “no validation runs yet” to “validation is consumed by pretrain diagnostics”; remove “Validation runs and metrics” from Out of scope and name block pretrain-diagnostics as a validation reader. Index production, split and fingerprints are unchanged.
- `specs/00-overview.md`: row 7 becomes `| 7 | pretrain — one loop training φ, π and P; per-step/window/reference and selected gradient-flow diagnostics (pretrain-diagnostics) | P1, P4, P5 | 29 | G |`; add `specs/pretrain-diagnostics.md` to the `pretrain` package-map line. No training data-flow arrow changes.
- `configs/pretrain.yaml`: `diagnostics: configs/diagnostics.yaml`; after `micro_batch`: `diagnostic_window: 100`, `reference_steps: 500`, `reference_pairs: 64`, `calibration_bins: 10`, `saturation_threshold: 0.01`, `inc_classification_threshold: 0.5`, `quantiles: [0.05, 0.5, 0.95]`, `action_copy_eps: 0.1`. Existing `checkpoint_steps: 100` is retained and satisfies both divisibility checks; it is not changed to the stale starting value 500.
- `specs/runs.md`, `specs/shards.md`, `scripts/pretrain.sh`, model configs and all optimizer/loss-weight config values: no revision.

## Tests

New mirrors are `tests/pretrain/test_diagnostics.py`, `test_aggregate.py`, `test_reference.py`, `test_reference_resume.py`, `test_gradient_work.py` and `test_gradient_flow.py`; existing loss/config/loop tests are extended in place. CPU tests use tiny fake modules and fake index/clip artifacts; no real backbone or long training. One small `@pytest.mark.gpu` test in `test_reference.py` covers configured-device CUDA RNG restoration, and one in `test_gradient_flow.py` covers post-original-backward RNG restoration; every other check runs in the fast tier.
- Scalar oracles for BCE/Brier/std/quantiles (empty, singleton, even/odd, interpolation), calibration list length/order/boundaries and empty bins, soft targets including 0/0.5/1, saturation strict inequalities, TP/TN/FP/FN and one-class null balanced accuracy. Explicitly distinguish $\|z\|_2$ from squared norm and hinge.
- Preference output suites on populations with different comparable rates: every step/window/reference field is conditional on comparable pairs and null for no comparable pairs, while additive `P_succ` remains normalized by all pairs and `P=P_inc+P_succ`.
- Stratum oracles for comparable and inc formulas, empty catalogue entries and pooled-window/reference reductions. Global fit on hand index shards; degenerate all-comparable/all-incomparable fits with `sum_p=null` when no comparable train pairs and infinite-baseline nulls; a window uses stored global constants rather than its own optimum. Cover all `load_global_fit` sources and precedence, byte-identical `global-fit.json` and `reference/000000/baselines.json`, mismatch `RuntimeError`, no partial shard 0, and resume without an index scan.
- Two-pass selection against an independent largest-remainder/ordinal oracle: proportional quotas, fractional ties, zero strata, train-then-validation generator order, offsets across shards, no replacement and deterministic stored order. With neither source it performs two scans; with `global-fit.json` it performs only pass 2; with complete shard 0 it performs none. A crash before `global-fit.json` repeats both; a crash afterward repeats only pass 2. Instrument one shard plus bounded selection state, insufficient-pair abort before pass 2, and both bars' totals/lines.
- Interface/config flow: every `preference_suite` argument is required and forwarded from `PretrainConfig`, with no defaults; `WindowAccumulator` receives exactly `global_fit.strata`, all calibration/saturation/inc thresholds, quantiles and `action_copy_eps`, while per-step baselines come from `step_record` rather than another fit argument. Every direct `step_gradients` caller passes the same `cfg`; omitting it raises `TypeError`, and positional order is covered by the existing oracles.
- Each loss `terms` against the existing float64 loss oracle: mathematical component-sum identity, `total` and `forward` bitwise equal to the pre-block implementation, validity masks and single target/frame computation; mixed comparable/incomparable and zero-valid-target cases.
- Step schema and counts on mixed/unlabeled/pair-only micro-batches: `action_targets` counts valid rows rather than nonzero target elements, `shift_targets` counts $\chi^\pi$, and no pairs gives null preference fields. Existing `phi`, `pi`, `P` and `loss` remain equal to the old loop.
- Detached contract: keys, shapes, CPU devices, dtypes, placeholder labels, `q_sum` normalization and bounded retention. Verify transfer after each micro backward, no retained full policy/perception tensors and no raw GPU observations in `WindowAccumulator`.
- Gradient norms against a hand norm, None versus zero gradients, global clipping coefficient and head activity resolved through the explicit `preference` module argument. `update_stats` against actual before/after clones for one AdamW step, with and without weight decay, None gradients, zero parameter norm and `d=0`; groups and parameter order match the optimizer, and undefined `UpdateTerm.before/delta` are null rather than placeholders while `grad is None` retains zero delta.
- Weighted component losses preserve the original training scalar bitwise. `collect_micro` returns exact component gradients by hand, including `phi_dual` and unused parameters, leaves original `.grad` untouched, distinguishes an unlabelled P component from zero, restores post-original-backward CPU/CUDA RNG and counts NaN/+inf/-inf per tensor/group.
- Work manifest/layout and reduction against a small hand vector oracle with unequal micro-batch contributions: step vectors are summed before every norm/dot; cross-micro terms match direct full-step gradients, totals/group values and zero-denominator null cosines are exact, and no per-micro cosine averaging occurs. Manifest and `UpdateTerm.parameter` share stable module-qualified names with exactly one optimizer group per name; `str(id(parameter))` is never used. Pass `Groups` in an order different from manifest collection order and prove output follows `check_groups` order and group membership, with `TypeError` when `groups` is omitted and totals independent of traversal order. Instrument reads to prove one component vector during collection and the component vectors of one parameter across micro-batches during reduction, with uncompressed dtype/shape preservation and work deletion; the progress bar is owned by the caller, not `reduce_work`.
- `update_and_align` against cloned actual pre/post AdamW parameters: both alignment signs, group totals, `phi_dual`, `P` self-update and unavailable handling; every group/total bundle contains `update_dot` and `update_unavailable`, and it returns the same ordinary update statistics as `update_stats` in one shared iterator.
- Counterfactual evaluation on hand fake $\phi,P$: identical raw pair identities/order and labels, shared `evaluation_state` for baseline and post-step, post-update RNG/mode restoration, one post-update $\phi$ call per pair, frozen eval-mode $P_t$ for `Delta_only_phi`, current $P_{t+1}$ for `Delta_joint`, zero-based `micro_batch` matching work shard `<name(j)>` and zero-based labelled-only `pair_in_micro`, exact per-pair JSON and mean/median/quantile/harmful/helpful/zero reductions. Verify module/optimizer/gradient/RNG/mode preservation.
- Selected-step replay with labelled gaps and tiny $T$: first/middle/last labelled steps, floor-middle fallback, deduplicated roles, no global RNG consumption, immutable selection reuse and exact bars. No labelled steps produces the required line and no work.
- Accepted-artifact resume: pending records appear only inside the next published checkpoint; `checkpoint_losses_index` joins the correct preclip/LR record; staged checkpoints leave no accepted record; startup deletes all raw work, repeats unaccepted steps and rebuilds summary from at most three accepted records. Pruning retains `gradient_flow` directories.
- Real-versus-synthetic evidence schema: only accepted real pretraining checkpoint records can support dynamic conclusions; synthetic fixtures never set `evidence.kind="real_pretrain_step"` and cannot close hypotheses 2–4. `evidence.checkpoint_shard`, `evidence.pairs` and summary `metrics` are direct `out_dir`-relative lookup paths.
- Aggregation with unequal pair/clip counts: equal-step loss/baseline means versus pooled conditional distribution/calibration metrics, exact nested `losses`/`coverage.losses`, full `uniform`/`zero_shift` coverage and variable finite-only `copy_current` coverage, including `action_copy_eps=0` with a zero-probability target action producing null rather than Infinity; gradient/update means and counts, unavailable updates, full and final-partial windows, checkpoint divisibility errors, serialization and resume.
- `WindowAccumulator`: consecutive-step validation, full-only `drain(False)`, final full-plus-partial drain, empty final drain, chronological output, discarded state and rejection of a second final drain.
- Fixed initial artifacts: selection order, CPU `Z0` shape/dtype/order and `P0`; complete typed `ReferenceData` including initial pairs in new mode; all four step-0 conditions identical. Initial pair baselines/deltas and positive-step fixed-reference deltas match a hand oracle. After one fake update, hand-check `joint`, `P_only`, `phi_only` and unchanged `initial`. `joint.losses` contains all preference/perception/policy keys; the other conditions contain only preference losses.
- State preservation: parameters, buffers, gradients, optimizer state dict, module mode booleans and CPU RNG plus configured `cfg.device` CUDA RNG bitwise unchanged after normal and raised evaluation; default CUDA device is not read or changed; policy dropout is disabled during evaluation and restored afterward; local numpy selection draws do not touch schedule RNG.
- Cadence/order: no evaluation at step 499, checkpoint 500 written before reference shard 1, no extra final evaluation off cadence, resume before and after reference publication, reuse of shard 0 and its catalogue, rejection of a future shard/gap or mixed legacy/new layout, checkpoint pruning retaining all reference shards. A legacy shard 0 resumes and evaluates later metrics-only shards without reading pairs; new shards always contain both metrics and pairs. Call counts prove 128 pair iterations and no other model evaluation, including outbox recovery after source publication.
- An end-to-end tiny run with diagnostics versus the existing training oracle has identical parameters, optimizer state and training RNG at every checkpoint. Existing checkpoint resume tests retain bitwise training equality. Per-step/reference outboxes recover accepted-source events without forwards and never backfill watermarked history. Bars, `already done`/`nothing to do`, config loader including `diagnostics`, and all revised existing-spec tests. `tests/pretrain/test_config.py::test_starting_values` expects the current/intended `checkpoint_steps: 100`, closing that pre-existing mismatch without changing cadence.

## Library mapping

| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| probabilities | `torch.sigmoid` on detached logits | 2.14.1 | none | installed torch `_refs/__init__.py:1036-1037`: $1/(1+\exp(-x))$ |
| BCE | `torch.nn.functional.binary_cross_entropy_with_logits` | 2.14.1 | `weight=None`, `pos_weight=None`, `reduction="none"` | `specs/preference.md` [T] |
| policy probabilities | `torch.softmax(logits, dim=-1, dtype=None)` | 2.14.1 | `dim=-1`, `dtype=None` | installed torch `_refs/__init__.py:4362-4379`: subtract the dimensionwise maximum, exponentiate, divide by the sum over `dim` |
| deterministic top-1 | `torch.amax`, `torch.amin`, `torch.where` on action indices | 2.14.1 | `dim=-1`, `keepdim=True` for the maximum; nonmaximal indices replaced by $|A|$ | installed torch `_refs/__init__.py:2534-2576`; `specs/preference.md` [T] for `where` |
| norms | `torch.linalg.vector_norm(..., ord=2, dim=-1)` for states and `ord=2` over each parameter | 2.14.1 | state `dim=-1`; parameter gradients summed before the outer square root | `specs/perception.md` [V2]; `specs/pretrain.md` [T] clipping |
| optimizer state | `torch.optim.AdamW.state_dict`, post-step `state` | 2.14.1 | existing optimizer only | `specs/pretrain.md` [T], [A] and (6) |
| component gradients | `Tensor.backward(retain_graph=True)` on selected steps; `torch.autograd.grad` | 2.14.1 | `grad_outputs=torch.ones_like(L_c)`, `retain_graph=True`, `create_graph=False`, `allow_unused=True`, `materialize_grads=False` | installed torch `autograd/__init__.py:434-499`: returned rather than accumulated gradients, graph retention and unused-input semantics |
| modes/gradients | `Module.eval()`, `Module.train(mode)`, `torch.no_grad()` | 2.14.1 | prior boolean mode restored | `specs/policy.md` (2); `specs/perception.md` (6), (9) |
| frozen preference | `copy.deepcopy(P)`, `load_state_dict(P0)` | CPython 3.10.20; torch 2.14.1 | `strict=True` | torch `nn/modules/module.py` state-dict contract; no random initialization in the copy |
| RNG | `torch.get_rng_state`, `torch.set_rng_state`, `torch.cuda.get_rng_state`, `torch.cuda.set_rng_state` | 2.14.1 | `device=torch.device(cfg.device)` explicitly; state restored in `finally`; no default-device call | `specs/pretrain.md` [T] |
| selection | `numpy.random.Generator(PCG64([seed, 4])).permutation(n)` | 1.26.4 | one local generator in part/stratum order during ordinal selection between the two scans | `specs/train-index.md` [N] |
| gradient-step replay | `numpy.random.Generator(PCG64([seed, 3]))` and existing `schedule` | 1.26.4; this repo | a fresh local generator only; training schedule RNG untouched | `specs/pretrain.md` (3); `specs/train-index.md` [N] |
| serialization | `torch.save`, `torch.load(..., map_location="cpu")`, JSON | 2.14.1; CPython 3.10.20 | `weights_only=True` on load; `json.dumps(..., indent=2)+"\n"` | `specs/pretrain.md` [T]; `specs/runs.md` (3) |
| atomic artifacts | `write_shard`, `complete_shards`, `shard_path`, `write_file`, `part_path`, `publish` | this repo | shard order (21) | `specs/shards.md` |
| scalar math | `math.log`, `math.sqrt`, `math.floor`, `sorted` | CPython 3.10.20 | base $e$; quantile (6) and largest remainder implemented directly | CPython 3.10.20 math/list semantics |

## Out of scope
- Any change to training losses, weights, clipping, LR, optimizer, model architecture, labels, VLM sampling or train/validation split.
- Additional full training runs, full-dataset model evaluation, optimizer interventions, or global action-marginal fitting by a full clip-metadata scan.
- Dynamic conclusions from synthetic probes or from scalar training losses without accepted real checkpoint gradient-flow records; those probes test only the instrumentation.
- Backfilling component-gradient work into past steps for which no selected-step records were captured; real dynamic evidence cannot be reconstructed from scalar logs alone.
- Backfilling `Z0/P0` or reference metrics into runs started before this block. Such runs keep the ordinary fingerprint/resume rules; this block defines initialization and resume only for runs whose reference shard 0 was created at step 0.
- Raw direction accuracy of oriented preference labels, per-window preference-constant refitting, NaN/Infinity JSON values, padding or distributed diagnostics.
- Pruning reference shards, resampling a reference on resume, and evaluation at a non-cadence final step.
