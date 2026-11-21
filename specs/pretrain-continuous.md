# Continuous pretraining
Status: approved (2026-11-21)

## Contract

Add a continuous alternative to [classical pretraining](pretrain.md), using
[continuous data](continuous-data.md). User decisions of 2026-11-21: a frame and
its description enter phi once per `delta` kept ticks; full attention sees only
the current observation; Gated DeltaNet retains its convolutional and recurrent
states through the fragment; backpropagation is truncated to K observations;
past preference histories are stored detached and used when a compared clip
completes. Source episode boundaries and annotation clips are not stream resets.
The later decision to reset memory at every continuous-fragment boundary takes
precedence over preserving it across missing recordings of the same world.

The existing implementation lives under `training/pretrain/classical/`; the new
loop lives under `training/pretrain/continuous/`. Modules genuinely shared by
both remain under `training/pretrain/`, in purpose-named subpackages where
needed. Classical losses, scheduling and artifact semantics remain those of
`pretrain.md`; moving modules must preserve that mode's numerical behavior.

There are two mode configs, `configs/training/pretrain/classical.yaml` and
`configs/training/pretrain/continuous.yaml`. The higher-level
`configs/training/pretrain/config.yaml` selects `mode: classical|continuous` and
the corresponding config path. Model/optimizer values shared by the modes live
above their mode-specific configs. The existing `scripts/pretrain.sh` entry
point dispatches the selected mode; data indexes, output paths and checkpoint
formats are mode-specific and reject incompatible resume.

## Stream and gradients

Observation j contributes its image, metadata and latent readout marker in the
existing perception format. Full-attention keys/values contain only tokens of
that observation; attention is causal within it. GDN convolution history and
recurrent matrices persist across observations. Clear only full-attention state
between observations. Reset all recurrent state at a fragment start and at a
new epoch's visit; do not reset at an episode, shard, preference event or update.

Compute M-RoPE positions within the current observation, starting from zero each
time. Relative position metadata uses the first retained observation of the
fragment as its fixed origin. No growing token-ID/position history is retained.
Inference and training implement the same observation/memory semantics.

Within a fragment, consecutive windows of at most `tbptt_states = K` observations
form optimizer steps. The last window may be shorter. After each window's
backward/update, detach every live recurrent state and retained latent history;
keep their numerical values. Never retain an autograd graph for an entire world
or re-encode a historical endpoint to substitute current-weight latents.
The current window's latents retain their graph until backward; a history may
therefore contain an older detached prefix and a differentiable suffix.
Stale state and stale preference histories after updates are accepted
approximations, not gradients of the classical fixed-parameter objective.

The training state interface must be functional: checkpoint recomputation may
not mutate state already consumed by autograd. Stock inference `DynamicCache`
is not an accepted training implementation: the pinned Transformers path drops
cache with gradient checkpointing and mutates recurrent buffers in-place.
Non-reentrant activation checkpointing must preserve both the recurrent values
and the K-window gradient. CPU and Spark GPU backward tests are required.

## Losses and event order

Let `I_phi` be all valid adjacent-grid intervals in the training split under the
existing action/state validity rules, with at least one recognized legal action.
For interval j use exactly the bracket of perception Eq. (11), including its
changed-pair mask, `n_j^2` bound, two endpoint norm penalties and dual
stop-gradients. Sum it once when its right-hand observation becomes available.
An idle or otherwise invalid interval still carries observations through phi,
but does not introduce a new zero-action METRA constraint or an extra norm loss.

Let `I_pi` be grid positions j for which intervals j and j+1 are both valid and
available within the fragment. Use the two terms of policy Eqs. (7)-(9) once
each at every such position, without the classical clip reduction: the history ends
at j, the target actions and mask come from interval/state j+1, and the factual
shift target is `sg(z[j+1] - z[j])`. Preserve the existing one-interval delay,
frequency targets, single-action shift mask and `shift_weight`. Read at most
`policy.history` past observations with known action metadata. Known idle
intervals remain in the history; unknown/unsupported action spans truncate that
history. A history never inserts a fabricated interval. Empty action
input is the existing one-hot no-action token, not an all-zero distribution.
Raw action metadata may be read ahead for targets; future frames/latents must
not be inputs to phi or the policy readout. Score action CE in j's own window,
using raw target metadata; this keeps its phi gradient even for K=1. Score the
shift term when z[j+1] arrives. Within one window it reuses j's readout/dropout.
Across a K boundary, recompute the policy readout/shift from the saved detached
history ending at j, with the updated policy weights and a new saved-RNG dropout
draw. Do not add z[j+1] to that input history. This delayed shift trains pi but
has no phi gradient across the boundary; it is an explicit TBPTT approximation.
Pending shift snapshots include the exact latent/action history and indices.

Clips exist only as preference endpoints. At completion of a mapped clip, gather
the final `preference.history` latents of its mapped full history (or the shorter
stored history as allowed by the preference model). For every original labeled
edge whose other endpoint has already completed, call P on those histories and
score the existing preference loss, including both-order weighting and original
A/B orientation. The older endpoint's **entire history is detached**, even if
it was encoded in the current K-window. The current endpoint keeps only its
available within-window gradient. Store its detached history for later events.

Process each original retained edge exactly once per epoch, at completion of its
second endpoint. Simultaneous endpoint completions have a deterministic order
by canonical mapped history ID; edge order is by original edge ID. Distinct
original edges are not averaged or deduplicated after mapping. The archive
survives fragment changes, including comparisons between fragments/world groups;
clear it at the next epoch. Store it in bounded-memory disk shards, with pending
edge counts, and retire histories only after every incident edge is consumed.
The continuous index also excludes `collapsed_pair` edges whose actual P input
observation-ID suffixes coincide, even if their full annotated histories differ.
Fingerprint `preference.history` and report this filtering separately.

Let `N_phi`, `N_pi`, `N_P` be the sizes of these three complete training
populations, and M the number of K-windows per epoch. For update window B,

\[
 L_B=M\left[
 \frac{w_\phi}{N_\phi}\sum_{j\in I_\phi(B)}\ell^\phi_j+
 \frac{w_\pi}{N_\pi}\left[\sum_{j\in C(B)}\mathrm{CE}_j+
 \sum_{j\in S(B)}w_\Delta\chi_j d^2_j\right]+
 \frac{w_P}{N_P}\sum_{r\in R(B)}\ell^P_r\right].
\]

C(B) and S(B) are respectively the CE and completed-shift events of that window;
each eligible policy position contributes once to each across the epoch, with
the same N_pi denominator. Count these two event streams separately.
Omit an empty population and its denominator. An entirely empty loss window
advances the stream/schedule with all gradients `None`; it performs no parameter
decay/update. For fixed realized losses, the arithmetic mean over windows equals
the three population means. Do not divide by a clip's degree, event-local pair
count, fragment length or current window size. There is no extra own loss when
a preference endpoint completes. Pair micro-batching may bound P's working
memory but uses these same denominators and sums gradients before the update.

Shuffle fragment order per epoch with the saved seeded schedule RNG; preserve
chronology inside each fragment. Reuse classical optimizer groups, precision,
clipping and learning-rate schedule on the resulting step count. References to
world-level splitting mean continuous-data's conservative world groups, never
an independently shuffled/split part of the same recurrent history.

## Interfaces and artifacts

- Continuous config: complete dataset/index paths, `tbptt_states`, bounded
  preference micro-batch/archive/cache limits, output path and the existing
  shared optimization/model/diagnostic settings. K is configurable and positive;
  its initial performance setting is verified on the Spark before a full run.
- Perception exposes a differentiable observation step returning `(z, state)`
  and an inference adapter with the same semantics. `state` includes GDN
  convolution/recurrent values and fragment metadata origin, never historical
  full-attention K/V. The classical interface remains supported.
- A checkpoint atomically records weights, optimizer/schedule, torch/CUDA and
  ordering RNG, epoch/fragment/source/shard/grid cursor, recurrent state, recent
  latent/action history, pending own targets, preference archive snapshot and
  consumed-edge position. Checkpoints are taken only after complete updates and
  detach. Archive publication and checkpoint references must recover together;
  a restart neither repeats nor skips a labeled edge or ordinary target.
- Model/run metadata declares `pretrain_mode: continuous` and the perception
  attention/state semantics. Existing consumers must use the compatible stream
  adapter or reject that artifact, never silently reset every classical history
  length. This contract does not otherwise change RL/MCTS mathematics.
- Diagnostics preserve the shared reporting interfaces with explicit interval,
  target and pair counts replacing classical clip denominators. Reference
  evaluation uses frozen weights, a fresh archive/state, and complete prefixes
  of selected fragments in chronological order. It never encodes a reference
  clip independently or contaminates training RNG/state. Gradient diagnostics
  must include the same truncated and detached paths as the actual update.
- With `retrain: true`, keep the existing frozen-phi phase: regenerate continuous
  latents from fragment starts with final frozen phi; retain pi/P weights, use
  a fresh heads-only optimizer/schedule and the same retained labels. The phase
  has its own fully resumable stream/archive/checkpoint state.

## Acceptance

- Classical mode retains its existing loss, gradient, scheduling and resume
  oracles after the directory/config dispatch change.
- Clearing full attention alone preserves recurrent influence from earlier
  observations; a fragment reset removes it. Attention cannot read a token from
  an earlier observation. State/token storage does not grow with fragment age.
- A tiny deterministic recurrent model matches an independent unroll for both
  values and gradients inside K; gradients do not cross a detach boundary,
  values do. K=1, partial windows and non-reentrant recomputation pass backward.
  At K=1, CE trains phi, the delayed shift trains pi without a phi gradient,
  and each eligible CE/shift event is counted exactly once.
  The corresponding real Qwen CUDA/FLA transition runs on the Spark.
- Own transitions crossing episode, shard and K-window boundaries are counted
  once; invalid/idle intervals and fragment ends have the stated masks. Changing
  preference degree does not change own-loss population weights.
- Every retained original edge is consumed once, including cross-fragment edges,
  ties, aliased endpoints and both A/B temporal orders. The older history has
  no gradient; the current history has exactly the permitted K-window gradient.
- Interrupted/resumed runs reproduce uninterrupted CPU outputs, optimizer state,
  counts and edge consumption; GPU comparisons use the established numerical
  tolerance for the pinned recurrent kernels. Reference and retraining paths
  exercise real stream prefixes rather than isolated clips.
