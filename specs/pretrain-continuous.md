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

There are two mode configs, `configs/training/pretrain/classical/config.yaml` and
`configs/training/pretrain/continuous/config.yaml`. The higher-level
`configs/training/pretrain/config.yaml` selects `mode: classical|continuous` and
the corresponding config path. Model/optimizer values shared by the modes live
above their mode-specific configs. The existing `scripts/pretrain.sh` entry
point dispatches the selected mode; data indexes, output paths and checkpoint
formats are mode-specific and reject incompatible resume.

The pretrain config directory keeps only `config.yaml` (selection) and
`base.yaml` (shared settings) at its root. Classical index variants live in
`classical/index/`; continuous indexing uses `continuous/index.yaml`.
The six former flat config paths remain read aliases when their files are absent;
an existing file at an old path takes precedence. Config reads outside the
repository are not redirected. Pretrain index and sampling fingerprints use a
scoped compatibility hash: for the relocated pretrain configs and selector,
reverse only the known path substitutions before hashing the same bytes and
inheritance graph. The general raw-byte config hash is unchanged. Training resume
still checks all training/model/diagnostic config content. Index compatibility
checks only data-defining inputs: dataset, labels, index settings, preference
history/config and action-mask semantics. Changing K, B or optimizer settings
does not require rebuilding the index. Complete existing indexes retain their
metadata bytes, preserving downstream training identities.

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

For every new observation t, `tbptt_states = K` defines a sliding gradient
window `[max(fragment_start, t-K+1), t]`, including the current observation.
Replay this window with current weights from a detached memory anchor immediately
before its first observation. Replace the live suffix's latents/states with the
replayed values. When the window advances, its next anchor is the detached state
after the first replayed observation, not the original historical state at that
position. Earlier memory is never recomputed. Preserve the fragment origin.
Historical stochastic draws are replayed without advancing the global RNG again.
Only newly available losses/events at t are scored; replay never repeats labels,
archive publications, counters or ordinary losses. Histories before the sliding
cutoff and archived earlier preference endpoints remain detached.

With fixed weights and saved stochastic draws, overlapping observations have
identical values. The implementation may reuse at most K local observation graphs
with detached state inputs. For each event, propagate its state cotangent through
exactly its K-window; accumulate output cotangents and apply each observation's
parameter/encoder VJP once before retiring it. Flush all remaining VJPs before
clipping or updating, then discard these ephemeral graphs. This uses linearity of
the VJP and preserves the loss and gradient horizon; floating-point summation order
may differ from independent full replay. No cached graph crosses a weight update.
Token-embedding cotangents may be concatenated across the optimizer batch and
applied by one vocabulary-embedding VJP, avoiding a dense vocabulary gradient
allocation for every observation. Flush these contributions at fragment resets
and before clipping/updating, preserving unused/frozen parameters.

Independently, `batch_size = B` counts new observations, including idle or invalid
ones. Accumulate their gradients, then clip and update once. B may be smaller or
larger than K. Batches cross fragment boundaries (reset only stream memory there),
and the final batch of an epoch may be shorter. An optimizer update never clears
the replay inputs or anchor: the next event rebuilds the full available K-window
with updated weights. The anchor and archived histories may contain older-weight
values; this is an explicit approximation, not full-history current-weight BPTT.

The training state interface must be functional: checkpoint recomputation may
not mutate state already consumed by autograd. Stock inference `DynamicCache`
is not an accepted training implementation: the pinned Transformers path drops
cache with gradient checkpointing and mutates recurrent buffers in-place.
The boolean `activation_checkpointing` controls non-reentrant activation
checkpointing of phi. Initially false, it keeps the bounded K-observation
activations to avoid repeated forward recomputation; true trades speed for memory.
Older materialized configs that omit it retain checkpointing.
Both settings must preserve recurrent values and the K-window gradient. CPU and
Spark GPU backward tests are required.
Reference evaluation may retain the backbone autocast's parameter casts across
its chronological prefixes. The cache must close before returning to training or
restoring buffers, so it cannot survive a weight update or change head precision.

## Losses and event order

Let `I_phi` be all valid adjacent-grid intervals in the training split under the
existing action/state validity rules, with at least one recognized legal action.
For interval j use exactly the bracket of perception Eq. (11), including its
changed-pair mask, `n_j^2` bound, two endpoint norm penalties and dual
stop-gradients. Sum it once when its right-hand observation becomes available.
An idle or otherwise invalid interval still carries observations through phi,
but does not introduce a new zero-action METRA constraint or an extra norm loss.

Let `I_pi` be grid positions j whose current interval j is valid and available
within the fragment. Use the two terms of policy Eqs. (7)-(9) once
each at every such position, without the classical clip reduction: the history ends
at j, each input action precedes its latent state, and the target actions and mask
come from the current interval/state j. The same current action's factual shift
target is `sg(z[j+1] - z[j])`. Every valid outgoing interval contributes one
position, including the first and last intervals; no second valid interval is
required. Preserve frequency targets and `shift_weight`; the shift mask requires
exactly one action in the current interval, allowed at its starting state. Read at most
`policy.history` past observations with known action metadata. Known idle
intervals remain in the history; unknown/unsupported action spans truncate that
history. A history never inserts a fabricated interval. Empty action
input is the existing one-hot no-action token, not an all-zero distribution.
Raw action metadata may be read ahead for targets; future frames/latents must
not be inputs to phi or the policy readout. Score action CE in j's own window,
using raw target metadata; this keeps its phi gradient even for K=1. Score the
shift term when z[j+1] arrives, recomputing j's policy readout from its refreshed
history with the current event's sliding cutoff. Reuse j's saved dropout draw,
including across optimizer updates, without advancing the global RNG again.
Do not add z[j+1] to that input history. The detached factual target does not
differentiate through z[j+1], so shift reaches at most K-1 phi observations;
at K=1 it trains pi only. Pending snapshots include the latent/action history,
indices and RNG needed to reconstruct the readout.

The initial observation and the first observation after unknown/disconnected
action metadata use the no-action input. A known idle interval preserves the
history and supplies the same no-action input to its successor.

At the same CE event, evaluate the policy change probability `p_j` on exactly
the same readout/dropout realization. For each allowed occurrence of a current
action a, let `r_{j,a}` indicate membership of its name in the preceding raw
interval. Previous multiplicities and masks do not change that indicator;
current duplicates retain their multiplicity in the frequency target `q_{j,a}`.
Let `g_j` indicate that this preceding interval exists, is adjacent and has known
action metadata. Its own loss validity is not required. Define

\[
 \ell^{change}_j=g_j\sum_a q_{j,a}
 [r_{j,a}p_j+(1-r_{j,a})(1-p_j)]
 [-\log\operatorname{softmax}(\lambda_j)_a].
\]

This auxiliary term differentiates both the action CE and change probability;
it does not replace ordinary CE or shift. First/unknown/disconnected predecessors
exclude only this term. A known idle predecessor has an empty membership set,
so every current action is a change. The membership mask is independent of
action-input dropout. Score this term in j's own window, retaining the same
sliding-window phi gradients as CE, including K=1. `w_change=0` omits its graph
and leaves otherwise unused change-head parameters without gradients.

Clips exist only as preference endpoints. At completion of a mapped clip, gather
the final `preference.history` latents of its mapped full history (or the shorter
stored history as allowed by the preference model). For every original labeled
edge whose other endpoint has already completed, call P on those histories and
score the existing preference loss, including both-order weighting and original
A/B orientation. The older endpoint's **entire history is detached**, even if
it was encoded in the current K-window. The current endpoint keeps only its
available sliding-window gradient. Store its detached history for later events;
never rewrite this archived endpoint when its live observations are replayed.

Process each original retained edge exactly once per epoch, at completion of its
second endpoint. Simultaneous endpoint completions have a deterministic order
by canonical mapped history ID; edge order is by original edge ID. Distinct
original edges are not averaged or deduplicated after mapping. The archive
survives fragment changes, including comparisons between fragments/world groups;
clear it at the next epoch. Store it in bounded-memory disk shards, with pending
edge counts, and retire histories only after every incident edge is consumed.
`archive_keep_shards` is a required positive integer, initially 2. After a
checkpoint is durably published and old checkpoint payloads are pruned, keep
the newest `archive_keep_shards` latent shards of the current epoch **plus**
every shard required by current live histories or any retained checkpoint.
This is a retention floor for recent shards, not a hard disk quota; necessary
histories may exceed it. Older epochs keep only checkpoint-required payloads.
Delete other `histories.pt` files and unreferenced metadata snapshots, preserving
numbered directories for contiguous shard publication and rollback. Old epochs'
working metadata databases may be removed; checkpoint snapshots remain the
recovery authority. Validate retained snapshot checksums and the presence of
all referenced payloads before deleting anything. Garbage collection never drops a comparison,
re-encodes a history, or changes the loss, update order or RNG state.
The continuous index also excludes `collapsed_pair` edges whose actual P input
observation-ID suffixes coincide, even if their full annotated histories differ.
Fingerprint `preference.history` and report this filtering separately.

Let `N_phi`, `N_pi`, `N_P` be the sizes of these three complete training
populations. Let U = ceil(N_observations / batch_size) be the number of optimizer
batches per epoch, without rounding separately per fragment. For update batch A,

\[
 L_A=U\left[
 \frac{w_\phi}{N_\phi}\sum_{j\in I_\phi(A)}\ell^\phi_j+
 \frac{w_\pi}{N_\pi}\left[\sum_{j\in C(A)}\mathrm{CE}_j+
 \sum_{j\in S(A)}w_\Delta\chi_j d^2_j\right]+
 \frac{w_{change}}{N_\pi}\sum_{j\in C(A)}\ell^{change}_j+
 \frac{w_P}{N_P}\sum_{r\in R(A)}\ell^P_r\right].
\]

C(A) and S(A) are respectively the CE and completed-shift events of that batch;
each eligible policy position contributes once to each across the epoch, with
the same N_pi denominator. Count these two event streams separately.
Omit an empty population and its denominator. An entirely empty loss batch
advances the stream/schedule with all gradients `None`; it performs no parameter
decay/update. For fixed realized losses, the arithmetic mean over batches equals
the weighted population means, with the auxiliary term using the same N_pi
denominator and its independent nonnegative `w_change`. Do not divide by a clip's degree, event-local pair
count, fragment length, B or actual tail-batch size. There is no extra own loss when
a preference endpoint completes. Pair micro-batching may bound P's working
memory but uses these same denominators and sums gradients before the update.

Shuffle fragment order per epoch with the saved seeded schedule RNG; preserve
chronology inside each fragment. Reuse classical optimizer groups, precision,
clipping and learning-rate schedule on the resulting step count. References to
world-level splitting mean continuous-data's conservative world groups, never
an independently shuffled/split part of the same recurrent history.

## Interfaces and artifacts

- Continuous config: complete dataset/index paths, `tbptt_states`, `batch_size`, bounded
  preference micro-batch/archive/cache limits, output path and the existing
  shared optimization/model/diagnostic settings. K and B are positive integers;
  its initial performance setting is verified on the Spark before a full run.
- Perception exposes a differentiable observation step returning `(z, state)`
  and an inference adapter with the same semantics. `state` includes GDN
  convolution/recurrent values and fragment metadata origin, never historical
  full-attention K/V. The classical interface remains supported.
- A checkpoint atomically records weights, optimizer/schedule, torch/CUDA and
  ordering RNG, epoch/fragment/source/shard/grid cursor, recurrent state, recent
  latent/action history, replay inputs and their RNG, detached replay anchor,
  pending own targets with policy RNG, preference archive snapshot and
  consumed-edge position. Checkpoints are taken only after complete updates and
  detach. Archive publication and checkpoint references must recover together;
  a restart neither repeats nor skips a labeled edge or ordinary target.
  Checkpoint `metadata.json` also records the epoch and archive descriptor so
  garbage collection can discover dependencies without loading model weights.
  Older checkpoint headers fall back to their checkpoint payload. Restoring
  an archive rejects a missing payload referenced by its live histories, while
  accepting collected payloads that its snapshot no longer needs. Config
  fingerprint checks remain unchanged, including for the new retention field.
- Model/run metadata declares `pretrain_mode: continuous` and the perception
  attention/state semantics. Existing consumers must use the compatible stream
  adapter or reject that artifact, never silently reset every classical history
  length. This contract does not otherwise change RL/MCTS mathematics.
- Diagnostics preserve the shared reporting interfaces with explicit interval,
  target and pair counts replacing classical clip denominators. Reference
  evaluation uses frozen weights, a fresh archive/state, and complete prefixes
  of selected fragments in chronological order. It never encodes a reference
  clip independently or contaminates training RNG/state. The continuous config
  sets `reference_steps: 5000`; the initial reference at step zero remains.
  Gradient diagnostics must include the same truncated and detached paths as the
  actual update.
  `gradient_diagnostics: false` skips gradient-flow selection/collection and all
  gradient/update statistics; their record fields are null, without changing
  weights, optimizer, RNG or loss/reference records.
  Observations with preference endpoints, and all observations when gradient
  diagnostics are enabled, run their complete sliding forward/backward in eager
  compiler stance so retained graphs survive repeated backward. FP8 arithmetic
  remains active. Reused observation graphs also run in eager compiler stance:
  their state VJPs retain graphs until the final parameter VJP. Component-gradient
  diagnostics use independent replay so every component retains direct paths to
  the model parameters.
  The required boolean `publish_diagnostics`, initially false, independently
  controls the shared diagnostic publisher. When false, no DiagnosticRun is
  loaded, registered, recovered, staged or accepted; no global catalogue/index
  scans or source hashing run. Local losses, window aggregates and fixed-reference
  artifacts remain available. The reference shard count is scanned once on startup
  and advanced after each atomic publication. Disabling publication preserves
  local results and checkpoint continuation.
  Raw components report `pi_change` separately, averaged over CE events including
  excluded predecessors as zero; raw `pi` remains ordinary CE plus shift.
  The weighted `pi` component and its gradient
  role include both `w_pi`-weighted CE/shift and `w_change`-weighted auxiliary loss,
  counted only once in the total loss.
- With `retrain: true`, keep the existing frozen-phi phase: regenerate continuous
  latents from fragment starts with final frozen phi; retain pi/P weights, use
  a fresh heads-only optimizer/schedule and the same retained labels. The phase
  has its own fully resumable stream/archive/checkpoint state.

Continuous indexes use `mcpref-continuous-index-v2`: `N_pi` counts every valid
current interval. Version 1 indexes must be rebuilt in a new index output
directory from the existing raw dataset. Existing v2 indexes remain readable
without rewriting: ignore their legacy `tbptt_states` and `training_config`
fingerprint fields and any precomputed `M`. Compute U from the observation
population and current B at training time; new indexes omit these obsolete
fields. All other compatibility checks remain active.

Training run headers and checkpoints carry
`policy_alignment: "preceding-action-v1"`; old or missing alignment tags reject
resume. Continuous run headers, checkpoints and model exports also carry
`gradient_semantics: "sliding-replay-v1"`. Old or missing gradient semantics
reject training resume, including completed model shortcuts and retraining.
Use a new output directory for the revised gradient contract; saved older
weights remain usable for inference and are not retroactively retrained.

## Acceptance

- Archive collection preserves old live histories and every retained checkpoint,
  frees unreferenced payloads across epochs, retains the current epoch's recent
  shard floor, and preserves continuation after interrupted publication or GC.
  Missing protected payloads or missing/corrupt metadata snapshots abort
  collection before any deletion. Restored model, optimizer, RNG and comparison
  coverage match an uninterrupted run without garbage collection.

- Both pretraining modes follow the preceding-action input and current-transition
  target contract of `specs/policy.md`; classical scheduling and resume oracles
  remain valid within the new alignment version.
- Clearing full attention alone preserves recurrent influence from earlier
  observations; a fragment reset removes it. Attention cannot read a token from
  an earlier observation. State/token storage does not grow with fragment age.
- A tiny deterministic recurrent model matches an independent unroll for both
  values and gradients inside K, including the refreshed advancing anchor;
  gradients do not cross the sliding cutoff, values do. K=1, partial windows,
  B<K, B=K, B>K and non-reentrant recomputation pass backward across updates.
  B counts new observations across fragments, the epoch tail updates once, and
  replay does not advance global RNG or duplicate events. Checkpoint resume
  restores replay inputs, anchor, dropout draws and accumulated update schedule.
  At K=1, CE trains phi, the delayed shift trains pi without a phi gradient,
  and each eligible CE/shift event is counted exactly once.
  The corresponding real Qwen CUDA/FLA transition runs on the Spark.
- Own transitions crossing episode, shard and optimizer-batch boundaries are counted
  once; invalid/idle intervals and fragment ends have the stated masks. Changing
  preference degree does not change own-loss population weights.
- Per-occurrence change weighting preserves current duplicate counts and ignores
  previous multiplicity/masks/dropout. Unknown predecessors exclude only the
  auxiliary term; known idle predecessors mark all current actions as changes.
  Independent gradients, K=1 continuation, frozen-phi retraining, zero auxiliary
  weight and exact checkpoint resume preserve these semantics.
- Every retained original edge is consumed once, including cross-fragment edges,
  ties, aliased endpoints and both A/B temporal orders. The older history has
  no gradient; the current history has exactly the permitted K-window gradient.
- Interrupted/resumed runs reproduce uninterrupted CPU outputs, optimizer state,
  counts and edge consumption; GPU comparisons use the established numerical
  tolerance for the pinned recurrent kernels. Reference and retraining paths
  exercise real stream prefixes rather than isolated clips.
