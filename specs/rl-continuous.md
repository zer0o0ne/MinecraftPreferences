# Continuous RL

Status: approved (2026-11-22): bounded replay from retained sampling memory,
with K=4 and S=64, including cross-cycle live continuation.

This extends [RL cycle](rl-cycle.md) to continuous perception. The existing
classical mode is unchanged. The user requested uninterrupted sampling memory
also across RL updates and approved the conditional replay objective below.

## Live world

Use the observation-local attention, persistent GDN state and fixed world origin
of [continuous pretraining](pretrain-continuous.md). Death, respawn, trajectory
shards and RL cycle boundaries do not reset perception memory.

At the end of sampling, state s_U has already been encoded. After joint training
and optional frozen-phi retraining, retain the numerical GDN state and latent /
preceding-action suffix ending at U. The next decision uses this retained history
as its base; new pi/P evaluations use the new weights, while existing MCTS caches
remain stale. The new phi first processes s_(U+1); do not encode s_U a
second time. Offline replay never replaces the live state. As in the existing
model-change handover, replace only the MCTS base with this retained REAL history
in float64; preserve nodes, coordinates, statistics and cached values. Subsequent
action selection and re-rooting retain MCTS's existing cache/reuse rules.

Accept old-weight recurrent values as in continuous pretraining. This is not a
reconstruction of the entire past with the latest weights. Only the validated
handover from the preceding RL cycle authorizes a model change; ordinary
fixed-model resume retains its identity checks.

## Conditional replay

Keep the mixed fresh / pending / preference index, random schedule, loss weights,
global normalization, optimizer continuity and stop-gradient targets of RL cycle.
Replace classical restart-based history reconstruction with the following
item-conditioned branches. Let L = max(policy.history, preference.history).

Positive-integer continuous RL config values, independent of pretrain TBPTT:

- `continuous.gradient_history: 4`: K, the gradient lookback through the primary
  endpoint; a fresh transition additionally differentiates its successor.
- `continuous.replay_stride: 64`: S, the spacing of sampling-memory snapshots.

At global state indices a divisible by S, persist immutable A_a: the GDN state
immediately BEFORE s_a, the original world origin and sampling-model provenance.
A_0 is empty. These are detached sampling states, never refreshed by training.

For a primary endpoint t, define:

    g(t) = max(0, t - K + 1)
    r(t) = max(0, t - max(L, K) + 1)
    a(t) = S * floor(r(t) / S)

Starting at detached A_a(t), replay observations a(t)..g(t)-1 with current phi
under no_grad, then g(t)..t with gradients. Read the last L latent rows and their
actual preceding actions; heads apply their own history limits. Thus every head
row is re-encoded by current phi conditional on the recorded anchor. Replay keeps
the original world origin and never resets at a snapshot or cycle boundary.

- Fresh t: continue the SAME branch through t+1 with gradients. Origin and factual
  successor therefore share one recurrent trajectory. The graph spans at most
  K+1 observations. When PPO is eligible, detach BOTH state and history after
  t+1, then continue t+2..t+H under no_grad for reward-only values.
- Pending t: encode its origin as above, detach state and history, then continue
  t+1..t+H under no_grad. Supply PPO only, as before.
- Preference: independently reconstruct each endpoint by the primary-endpoint
  rule; both sides retain their local phi gradients. Labels stay fixed.

Denote the resulting history Y_(t+k | t). After weight updates it may differ from
Y_(t+k | t+k), whose primary endpoint selects another anchor. This contract accepts
this conditional approximation; it does not claim a unique full-past history
under current phi. Each transition and its reward continuation use one branch.

Pair generation and final-memory publication use the same primary-endpoint rule
in eval/no_grad mode. Frozen retraining precomputes fixed advantages using the
same item branches and preserves the existing optimizer/freeze contract.

## Persistence and cost

Publish anchors atomically with trajectory shards. Retain all anchors and raw
observations referenced by pending or historical preference data. Fingerprints
include the replay config and anchor contents; model/checkpoint artifacts carry
the continuous mode. Reject missing or incompatible anchors before training;
never substitute empty memory. Migration of old shards without anchors is not
part of this change, and those shards are not rewritten or deleted.

Through primary endpoint t, at most S-1+max(L,K) observations are replayed. Fresh
items add one differentiable successor and at most H-1 reward-only observations;
pending items add at most H reward-only observations. This bound
does not grow with world age. Disk usage grows with stored observations/anchors;
this is not a bound on the existing MCTS base-history storage.

For the pinned Qwen3.5-2B config (18 GDN layers, 16 heads, 128x128 recurrent state,
6144x4 convolution state), FP32 recurrent plus BF16 convolution tensors occupy
18.84375 MiB per anchor, excluding serialization overhead. S=64 therefore adds
about 2.9 GiB per 10,000 observations. This is a structural estimate; verify the
actual CUDA state dtype and artifact sizes during implementation. K/S are initial
engineering choices, not measured optimal hyperparameters.

## Acceptance

- A two-cycle run with changed phi/pi/P keeps live memory and consumes each real
  observation once. Exact CPU resume equals uninterrupted execution.
- Independent recurrence oracles check anchor indices, origin/action metadata,
  cross-cycle pending tails, old preference endpoints, local gradient reach and
  reward-only stop-gradients, including K=1 and L greater/less than K.
- Fresh origin/successor and PPO futures share their anchor even across snapshot
  boundaries. Replay cannot mutate saved anchors or the live sampling state.
- At a cycle endpoint U divisible by S, A_U precedes s_U, live state follows s_U,
  and the next cycle neither overwrites A_U nor re-encodes s_U.
- Real Qwen CPU and Spark CUDA tests cover recurrent backward, activation
  checkpointing, changed-weight continuation and serialized resume.
- Exercise joint training, frozen retraining, pair production and final-memory
  publication; run affected tests, Ruff and the full fast suite.
