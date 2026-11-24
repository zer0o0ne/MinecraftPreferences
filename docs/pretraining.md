# Data collection and pretraining

Run these commands from the project checkout on the Spark, after `uv sync` and
the GUI-mask prerequisites described in `specs/continuous-data.md` are complete.
Model configs and the selected training config determine the device.

```bash
bash scripts/collect_perception_pretrain_data.sh
bash scripts/pretrain.sh
```

The first command builds/resumes the shared `data/pretraining` observation store,
continuous fragments, mapped original preferences and filtered classical clip
views. Both trainers require a completed dataset. Repeating a command resumes
from its last complete artifact; changed input fingerprints require a separate
output directory.

## Selecting the trainer

Edit `configs/training/pretrain/config.yaml`:

```yaml
mode: classical  # or continuous
classical:
  config: configs/training/pretrain/classical/config.yaml
  index: configs/training/pretrain/classical/index/shared.yaml
continuous:
  config: configs/training/pretrain/continuous/config.yaml
  index: configs/training/pretrain/continuous/index.yaml
```

Then run the same `bash scripts/pretrain.sh`. It validates the selected paths,
builds/resumes that mode's index and starts/resumes that mode's training.

Continuous training allows one process per output directory. A second invocation
fails before index building or model loading while the first holds the run lock;
the index builder also protects its own output directory. Locks release on process
exit, including crashes. The sibling `<out_dir>.lock` files remain in place and
must not be deleted while a process is running.

- `base.yaml` contains shared model, objective and diagnostic settings and
  inherits `configs/training/optimizer.yaml`.
- `classical/config.yaml` uses clip batches and writes `data/pretrain_classical`.
  Its `index/` directory contains the original `config.yaml` and the shared-data
  variants `shared.yaml` and `shared-labeled.yaml`.
- `continuous/config.yaml` accumulates gradients for `batch_size` new observations
  per update (currently 16). Each observation replays its own sliding window of
  `tbptt_states` observations (currently 4), including across optimizer updates,
  from the saved detached memory immediately before that window. Older GDN memory
  persists through the fragment. Output is `data/pretrain_continuous`.
  `preference_micro_batch` bounds the number of preference pairs per call.
  Archive shard/cache limits bound the historical-latent working set.
- `precision: fp32` in `base.yaml` stores weights, gradients and Adam moments in
  FP32; `bf16` selects BF16 storage. Backbone forward still uses BF16 autocast.
  The continuous leaf selects `fp32` to retain small optimizer updates.
  BF16 storage has no FP32 master weights: small AdamW updates can round to zero,
  including the dual and normalization parameters, even with nonzero gradients.
  FP32 storage preserves these small updates while retaining backbone BF16 compute.
  Changing precision requires a fresh training output directory.
- Continuous CUDA training enables deterministic backward before model construction,
  including the repeated attention VJPs in the sliding window. The launcher sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` when absent; `:16:8` is also accepted. It disables
  cuDNN benchmarking and unnecessary filling of uninitialized allocations. Torch
  flags are restored when the call exits. Direct Python callers constructing CUDA
  modules themselves should enter `continuous.numerics.reproducible` first.
- `gradient_diagnostics: false` skips gradient/update measurements and gradient-flow
  artifacts. Continuous `publish_diagnostics: false` also disables the shared
  catalogue publisher and its repeated historical scans. Local losses and fixed
  reference evaluation remain enabled.
- `retrain: true` adds a resumable frozen-perception phase. The final `model.pt`
  includes its trained heads; continuous also retains `joint-model.pt`.

Previously recorded pretrain config paths remain readable after this relocation.
Pretrain index/sampling fingerprints preserve their old identities for these
path-only changes, including inherited configs. Existing dataset/index metadata
does not need migration; changes to configuration contents still invalidate resume.

Switching modes selects separate index, checkpoint and diagnostic directories.
The continuous index fingerprints its dataset and preference semantics; changing
K, B or training precision alone does not rebuild the index. Training checkpoints
fingerprint the complete selected configuration, so use a new training output
directory when changing it. The
continuous index's `training_config` must point to the selected continuous leaf.
`checkpoint_steps` must be divisible by `diagnostic_window`, and
`reference_steps` by `checkpoint_steps`.
The continuous leaf sets these to 1000, 1000 and 5000 respectively; classical
settings remain separate.

Before continuous diagnostics start, stderr reports the indexed training
observations/fragments, optimizer update count, B/K, storage precision, activation
checkpointing and all checkpoint/diagnostic/reference cadences. These counts do
not estimate runtime. `diagnostic_window` aggregates ordinary update records;
it does not set the length of a reference prefix.

With training preference pairs, a missing initial reference baseline requires a
forward replay from each selected fragment's start through its last selected
endpoint, before the first optimizer update. The report gives the actual selected
pair count and number of unique prefix fragments; `reference_pairs` does not bound
the number of replayed observations. A complete saved baseline is reused. Each
scheduled reference performs fresh prefix replay with the current weights, even
when both `gradient_diagnostics` and `publish_diagnostics` are false. An interrupted
reference pass without a complete published shard restarts those prefixes;
there is no checkpoint inside the pass.

Changing cadence, diagnostic flags, B/K or precision changes the training config
fingerprint and cannot resume an existing training output unchanged. It does not
require rebuilding a compatible continuous data index. Choose the settings before
the long run; use a separate training output directory for a changed configuration.

Continuous model artifacts declare observation-local attention and fragment
memory. Gameplay supports them: set `pretrain` in
`configs/runtime/sampling.yaml` to `configs/training/pretrain/continuous/config.yaml`
(or the selector with `mode: continuous`), set `checkpoint_dir` to a continuous
model or published checkpoint directory. Use `mode: pretrained` for fixed weights
or `mode: rl` for sampling followed by joint training. Run `bash scripts/play.sh`
with a new gameplay output directory.
The model's GDN memory and coordinate origin persist through all trajectories
of that world; `perception.history` does not reset or limit them. Only the
explicit latent buffer is capped at `max(policy.history, preference.history)`.
The sampler persists this state for same-weight continuation with a matching
external world; the play launcher still starts a new world per invocation.

In RL mode, `configs/training/rl/cycle.yaml` sets:

```yaml
continuous:
  gradient_history: 4
  replay_stride: 64
```

The sampler saves detached GDN memory before observations 0, 64, 128, ... .
Training loads the snapshot before the needed head/gradient history, replays the
prefix without gradients, then differentiates the final four observations through
the endpoint (plus the successor for a fresh transition). PPO reward-only futures
continue the same branch without gradients. Both preference endpoints get their
own local phi gradients. Head histories are re-encoded under current weights;
the anchor itself retains its old sampling-time values. This is conditional
replay, not full-world re-encoding with the latest phi.

The live world's memory stays separate from replay. At an RL boundary U, s_U has
already been consumed: retain its state/history, install the new weights, and
first apply the new phi to s_(U+1). Frozen-head retraining and historical
preferences use the same replay rule. State snapshots and their hashes are part
of trajectory shards; old trajectories without anchors cannot start continuous
RL without migration. See [Continuous RL](../specs/rl-continuous.md) for exact
indices, stop-gradients and resume rules.

## New preference annotations

After shared data collection completes, start the VLM service as usual and run:

```bash
bash scripts/serve_vlm.sh
bash scripts/collect_preferences_pretrain_data.sh
```

New clips are selected from the completed shared dataset. The command creates
the pool, VLM responses, corrected labels and `data/train_index_shared_labeled`.
The default pretraining selection uses the migrated original annotations. To
include the new annotations in classical training, select
`configs/training/pretrain/classical/index/shared-labeled.yaml` as `classical.index` and
set the classical leaf's `index_dir` to `data/train_index_shared_labeled`.
For continuous training, set `pair_clips_dir: data/pair_clips_shared` and
`pair_labels_dir: data/pair_labels_shared` in its index config. Use new output
directories when changing a completed index or training run's inputs.

For bounded profiling and disabling it per invocation, see
[training profiling](training-profiling.md).

## Policy temporal alignment

Both modes now use the same convention as sampling and RL: the row at state `t` contains `z[t]` and the preceding action, predicts the current interval action `a[t]`, and trains that action's shift against `sg(z[t+1] - z[t])`. The first input action is no-action. Classical losses average over all `L-1` transitions; continuous losses count every valid current interval.

Runs/checkpoints created before `policy_alignment: preceding-action-v1` cannot resume under this objective. Use a new training output directory. Rebuild old continuous indexes in a new index directory (`mcpref-continuous-index-v2`); existing raw datasets and classical indexes can be reused. Existing policy weights retain their old learned semantics until retrained.
