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
  config: configs/training/pretrain/classical.yaml
  index: configs/training/pretrain/shared-index.yaml
continuous:
  config: configs/training/pretrain/continuous.yaml
  index: configs/training/pretrain/continuous-index.yaml
```

Then run the same `bash scripts/pretrain.sh`. It validates the selected paths,
builds/resumes that mode's index and starts/resumes that mode's training.

- `base.yaml` contains shared model, objective and diagnostic settings and
  inherits `configs/training/optimizer.yaml`.
- `classical.yaml` uses clip batches and writes `data/pretrain_classical`.
- `continuous.yaml` uses `tbptt_states` observations per update (initially 4),
  keeps GDN memory through each fragment, and writes `data/pretrain_continuous`.
  `preference_micro_batch` bounds the number of preference pairs per call.
  Archive shard/cache limits bound the historical-latent working set.
- `retrain: true` adds a resumable frozen-perception phase. The final `model.pt`
  includes its trained heads; continuous also retains `joint-model.pt`.

Switching modes selects separate index, checkpoint and diagnostic directories.
The continuous index fingerprints the complete selected training and preference
configs, including K and preference history; choose new index/training output
directories when changing them for an experiment. The
continuous index's `training_config` must point to the selected continuous leaf.
`checkpoint_steps` must be divisible by `diagnostic_window`, and
`reference_steps` by `checkpoint_steps`.

Continuous model artifacts declare observation-local attention and fragment
memory. The existing classical RL/play consumers explicitly reject them; the
compatible perception API is `ContinuousPhiStream`. This pretraining addition
does not change RL/MCTS memory semantics.

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
`configs/training/pretrain/shared-labeled-index.yaml` as `classical.index` and
set the classical leaf's `index_dir` to `data/train_index_shared_labeled`.
For continuous training, set `pair_clips_dir: data/pair_clips_shared` and
`pair_labels_dir: data/pair_labels_shared` in its index config. Use new output
directories when changing a completed index or training run's inputs.

For bounded profiling and disabling it per invocation, see
[training profiling](training-profiling.md).
