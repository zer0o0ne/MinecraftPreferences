# Runtime gameplay
Status: approved (2026-11-17)

## Launch and modes
- `bash scripts/play.sh` runs `python -m mcpref.runtime.play.run` with `configs/runtime/play.yaml`. The only optional CLI argument is a different config path. All launch settings come from that config and its referenced configs; there are no CLI overrides for mode, duration, device, or output paths.
- The runtime config selects `mode: pretrained | rl`, a positive `trajectories` count, `out_dir`, `device`, `sampling`, and `rl_cycle` config paths. Sampling steps per trajectory, decision seed, environment and MCTS settings remain in the referenced runtime sampling/environment/MCTS configs. Training mathematics remain in the referenced training configs.
- `pretrained` loads the completed pretraining checkpoint and plays with the existing MCTS sampler. Model weights remain fixed; no optimizer, preference-pair generation, or RL memory update runs.
- `rl` executes the existing RL cycle once per trajectory, starting from the completed pretraining checkpoint and carrying trained models and optimizer state into subsequent cycles. MCTS and training mathematics are unchanged.
- Both modes create exactly one live environment for the entire invocation. Consecutive trajectories continue the same world and global action indices; trajectory boundaries do not reset the world. Close the environment on success or failure. Progress reports the requested trajectory count as well as existing per-stage progress.

## Run artifacts and configuration
- Paths inside YAML are relative to the repository root. The runtime output directory must be inside the repository and must not already exist, including an empty directory. Reject a reused output directory before creating an environment or changing its contents. Selecting a new directory explicitly starts a new world.
- This command does not resume a stopped process: current environment continuation records restore Python state but not the Minecraft simulator world. A single invocation may collect many trajectories. Full world snapshots and resumption after process termination are outside this change.
- All generated sampling, RL, pair/history, and diagnostic artifacts belong to the selected run directory:

```text
data/runtime/play/          # configured out_dir
  run.json                 # mode, effective config paths, completed count, status
  configs/                 # resolved effective configs with isolated output paths
  sampling/                # existing sampling shards and metadata; video input
  diagnostics/             # diagnostics for this run
  rl/                      # RL cycles (RL mode only)
  pairs/                   # generated preference pairs (RL mode only)
  history/                 # accumulated preference history (RL mode only)
```

- Effective configs redirect sampling, diagnostics and RL outputs into this directory and set model/training execution devices from the runtime config. The runtime `sampling` reference replaces the sampling reference inside the effective RL config. Preserve the source pretraining checkpoint path and check that pretraining is complete in both modes. Never edit source configs. Existing checkpoint, shard and fingerprint formats remain unchanged. Other referenced configs and pretrained artifacts are read-only dependencies; they must remain unchanged during the run.
- Save the effective launch configuration and run status atomically. Update the completed trajectory count only after its sampler/RL cycle succeeds. On failure retain completed artifacts for inspection and video export; report failure instead of claiming completion. The directory remains occupied.
- The default runtime config uses `mode: pretrained`, `trajectories: 1`, `out_dir: data/runtime/play`, `device: cuda`, and references the existing runtime sampling and RL cycle configs. Users edit these values before launching.

## Separate video command
- `bash scripts/export_gameplay.sh` reads `configs/runtime/video.yaml`; the only optional CLI argument is a different config path. Remove the former source/output positional arguments and range flags.
- The video config contains `sampling_dir`, `output`, `start_step`, `stop_step`, `last_steps`, `fps`, `ffmpeg`, `crf`, and `preset`. Specify either integer `start_step` and `stop_step` with `last_steps: null`, or positive integer `last_steps` with both range fields null. Paths may be absolute or repository-relative.
- Defaults point to `data/runtime/play/sampling` and `data/runtime/play/videos/latest.mp4`, selecting the last 300 completed actions. `fps: 10`, `ffmpeg: ffmpeg`, `crf: 18`, and `preset: veryfast` retain their current meanings. A shorter available recording fails clearly; no silent clipping or destination overwriting.
- All frame selection, snapshot, timing and atomic MP4 guarantees in `specs/gameplay-video.md` apply identically to either gameplay mode. Export may read completed shards while gameplay continues. Video output is separate from the sampling directory.

## Acceptance
- Both modes honor a runtime config, preserve one environment across two trajectories/cycles, close it on failure, and isolate all generated data under the configured directory.
- Pretrained mode uses MCTS and leaves model weights unchanged. RL dispatch preserves existing cycle behavior and optimizer/model continuity. Repeated sampling reports per-trajectory inventory milestones without retaining earlier trajectories' inventory lists.
- A reused output directory is rejected before simulator construction; no source config or existing run is changed. Effective configs and run metadata identify the selected mode and actual output paths.
- Video selection and encoding settings are read exclusively from YAML. Exercise both selection modes through the CLI, including a real FFmpeg encode/decode of saved-format records and invalid configurations.
- Run Ruff, relevant tests, shell entry-point checks and the full fast suite. Report separately any simulator/GPU acceptance that cannot run on local macOS without the Spark and pretrained artifacts.
