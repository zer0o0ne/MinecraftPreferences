# Gameplay video
Status: approved (2026-11-17)

## Purpose and interface
Export a selected interval of saved gameplay to a playable MP4, including while later trajectories are being collected. The source may come from pretrained+MCTS play or RL; no Minecraft instance or model inference is needed for export. `specs/runtime-play.md` defines the common gameplay launcher and run layout.

```bash
bash scripts/play.sh             # configs/runtime/play.yaml
bash scripts/export_gameplay.sh  # configs/runtime/video.yaml
```

`python -m mcpref.runtime.video.export [CONFIG]` reads the default `configs/runtime/video.yaml` or the optional alternative config file. The shell script is a thin wrapper. All source paths, selection and encoder settings are in YAML; there are no setting overrides on the CLI. Paths within YAML are absolute or relative to the repository root, independent of the caller's working directory. The module returns 0 and prints the output path on success, or reports an operational failure with a concise stderr message and nonzero exit code.

The complete default config is:

```yaml
sampling_dir: data/runtime/play/sampling
output: data/runtime/play/videos/latest.mp4
start_step: null
stop_step: null
last_steps: 300
fps: 10
ffmpeg: ffmpeg
crf: 18
preset: veryfast
```

To export actions 10000 through 10299 instead, set `start_step: 10000`, `stop_step: 10300`, `last_steps: null`, and choose an output filename. Exactly one complete selection is required. The three selection fields are integers or null, never booleans or fractional values. Step indices are global, zero-based environment action indices, not optimizer steps. Range selection includes actions A through B-1 and their endpoint observations; require `0 <= A < B <= available_steps`. Latest-N selection requires `1 <= N <= available_steps`, without silent clipping. Both modes may cross trajectory boundaries.

Either 300-action selection produces 301 frames and a 30.1-second file at `fps: 10`. Use a new output filename for a subsequent export. Playback speed is set only by `fps`. FFmpeg with libx264 must be on PATH or configured with its executable path; `setup_env.sh` provides it on the Spark.

## Source and frame selection
- Use the completed shards of `specs/env-sampling.md`. Fix the completed shard count with one initial directory listing; ignore `.part` shards and later publications. Obtain steps per trajectory from `run.json.config.steps`, or (when inherited) the positive integer ratio `counts.steps / counts.trajectories`; validate contiguous step filenames in the first completed shard against that length, without loading its records. This detects a missing final step as well as an internal gap. If inherited steps are not yet represented in counts, fail explicitly and request a retry after `run.json` publication; never guess the length from potentially incomplete files. Availability is the recorded trajectory length times the completed shard count, independent of possibly lagging counts. Current source config files are never read.
- This consumer permits one concurrent append-only sampler: completed shards are immutable and published atomically, with no removal or replacement. This narrowly extends the no-concurrent-reader domain of `specs/shards.md` for this consumer only.
- For [A,B), emit `before.frame` of action A, followed by `after.frame` of actions A through B-1: exactly B-A+1 frames, in order, with no duplicated observation at shard boundaries. Each record's stored global index must match its requested index. Read only selected step files, one record at a time, on CPU with `torch.load(..., map_location="cpu", weights_only=True)`; do not load models, trees into runtime objects, or trajectory continuation caches.
- Frames must be uint8 RGB with the same positive, even width and height. Preserve the recorded dimensions. Missing/corrupt selected records or incompatible frames fail the export; do not skip gaps.

## Time and output
- Playback rate is the independently configured `fps`, represented as the exact rational value of its decimal spelling. The B-A transitions occupy `(B-A)/fps` seconds; displaying the final endpoint for one frame makes the file duration `(B-A+1)/fps`. With the current agent interval delta=6 at 20 game ticks/s, `fps: 10` plays at three times nominal agent/game time. Changing video fps never changes training or sampling.
- Training pauses and inference delays are not replayed. Extra ticks used by explicit or automatic GUI closing are not timestamped in existing records, so the acceleration relative to actual game time is nominal. No time-based range selector is provided in this version.
- Encode RGB frames incrementally through FFmpeg into H.264 MP4, `yuv420p`, no audio, no interpolation or frame duplication, and `+faststart`. Frame rate, input pixel format and dimensions are explicit. `configs/runtime/video.yaml` contains `ffmpeg: ffmpeg`, `fps: 10`, `crf: 18`, `preset: veryfast`; numeric defaults live only in this config. `fps` is a finite positive number, `crf` an integer in [0,51], `preset` a supported x264 preset, and `ffmpeg` a nonempty executable name or path. No new Python dependency is needed; the Spark already installs pinned FFmpeg through `setup_env.sh`.
- Write a unique temporary file beside the destination and atomically publish only after successful encoder completion. Refuse an existing destination, including one created during encoding; failures leave no finished MP4 or running encoder. Output must be outside the sampling directory (resolved paths), so neither source files nor the shard layout can be affected. A progress bar on stderr has exactly B-A+1 frames as its total.
- Export is an explicit command. Automatic per-cycle videos, live streaming, overlays and intermediate simulator-tick recording are outside this change.

## Acceptance
- Select a range within one shard, across two shards, at exact shard boundaries, and containing one transition. Independently identifiable frames verify the initial/final observations, ordering and absence of boundary duplicates.
- Latest-N selection uses the fixed completed-shard snapshot, ignores staged data and tolerates stale run counts and inherited/changed source configs. Invalid or unavailable ranges, corrupt selected data, and existing destinations fail clearly without publishing a video.
- Memory use is independent of interval length; unrelated step files and continuation/model artifacts are not loaded. Encoder start/write/exit failures clean up the subprocess and temporary output.
- A real FFmpeg encode/decode round trip on a small saved-format trajectory verifies H.264, dimensions, exact rational frame rate, frame count, approximate decoded colours and duration. Exercise the CLI and wrapper; run relevant tests, Ruff and the full fast suite. If actual RL artifacts are unavailable, explicitly report that the end-to-end example uses a synthetic saved-format trajectory.

## References
- `specs/env-sampling.md` Datasets; `specs/env.md` (5), (10); `src/mcpref/runtime/sampling/shards.py` and `codecs.py`; `configs/datasets/clips.yaml`.
- [FFmpeg rawvideo input](https://ffmpeg.org/ffmpeg-formats.html#rawvideo): explicit framerate, pixel format and dimensions.
- [FFmpeg MP4 muxer](https://ffmpeg.org/ffmpeg-formats.html#mov_002c-mp4_002c-ismv): `+faststart`.
- [FFmpeg libx264 encoder](https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb): explicit CRF and preset.
