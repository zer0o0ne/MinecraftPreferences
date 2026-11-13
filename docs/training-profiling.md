# Training profiling

Capture a bounded window of complete optimizer iterations in pretrain and RL,
including frozen-phi retraining. `configs/training/profile.yaml` is loaded by
default and currently has `enabled: true`. Set `enabled: false` to disable it.
This separate file does not change the training config, its fingerprint, the
objective, checkpoint contents, or the resume position.

From the repository root:

```bash
bash scripts/pretrain.sh
bash scripts/rl-cycle.sh
```

Direct `python -m` training invocations use the same default profile config.
`MCPREF_PROFILE=path/to/profile.yaml` selects another file; `MCPREF_PROFILE=`
disables profiling for one invocation. If the default file is absent, profiling
is disabled; an explicitly selected missing file is an error.
The RL window starts at optimizer training, after rollout and pair generation.
Model loading, initial reference construction, and retrain advantage preparation
are outside the measured iteration.

`configs/training/profile.yaml` contains:

```yaml
enabled: true
skip_steps: 2
steps: 1
top_ops: 30
```

`skip_steps` counts optimizer steps actually executed in this invocation, even
when resuming at a nonzero checkpoint. `steps` bounds the number captured; each
gets its own profiler instance and trace. The window restarts for each training
stage. Training continues normally after the window. Increase `skip_steps` if
FP8/Inductor compilation is still occurring; a fixed warmup count cannot guarantee
that all shapes have compiled. `top_ops` limits the operator tables, not the trace.
The three numeric fields are required; `skip_steps` is a nonnegative integer, the
others are positive integers. `enabled` is a YAML boolean (true when omitted).

The output path is printed to stderr. Files live under:

```text
<training-output>/profiles/<stage>/<unique-invocation>/step-000003/
  summary.txt
  summary.json
  trace.json.gz
```

Stages are `pretrain`, `pretrain-retrain`, `rl`, and `rl-retrain`. Pretrain uses its
configured `out_dir`; RL uses the cycle directory. Repeated runs never overwrite
earlier profiles. Open the compressed Chrome trace in Perfetto to inspect CPU
ranges, CUDA kernels, copies, waits, and compilation events. `summary.txt` contains
the phase table and operators sorted separately by self CPU and self device time.
`summary.json` also records device/Torch version, configured batch/microbatch,
actual item count, gradient-diagnostic mode, peak CUDA allocated/reserved bytes,
and each Qwen call's row count, frame count, token lengths, padding fraction, and
whether autograd was enabled. A preference item can supply two Qwen rows.

## Interpreting the measurements

- `wall_s` measures the whole instrumented iteration, with device synchronization
  only at its boundaries. Checkpoints, reference events, and diagnostics performed
  in that iteration are included. Profiler startup, export, and report writing are
  excluded; collection overhead is included. The progress bar can therefore show
  a longer interval while a profile is exported.
- `host_s` is an inclusive CPU span: Python, operator dispatch, I/O and any GPU
  waits inside that region. It is **not** pure CPU computation time. In particular,
  `.cpu()` can wait for earlier asynchronous GPU work.
- `device_s` is the profiler's attributed device work, not elapsed wall time or
  utilization. Nested ranges and CPU/GPU work overlap: do not add these columns or
  sum parent and child rows. Use the timeline to investigate idle gaps and overlap.
- CPU-only runs report device time and CUDA memory as unavailable. CUDA profiling
  requires PyTorch/CUPTI support; an unsupported CUDA profiler raises explicitly,
  and a trace with no CUDA device events is flagged instead of reporting zero GPU
  work as a valid measurement. MPS records host activity only.
- `backward` includes gradient-checkpoint recomputation. Top-level `qwen.vision`
  and `qwen.language` hooks label the original forwards; their sum is not the full
  training cost of Qwen.

Useful regions:

| Region | Work measured |
| --- | --- |
| `data.index_read` | Index shard reads and item lookup preparation |
| `data.input_wait` | First clip read and waits for the next prefetched microbatch |
| `data.history_read` | RL trajectory/history reads, including cache hits |
| `phi.prepare`, `qwen.preprocess` | Frame/meta construction, image processing, tokenization, positions and padding |
| `qwen.transfer` | Input construction/copies to the model device |
| `qwen.forward`, `qwen.vision`, `qwen.language`, `phi.head` | Perception forward and its main components |
| `rl.gradient_histories`, `rl.reward_histories` | Differentiable vs no-grad reward-history encoding |
| `heads_losses`, `preference_loss`, `rl.heads_losses` | Heads and objectives after perception |
| `backward` | Backpropagation, including recomputation |
| `optimizer.clip_grad`, `optimizer.adamw` | Gradient clipping and parameter update |
| `diagnostics.*` | Observations, host reductions, gradient/update diagnostics and publication |
| `checkpoint`, `diagnostics.reference` | Periodic checkpoint and reference evaluation |
| `device_wait` | Remaining device work at the end of the step |

`data.input_wait` is the visible stall, not all background disk work: prefetch
reads can overlap computation and its worker is not instrumented by these ranges.
Detailed shape/stack/memory-event recording is disabled to avoid retaining
training tensors and inflating trace size; allocator peaks are read separately.
Temporary Qwen hooks are removed after each captured step, including exceptions.

For the observed approximately 100-second iteration at `batch=16`, compare
`micro_batch=1` and `micro_batch=8` with the same data, model/optimizer starting
state, loss composition, precision, diagnostics, and checkpoint cadence. Keep
the total batch fixed. Look first at Qwen row counts and call counts, then
`device_s` for forward/backward, CPU preprocessing/input waits, and periodic work.
Compare steady-state throughput without profiling separately; the profiled step
is intended to locate the bottleneck, not to establish an unbiased speedup.

API and timing semantics follow the installed PyTorch version's
[profiler documentation](https://docs.pytorch.org/docs/2.14/profiler.html) and
[CUDA asynchronous execution notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution).
