# Runtime Optimization Plan

Status: scope approved by the user on 2026-11-09.

## Objective

Reduce training and data-preparation runtime without changing the mathematical objective. Bitwise-identical results are preferred; where batched, fused, or FLA kernels make that impractical, only roundoff-level numerical differences are allowed. Artifact ordering, RNG semantics, stop-gradient boundaries, checkpoint resume behavior, and dataset durability must remain intact.

This plan covers the complete scope of the runtime audit except section 2.2. It is an implementation sequence, not permission to change all affected contracts in one increment.

## Explicit exclusion

Former section 2.2, optimization of per-step weight diagnostics, is excluded completely:

- no `torch._foreach_norm` rewrite of `gradient_stats` or `update_stats`;
- no reuse of diagnostic gradient norms for clipping;
- no foreach reconstruction of AdamW update statistics;
- no batching of `state["step"]` transfers for those statistics;
- no removal or reduced frequency of per-step diagnostics.

Routine changes explicitly listed below, including fused AdamW and deferred loss/observation transfers from former section 2.3, remain in scope. They must not alter the excluded diagnostic implementation.

## Global working rules

1. Deliver small, independently verified increments. Do not combine contract changes with unrelated rewrites.
2. Before each increment, record a focused baseline: runtime, the relevant operation counts, memory where material, and the exact artifacts or tensors used for comparison.
3. Tests use independent oracles. Do not validate an optimization against another copy of the optimized implementation.
4. Preserve the public schedule, seeds, dropout draws, and shard commit order unless an approved contract amendment says otherwise.
5. Compare one complete optimizer step for training changes: losses, gradients, parameters, optimizer state, RNG state, and resume from the resulting checkpoint.
6. Compare data artifacts byte-for-byte where the format is unchanged. VLM answers are compared at the request/payload level, not by forcing identical remote answers.
7. Run Ruff on touched files, relevant unit/integration tests, and the original runtime scenario. Run relevant GPU/slow tests when they cover the change. Run the full fast suite before declaring the overall plan complete.
8. Use real artifacts for focused acceptance runs; do not rerun an entire real pretraining or data-collection job merely to validate a local transition.
9. If an optimization does not improve the measured scenario or violates equivalence, stop and report it rather than silently keeping the change.
10. Where an approved spec fixes call counts, interfaces, commands, or file layouts, amend the concise contract before implementing the corresponding item.
11. For RL-2 and PT-10, first run a feasibility harness that reports actual numerical errors, memory and latency; the user will approve concrete tolerances from those measurements before the optimization is retained.

## Phase 0 — Reproducible baselines and acceptance harnesses

Establish focused fixtures and measurements for later phases:

- one RL optimizer step containing fresh, pending, and preference items, with horizon boundaries and perception restarts;
- one pretrain step with unlabeled and paired items;
- one real 12-frame clip for YUV/RGB, cursor, meta, and changed-mask checks;
- one small recorded trajectory with repeated histories and MCTS-bearing step records;
- one representative VPT video for decode-output and runtime comparisons;
- one preference label shard with reused frames;
- one persistent History generation for nearest, lookup, cleanup, and publication comparisons.

Record baseline wall time and, where applicable, perception/preference call counts, record reads, host synchronizations, pipe bytes, allocations, and peak memory. These counters are diagnostic measurements, not new training outputs.

## Phase 1 — RL training core

### RL-1: Cache immutable trajectory records

Current locations: `src/mcpref/sampling/shards.py`, `src/mcpref/rl/loss.py`.

- Compute `steps_per_shard` once per RL cycle instead of reloading `state.pt` for every operation.
- Add a bounded LRU for globally indexed step records, shared by `read_history` and `load_step`.
- Size the working cache around the active `O(L+H)` region; do not materialize the whole trajectory dataset.
- Keep raw records on CPU. Do not cache differentiable tensors across optimizer updates.
- As a later format increment, separate the training payload (`before`, `after`, `action`, `probabilities`) from the stored MCTS tree so training does not deserialize the tree. The current record format remains unchanged until that separate contract amendment.

Acceptance:

- identical `RawHistory`, actions, and probabilities;
- substantially fewer `torch.load` calls and unchanged training results;
- bounded memory and identical resume behavior.

### RL-2: Reuse causal perception prefixes within one optimizer step

Current locations: `src/mcpref/rl/history.py`, `src/mcpref/rl/loss.py`.

- Cache no-grad latents by stream-restart segment for the duration of one `step_gradients` call.
- Encode the longest required no-grad prefix once and derive shorter histories by slicing.
- Share this cache across micro-batches of the same optimizer step.
- Keep separate grad-bearing passes for `origin`, `successor`, and preference inputs that require gradients into `phi`.
- Preserve the existing stop-gradient treatment of futures and advantages.
- Invalidate every latent cache after the optimizer update; never carry latents into the next step or cycle.
- Verify prefix equivalence for full attention and gated delta/FLA paths. Attention dropout is zero; any other stochastic behavior would block this design until its RNG semantics are addressed.

Acceptance:

- prefix latents and required gradients match the independent full-history oracle within the approved numerical tolerance;
- no gradient flows through futures or advantages;
- the number of perception forwards falls materially on the real-artifact scenario;
- optimizer-step and checkpoint comparisons pass.

### RL-3: Batch preference advantage evaluation

Current location: `src/mcpref/ppo/reward.py`.

- Gather all consequence/origin pairs of a micro-batch.
- Group by history lengths to avoid padding and changing the model's positional inputs.
- Run batched `P` for each length group, normally one call for the fixed-length case.
- Restore the original item/horizon layout and preserve ascending `k` summation of discounted values.
- Keep the entire advantage computation under `no_grad`.

Acceptance:

- independent float64 reward oracle, including unequal history lengths;
- no preference or perception gradients through reward;
- materially fewer preference calls with equivalent advantages;
- required amendment to the literal `H`-calls-per-item contract is approved first.

### RL-4: Construct latent/action histories without repeated concatenation

Current locations: `src/mcpref/rl/history.py`, `src/mcpref/sampling/history.py`.

- Replace iterative `append` calls in `encode_raw` with direct construction of the final truncated `Y` and one-hot `C` tensors.
- Preserve row order, action indices, the no-action column, and the last-`L` truncation exactly.

Acceptance: bitwise-identical `Y` and `C` on hand-built and real histories, with fewer tensor copies and launches.

### MCTS-1: Build only model-required history tails during simulation

Current locations: `src/mcpref/mcts/tree.py`, `src/mcpref/mcts/search.py`, `src/mcpref/policy/network.py`, `src/mcpref/preferences/network.py`.

- Add a model-input path that constructs only the tail consumed by the policy and preference windows instead of materializing the complete base-plus-node history.
- Build a leaf's required history once per simulation and share it between leaf evaluation and expansion.
- Keep the existing full `history()` behavior for `advance`, re-rooting, persistence, and every other contract-visible consumer; never truncate `tree.base`.
- Prefer per-simulation construction over persistent node caches whose invalidation would depend on re-rooting and network updates.
- Preserve cached leaf values, working versus exact latents, incoming-action rows, model-specific `L_pi`/`L_P` truncation, decisions, and RNG.

Acceptance:

- identical policy and preference inputs, logits, values, decisions, and caller RNG on hand-built trees and independent simulation oracles;
- full `history()` and re-rooted `tree.base` remain exactly unchanged, including paths longer than the model windows;
- fewer history constructions and tensor concatenations in the measured MCTS scenario, with no regression against the MCTS latency budget.

## Phase 2 — Exact local CPU and I/O improvements

### PT-1: Prepare each pretrain clip once

Current locations: `src/mcpref/pretrain/micro_loss.py`, `src/mcpref/perception/phi.py`, `src/mcpref/perception/loss.py`.

- Compute `(frames, metas)` once per clip occurrence in `micro_loss`.
- Pass the same objects to `phi.frames_z` and to the perception loss for `changed()` and diagnostics.
- Preserve cursor drawing, meta origins, RGB rounding, and the exact changed mask.
- This aligns the code with the existing requirement in `specs/perception.md` that frames, metas, and `chi` are computed once; it is not a new mathematical decision.

Acceptance: identical latents, changed masks, losses, and gradients, with one YUV/RGB and meta preparation per clip occurrence.

### DATA-3: Remove avoidable clip-write copies and release pass-1 frames

Current locations: `src/mcpref/clips/store.py`, `src/mcpref/clips/episode.py`.

- Replace `b"".join(...)` for `frames.yuv` with `os.writev`, handling partial writes, or sequential writes if `writev` is unsuitable on the target filesystem.
- Release the pass-1 frame mapping before starting pass-2 decoding; retain only clips, classes, and other required state.

Acceptance: byte-identical clip files and lower copy/allocation volume, with no change to sampling or shard order.

### DATA-2a: Reuse the ffmpeg stdout buffer

Current location: `src/mcpref/vpt/ffmpeg.py`.

- Read raw frames into a reusable `bytearray` using a fill loop that handles short reads correctly.
- Allocate immutable `bytes` only for frames in `keep`.
- Preserve the current command, frame count validation, timeout, and error behavior.

Acceptance: identical selected frame bytes and failure behavior, with fewer discarded-frame allocations.

### DATA-4: Cache encoded VLM image payloads

Current locations: `src/mcpref/preferences/labels.py`, `src/mcpref/vlm/client.py`.

- Cache PNG/base64 data URLs for unique shown frame objects for the lifetime of one label shard.
- Keep the frame object in each cache entry so object IDs cannot be reused unsafely.
- Preserve prompt order, reversed-order requests, samples, seeds, retry bodies, and request composition.

Acceptance: byte-identical request image payloads and fewer PNG encodes; no change to response handling or persisted request order.

### PT-4, PT-6, PT-7: Remove small repeated tensor and target computations

- Cache RoPE cosine/sine values by positions, device, dtype, head width, and base in `src/mcpref/transformer/layers.py`; do not add them to model checkpoints.
- Compute policy `weights(clip)` once and share it between policy input construction and loss diagnostics.
- Reuse the factual squared shift for the zero-shift diagnostic instead of recomputing the same metric.
- Reuse `alpha` tensors by device and dtype instead of recreating them in every metric call.

Acceptance: equivalent outputs and gradients, unchanged checkpoint contents, and no additional RNG draws.

## Phase 3 — Remaining pretrain and shared-model runtime work

### PT-2: Defer loss and observation host transfers

Current locations: `src/mcpref/pretrain/micro_loss.py`, `src/mcpref/pretrain/loss.py`, `src/mcpref/pretrain/observation.py`, `src/mcpref/rl/loss.py`.

- Keep logging values as detached GPU scalars or compact sufficient statistics until the end of the relevant step.
- Replace scattered `.item()`/`float(...)` calls with batched `stack(...).cpu().tolist()` transfers.
- Do not insert host synchronization between forward and backward.
- Preserve Python-float accumulation order where required for logged values.

This item explicitly excludes `gradient_stats`, `update_stats`, clipping diagnostics, and all other former-section-2.2 work.

Acceptance: identical logged schemas and values within approved floating-point rules, fewer synchronizations, and unchanged gradients.

### PT-3: Evaluate fused AdamW

Current location: `src/mcpref/pretrain/update.py`.

- Benchmark the CUDA fused implementation with the existing groups, betas, epsilon, decoupled weight decay, and learning-rate schedule.
- Vet the pinned Torch implementation and pass relevant arguments explicitly.
- Retain the current implementation if numerical, resume, or runtime acceptance fails.

Acceptance: equivalent parameters and Adam moments, compatible checkpoints, no update for `grad is None`, and a measured improvement.

### PT-5: Batch the two preference-layer directions

Current location: `src/mcpref/preferences/layers.py`.

- For `LayerS`, combine self-attention inputs where sequence lengths match.
- For `LayerX`, combine queries `[A,B]` with keys `[B,A]`, then split the result.
- Preserve shared weights, positions, non-causal attention, and output order.

Acceptance: independent per-pair oracle and roundoff-only differences on GPU.

### PT-8: Overlap checkpoint host copies

Current locations: `src/mcpref/training/devices.py`, pretrain and RL checkpoint writers.

- Use bounded pinned-CPU staging and non-blocking copies for module and optimizer-state tensors. The user approved a 256 MiB chunk budget; a single larger tensor may occupy one chunk.
- Synchronize before serialization and keep atomic publication unchanged.

Acceptance: identical restored weights, optimizer state, RNG, and data position; bounded staging memory.

### PT-9: Later, measurement-gated runtime techniques

- Add bounded clip prefetch to overlap disk/JSON/YUV preparation with GPU work without caching the whole dataset or changing micro-batch order.
- Evaluate `torch.compile` only on clean tensor paths after batching and caching are complete; do not begin with the full HF backbone or selected gradient-flow diagnostics.
- Reconsider `micro_batch` only after the RL item-level code is genuinely batched and memory measurements support it. Preserve dropout/RNG semantics.

These items require individual measurements and, where the current contract excludes parallel loading or fixes call structure, prior contract amendments.

### PT-10: Batch independent images in vision attention

Current locations: `src/mcpref/perception/phi.py` and the pinned Transformers Qwen3.5 vision-attention implementation.

- For the SDPA path, combine independent per-image Q/K/V chunks along the batch dimension instead of invoking attention separately for every image.
- Run one attention call per vision block for the equal-sized 12-image case, then restore outputs to the original packed sequence order.
- Preserve each image's independent context, `cu_seqlens` boundaries, rotary embeddings, scaling, non-causal attention, and dropout behavior; never mix tokens across images.
- Keep unsupported attention implementations and unequal-length cases on a vetted equivalent path rather than silently changing their behavior.
- Vet the exact pinned Transformers implementation and avoid an unscoped global monkey patch.
- Treat the reduction from 288 to 24 attention calls as a call-count improvement, not an assumed 12x end-to-end speedup; measure vision forward, backward, and complete `phi` latency and memory.

Acceptance:

- hidden states and gradients match the independent per-image attention oracle within the approved numerical tolerance;
- packed outputs remain in the original image and token order;
- no change to text-side latents beyond allowed roundoff, checkpoint contents, or RNG;
- a measured runtime improvement without an unacceptable memory regression.

## Phase 4 — R3 pair production and R5 History

### R3-1: Cache fixed-model evaluation latents and batch comparisons

Current locations: `src/mcpref/rl/history.py`, `src/mcpref/rl_pairs/scores.py`, `src/mcpref/rl_pairs/sample.py`.

- Cache evaluation latents for the whole `sample_pairs` invocation, where `phi` is fixed and evaluation is under `no_grad`.
- Reuse them across seed scoring, seed insertion, and discovered comparisons.
- Batch independent eval-only preference calls after deduplication.
- Use a new cache for final-key publication because R5 uses the final, updated `phi`; recompute only the selected seeds as required.

Acceptance: identical selected seeds, alternatives, comparisons, and final keys; no stale start-of-cycle latents used after training.

### R3-2: Remove repeated History scans and per-vertex synchronizations

Current locations: `src/mcpref/rl_pairs/alternatives.py`, `src/mcpref/rl_pairs/history.py`.

- Compute nearest distances one shard at a time with vectorized operations and maintain only the required top results with the existing `(distance, StateRef)` ordering.
- Add a ref-to-shard/offset index or bounded lookup cache so `_raw` does not rescan every base shard per miss.
- Replace full cleanup scans with a deterministic priority structure. Because eviction removes comparison references and can change remaining vertices' scores, update affected entries through reverse-comparison adjacency after every removal; a static heap is not sufficient.
- Preserve shard-bounded reading and deterministic cleanup checkpoints.

Acceptance: identical nearest sets, beam results, eviction sequences, cleanup progress, and published generations on a persistent real History.

## Phase 5 — Data collection and preparation pipeline

### DATA-1: Classify directly from masked YUV samples

Current locations: `src/mcpref/clips/episode.py`, `src/mcpref/recognition/classifier.py`, `src/mcpref/vpt/yuv.py`.

- Gather Y and corresponding half-resolution U/V samples only at `model.flat` coordinates.
- Reproduce the existing float64 operation order, two-step green subtraction, round-half-up, clipping, and integer count reductions exactly.
- Vectorize count computation over a batch only after single-frame equivalence is established.
- Extend the pluggable classifier contract deliberately; do not bypass other classifier implementations silently.
- Optionally add bounded cross-candidate batching while preserving deterministic per-frame classes, per-episode seeds, and draw commit order.

Existing evidence: on 12 real frames, all 44,799 selected RGB values and all classes matched, with approximately 8–17x speedup in repeated microbenchmarks. End-to-end acceptance still requires the real classifier pipeline.

### DATA-2b: Emit only selected frames from ffmpeg

Current locations: `src/mcpref/vpt/ffmpeg.py`, `src/mcpref/clips/episode.py`.

- Add an ffmpeg `select` path for pass 2 so only requested raw frames cross the pipe.
- Preserve passthrough timestamps, pixel format, selected byte identity, timeout, and failure propagation.
- Evaluate pass-1 selection separately: it must retain independent validation of exactly `n` decoded input frames.
- Do not fuse both decodes by spooling all raw frames unless a separate measurement proves that the memory/disk trade-off is beneficial.

Acceptance: identical selected bytes and validated frame counts, with a measured reduction in pipe traffic and end-to-end decode time. Amend the exact-command contract before implementation.

### DATA-5a: Stream ranged video downloads to their final offsets

Current locations: `src/mcpref/vpt/downloads.py`, `src/mcpref/clips/episode.py`.

- Add a video-specific fetch-to-path implementation using streaming writes or `pwrite` at each range offset.
- Retain the existing bounded per-file range-fetch executor.
- Include response-body iteration in retried attempts, release completed part buffers promptly, and preserve retries and lowest-index failure behavior.

Acceptance: byte-identical MP4 files, bounded part memory, equivalent HTTP failure behavior, and no measured runtime regression.

### DATA-5b: Reuse the range-fetch executor

Current location: `src/mcpref/vpt/downloads.py`.

- Reuse a bounded range-fetch executor rather than constructing one per file; this is separate from DATA-5a.

Acceptance: byte-identical artifacts, preserved failure behavior, and a measured improvement.

### DATA-6: Refill preference-pool workers as individual jobs finish

Current locations: `src/mcpref/pool/shards.py`, `src/mcpref/clips/plan.py`.

- Replace whole-draw waiting with completion-driven refilling from remaining episodes and subsequent draws.
- Keep a bounded global outstanding-job count.
- Preserve draw RNG capture, per-episode seeds, result association, error ordering, and strict draw-order shard commits.
- Consider ordered examination prefetch only if profiling shows planner-side network starvation.

Acceptance: byte-identical shards for multiple worker counts and interruption/resume scenarios, with less worker idle time.

### DATA-7: Replace repeated long-episode scans

Current locations: `src/mcpref/vpt/actions.py`, `src/mcpref/clips/validity.py`, `src/mcpref/clips/sample.py`, `src/mcpref/gui/actions.py`.

- Replace event `counts` with an `O(T+F)` running-counter pass.
- Cache interval action lists and validity results for reuse by accepted clips.
- Use integer prefix/sliding counts for valid interval/state windows.
- Avoid rescanning all segments for every interval where an active-segment sweep gives the same ordered result.
- Compute GUI spans and needed frames once for their consumers.
- Preserve the original yaw summation order near thresholds; do not introduce prefix-sum subtraction without a numerical boundary oracle.

Acceptance: identical action lists, turn counts, valid starts, accepted clips, and GUI exclusions.

### DATA-8: Remove redundant filesystem synchronization

Current location: `src/mcpref/shards/store.py`.

- Remove `fsync` from the read-only `complete_shards` path after amending its stated contract.
- Track directories already synchronized by the current process and avoid repeatedly syncing unchanged ancestor chains.
- Preserve syncing of newly created directory entries, staged file trees, atomic renames, and publication parents.
- Hoist repeated `mkdir(exist_ok=True)` calls out of per-file loops where applicable.

Acceptance: unchanged artifacts and crash-recovery tests, with fewer filesystem sync calls and no weakened publication durability.

## Phase 6 — Structural follow-ups, separately approved

These remain in scope but must not block the earlier increments:

- **Packed clip storage:** store each shard's frame payloads contiguously with an offset index and consolidate metadata. Migrate readers and writers together and define resume/version behavior before implementation.
- **Training-only trajectory payload:** separate training fields from MCTS trees as described in RL-1.
- **Additional batching or compilation:** adopt only after the preceding structural fixes and fresh profiling identify the remaining bottleneck.

## Implemented increments and measurements

- **RL-1:** the focused 13-operation fresh/pending/preference trace uses a bounded capacity-5 cache. `torch.load` calls fell from 45 (13 metadata, 32 records) to 10 (one metadata, nine records), and cold-cache median runtime from 32.45 to 7.36 ms. Histories, actions and probabilities are identical; the exact fake-CPU two-cycle and recovery scenarios pass. The fixture is synthetic because no real `data/sampling` trajectory is available.
- **RL-3:** 16 origins with horizon 64 now require one length-grouped `P` call instead of 1,024 batch-1 calls. Median CPU float64 runtime fell from 17.13 to 2.57 ms with an identical result hash, including unequal-length and no-gradient oracles.
- **RL-4:** a five-state hand history now uses one `torch.cat` instead of nine; median encoding time fell from 70.7 to 35.5 µs. `Y/C` hashes are bitwise identical, including restart, truncation and retained no-action rows.
- **MCTS-1:** the focused leaf evaluation/expansion uses one history construction and zero concatenations instead of two and four. Focused CPU median latency fell from 114.5 to 93.9 µs; the existing real-sized GPU latency scenario fell from 134.9 to 43.4 ms. Full histories, re-rooted bases, decisions and RNG match the independent oracle.
- **PT-1:** two three-state clips require two rather than four clip preparations. Median small-fixture forward/backward time fell from 3.58 to 1.97 ms, with bitwise latents, changed masks, losses, diagnostics and gradients. This measurement does not claim full-backbone latency.
- **PT-2:** focused CUDA steps reduced scoped host transfers from 120 to five in pretrain and from nine to one in RL. Interleaved median step times improved from 249.25 to 129.22 ms and from 100.17 to 86.29 ms respectively. Losses, observations, gradients, parameters, optimizer state, RNG and restore comparisons pass.
- **PT-3:** on the real 25 GB pretrain checkpoint, fused CUDA AdamW reduced median update time from 2.006 to 0.946 s and peak allocated memory from 40.136 to 33.148 GiB. Parameters and moments match within `rtol=2e-6, atol=2e-7`; legacy/fused resume and RNG checks pass. CPU remains non-fused.
- **PT-4:** repeated canonical RoPE calls perform no new cosine, sine or frequency-`arange` operations and add no host synchronization. Direct fixed-position runtime fell by 45.4%; interleaved GPU policy/preference medians fell by 16.7% and 34.1%. Outputs, gradients, checkpoints and RNG are unchanged.
- **PT-5:** equal-length preference directions use one attention call per layer instead of two. CPU forward/backward medians improved by 35.5% (`LayerS`) and 32.0% (`LayerX`), CUDA medians by 17.2% and 14.7%; independent output/gradient oracles pass. Unequal lengths retain the separate path.
- **PT-6/PT-7:** the two-clip policy diagnostic call now makes two rather than four `weights` calls. Alpha allocations fell from two to one in the policy and two-clip `PhiLoss` fixtures, and from four to one in the two-micro-batch RL fixture. Interleaved eight-clip `D=40` CPU forward/backward medians fell from 2.212 to 2.020 ms (policy) and from 1.265 to 1.256 ms (`PhiLoss`); the RL `D=40` step fell from 1.114 to 1.111 ms, while the tiny `D=2` RL fixture showed no speedup. Losses, gradients, RNG and fixed-reference observations are bitwise identical; checkpoint state is unchanged. CUDA measurement was blocked by GPU memory pressure.
- **PT-8 (rejected):** on the real 26,677,736,137-byte checkpoint (708 module tensors and 708 optimizer states), the existing writer took 299.97 s with zero pinned bytes. The bounded 256 MiB candidate had not reached serialization after 874 s, exhausted the 15 GiB swap area, and was cancelled and reverted because the additional full ordinary-CPU copy eliminated the benefit. Its real pinned high-water and restored artifact were therefore not obtained; the instrumented CUDA chunk test peaked at 72 bytes for one oversized tensor under a 48-byte test budget, and the candidate's fused-resume/failure tests passed before removal. The real baseline artifact restored as ordinary non-pinned CPU tensors in 32.46 s at step 2100; CPU/CUDA RNG restoration and an exact fused continuation of a real 1,572,864-element Adam state passed.
- **DATA-1:** on the real 12-frame clip and classifier model, 15 interleaved pipeline pairs reduced median classification time from 319.96 to 18.40 ms (17.39x) and peak Python allocations from 31,802,224 to 4,800,577 bytes (84.9%). All 1,612,764 selected RGB bytes, classes, `gamma_id`, and the staged `frames.yuv`, `lines.jsonl` and `clip.json` SHA-256 values are unchanged. Cross-candidate batching remains deferred.
- **DATA-2a:** on the available synthetic 120-frame 640×360 H.264 video, reusable-buffer decode reduced median runtime from 156.5 to 146.5 ms and peak Python allocations from 4,852,402 to 4,510,502 bytes. Every selected-frame SHA-256 is unchanged.
- **DATA-2b:** pass-2 selection on the same video reduced pipe traffic from 41,472,000 to 4,147,200 bytes and mean decode time from 154.26 to 146.99 ms. Selected bytes are identical. The short synthetic-video timing is not extrapolated to five-minute VPT episodes.
- **DATA-3:** 31 alternating real-clip write pairs reduced median `frames.yuv` staging time from 3.08 to 2.24 ms and peak Python allocations from 4,189,322 to 50,968 bytes. The real 12-frame source and written SHA-256 are identical; pass-1 frames are released before pass 2.
- **DATA-4:** real shard `data/pair_clips/shards/000000` now requires 72 rather than 1,584 PNG encodes for 132 requests. Encoding time fell from 76.66 to 3.12 s; the complete ordered image-payload SHA-256 is unchanged.
- **DATA-5a:** 31 alternating local-server download pairs on the available synthetic 120-frame, 1,103,416-byte H.264 video changed median fetch-to-path runtime from 10.788 to 10.725 ms and mean runtime from 10.994 to 10.989 ms (no material regression), while peak Python allocations fell from 2,301,195 to 1,195,255 bytes (48.1%). Source slices and the complete MP4 are byte-identical with SHA-256 `5ff4d15ed29391e7cd74cc327aaa11070bae226d08addb9dde64735d75f029db`; mid-body retries, exhaustion, cleanup, wait-all and lowest-index failures pass. No real VPT MP4 is available, so this short local timing is not extrapolated to five-minute episodes. The shared executor remains deferred to DATA-5b.
- **DATA-5b:** one caller-owned eight-thread range executor is shared by every video fetch of a `Pipeline` and closed by both runners on success and failure. Across three 31-pair alternating local-server runs on the same synthetic 1,103,416-byte MP4, median fetch-to-path runtime fell from 13.620 to 12.497 ms (8.2%; per-run baselines 13.347–13.868 ms, shared 11.962–12.497 ms), and the mean of run means fell from 14.199 to 12.523 ms (11.8%). The median eight-task per-file executor lifecycle was 0.407 ms versus 0.103 ms for a round on the existing executor. All 186 timed MP4s matched the source SHA-256 `5ff4d15ed29391e7cd74cc327aaa11070bae226d08addb9dde64735d75f029db`; streaming retries, cleanup, wait-all/lowest-index failures, the global active-range bound and DATA-6 scheduler tests pass. These short synthetic local timings are not extrapolated to real five-minute VPT episodes.
- **DATA-6 (rejected):** unbounded completion-driven refill reached 0.3018 s in the direct three-worker slow-first scenario, but allowed a stalled oldest draw to accumulate unlimited completed later results and staging, so that result is invalid. Counting all pending jobs bounded retention, but its five-run median was 0.4039 s versus the 0.4017 s whole-draw baseline, with no runtime improvement. Under global rule 9 the optimization is rejected and the original pre-DATA-6 scheduler, tests and contracts remain in place.
- **DATA-7:** on a seeded 6,000-line episode (1,376 segments, 5,330 kept lines), median valid-start construction fell from 1,264.86 to 14.22 ms and cached sampling from 5.13 to 0.065 ms. On the real 6,000-line `shabby-viridian-beaver-…-170618` JSONL (377 segments under the benchmark's deterministic all-inventory needed-frame classes), valid starts fell from 426.68 to 15.16 ms, sampling from 20.21 to 5.89 ms, and GUI preparation from 4.65 to 2.43 ms. Valid starts, accepted clips, needed frames, GUI segments/cut/foreign lines and incremental GUI exclusions match the saved baseline exactly; independent naive action/validity/mode oracles and yaw boundary cases pass. The full real JSONL was fetched from the public URL in the stored index because local shards retain only 67-line clip spans; its complete original classifier predictions are unavailable, so GUI timing/exclusions use the stated deterministic classes rather than claiming original classifier-artifact coverage.
- **DATA-8 (safe portion):** repeated `make_dirs` on an unchanged directory inode now skips its already-synced ancestor chain; new children, failed syncs, publication renames and `complete_shards` retain their required synchronization. The focused second call dropped from ten `fsync` operations and 145.5 µs to zero and 15.8 µs. Repeated `options/` and `clips/` directory creation is hoisted. Removing the read-only `complete_shards` sync is rejected because it would weaken crash recovery.
- **R3-1a:** the focused 24-state, 24-seed, six-temporal-alternative scenario reduced encoder calls from 169 to 23 and encoded references from 314 to 24, with all 145 batch-1 preference calls unchanged. Median runtime over nine runs fell from 0.489 to 0.470 s; all 56 score/work/correction files are byte-identical. Direct per-ref/per-pair oracles, resume and fresh-final-key separation pass. The fixture uses real tensors and disk shards but is synthetic: no real R3 generation or sampling trajectory is available for an end-to-end latency measurement.
- **R3-1b (rejected):** the length-grouped candidate preserved CPU artifacts and RNG while reducing 145 preference calls to 47 and nine-run median runtime from 0.423 to 0.379 s. On the equivalent real-Preference CUDA fixture it reduced calls to 101, but changed 27 persisted float fields in 18 work shards (maximum absolute differences $5.96\times10^{-8}$ for $p$ and $1.19\times10^{-7}$ for incomparability), so corrected JSON was not byte-identical. Sequential cold-baseline/warmed-candidate runs took 2.589/1.367 s; a later warmed restored run took 1.065 s, so these uncontrolled timings support no speedup claim. Seeds, alternatives, pair orientation and RNG remained unchanged. The candidate was reverted; no real R3 generation or sampling trajectory is available for a production end-to-end measurement.
- **R3-2:** on 2,048 synthetic vertices in 64 persistent JSON shards, nearest evaluation reduced norm calls from 2,048 to 64 and median runtime from 20.157 to 7.986 ms; 96 hits plus 16 misses reduced raw shard reads from 3,786 to 160 and runtime from 319.4 to 12.5 ms. On a 1,024-vertex cleanup with 512 removals, the versioned heap/reverse adjacency reduced full staged scans from 513 to two and median runtime from 4.9337 to 0.0192 s. Naive nearest/direct-JSON/full-rescan oracles, dynamic eviction order, resume and published bytes match exactly. No real persistent History generation is available.
- **PT-10 (rejected):** equal-length vision batching reduced attention calls from 288 to 24 and was bitwise-identical in forward, including packed order and text-side latents. Vision-only total improved by only 1.01%, while complete `phi` forward/backward regressed from 3.398 to 3.537 s (4.08%) in all five paired repetitions. Complete incremental peak memory did not regress. Backward differences were comparable to same-method CUDA nondeterminism, but no tolerance is needed because the candidate is rejected on latency. Raw results are in `data/analysis/pt10-feasibility-measurements.json` (SHA-256 `c31343a0e66439b2dcdf0b1bb23828e6b087b6d1a0453db3e63b418dd9997a18`); production code was not changed.
- **RL-2 (rejected by feasibility):** on the real Qwen3.5-2B and a real 12-frame clip, independent prefixes and slices of one longest pass were not equivalent. Maximum latent absolute error reached 0.04221 in bf16 and selected backward relative-L2 error reached 23.25%; these are behavioural differences, not roundoff. Full attention and gated-delta/FLA paths were measured, with raw per-layer results in `data/analysis/rl2-feasibility-measurements.json` (SHA-256 `b062fce5d42387dbf4f1f39c48a7a0042eba86e7ae94b1265bf9a36a1c0c63b9`). The potential 2.28x forward speedup is rejected; production code was not changed.
- **PT-9b (two loss cores retained):** ordinary CUDA pretraining compiles only the tensor cores of `PreferenceLoss` and `PhiLoss`; the HF backbone, policy network/loss, preference network, dropout and selected gradient-flow diagnostics remain eager. On the real-size float32 GB10 fixture, 31 interleaved pairs gave `PreferenceLoss` forward/backward medians of 0.0927/0.0912 ms eager and 0.0689/0.0727 ms compiled (1.30x total), with forward/backward peaks of 2,560/2,560 versus 1,024/1,536 bytes. `PhiLoss` took 0.3212/0.7245 ms eager and 0.2009/0.5486 ms compiled (1.40x total), with peaks of 12,288/7,168 versus 5,120/6,656 bytes. First forward/backward compilation took 284.65/58.35 ms and 595.92/443.13 ms respectively; seven warm-up iterations per path took 5.23 ms and 16.46 ms in paired eager/compiled runs. Independent eager oracles passed at `rtol=2e-5, atol=2e-6`; checkpoint keys and CPU/CUDA RNG are unchanged, and a two-step optimizer/eager-checkpoint/compiled-resume comparison passes. The preference network (1.69x) and policy loss (1.15x) were rejected despite speedups: with one-token attention, changes to tiny numerical Q/K gradients are amplified by AdamW into maximum parameter differences of $1.27\times10^{-5}$ and $4.00\times10^{-4}$ respectively, outside that tolerance. Policy shift was 7.5% slower and was also rejected. The pinned Torch multiworker compiler pool exits on this host, so the retained callables use Inductor's scoped `compile_threads=1` option. Raw results are in `data/analysis/pt9b-compile-measurements.json` and `data/analysis/pt9b-optimizer-measurements.json`.
- **PT-9a:** one worker with one next-micro-batch future reduced median main-thread read-wait from 57.773 to 21.551 ms cold and from 5.788 to 2.671 ms warm on three real indexed micro-batches containing four 12-frame 640×360 clips. Over five alternating pairs per condition, median complete forward/backward gradient-step time fell from 10.10486 to 10.09770 s cold (0.07%) and from 10.13556 to 10.10424 s warm (0.31%), with four of five pairs improved in each. Cold means regressed from 10.12564 to 10.13515 s because of one prefetch outlier; warm means fell from 10.42953 to 10.10605 s. Incremental CUDA peak stayed 64,592,059,904 bytes and median sampled process-RSS increments did not regress. Read order, all logged losses and CPU/CUDA RNG states matched exactly. Raw results are in `data/analysis/pt9-prefetch-measurements.json` (SHA-256 `93055c20ef9cecaa588b3fa9e7b3c3e05e49dbc8cb5e373099edb38a98072626`).

## Current completion status

Every actionable increment through PT-9a/PT-9b and DATA-5b is implemented or explicitly rejected by measurements. PT-9c remains blocked because RL item-level computation is not genuinely batched and no supporting memory evidence exists. Phase 6 packed clip storage and training-only trajectory payloads remain deferred pending their separately approved format/version/resume contracts; additional structural batching remains profiling-gated. DATA-1's optional cross-candidate batching is not adopted. The overall plan is therefore not complete.

## Final completion criteria

The plan is complete only when:

1. Every item above is either implemented and accepted or explicitly reported as rejected by measurements/equivalence tests; section 2.2 remains untouched.
2. Training and data-prep benchmarks show the measured before/after results on the same scenarios.
3. Mathematical oracles, artifact comparisons, RNG checks, and checkpoint/resume checks pass.
4. Relevant Ruff, unit/integration, GPU/slow checks pass, followed by the full fast suite before final completion.
5. Affected specs and cross-references agree with the delivered behavior, with no silent changes to call counts, interfaces, commands, or artifact layouts.
