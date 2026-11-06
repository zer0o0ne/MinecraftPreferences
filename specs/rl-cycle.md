# RL cycle
Status: draft
Stage: block 13 of `specs/00-overview.md` — R1–R5 orchestration and the one joint training loop of $\phi,P,\pi$

## Sources
- [U] the user, 2026-11-03: R2's METRA, R3's fixed-label preference fine-tuning, R4's PPO and factual-shift regression form one joint training loop, analogous to pretraining. All three networks train together; current $\phi,P$ recompute PPO advantages on every optimizer step, and gradients of the policy and preference losses flow into $\phi$.
- [U] the user, 2026-11-03: batch formation follows pretraining's principle: trajectory items and fixed preference items mixed in their natural proportion, each item once per epoch, shard order and items within a shard reshuffled per epoch, a mini-batch within one shard, gradient accumulation over micro-batches and one shared update. Unlike pretraining pair clips, an RL preference item contributes only preference loss. A trajectory item contributes the losses for which it has the required data.
- [U] the user, 2026-11-03/05: cycle order is R1 → fixed preference-pair production → joint training → R5. Fresh trajectory states are available to pair production before permanent memory insertion. Pending PPO tails continue into the next cycle; shift and METRA need only one factual successor. Only selected seeds enter memory after training, with insertion keys encoded by final $\phi$; the next R1 uses the final networks and rebuilds perception streams.
- [U] the user, 2026-11-05: RL is divided into the cycle, pair sampling with persistent memory (block 32), and the loss block including PPO, factual shift and METRA (block 12); former candidate blocks 10 and 11 are not separate work items. One cycle creates one trajectory shard and uses per-cycle `StateRef.episode` values.
- [U] the user, 2026-11-05: for cycle $c$, $U^-$ is the action count before R1 and $U=U^-+N$ after it. Fresh actions are $[U^-,U)$, R3 states are $[U^-,U]$, eligibility is $t+H\le U$, and the tail is $[\max(0,U-H+1),U)$. Store the tail in an explicit atomic shard. Use self-contained mixed-index items, an initial global `PCG64([seed,2])` permutation, `training.schedule` with `[seed,3]`, and omit a loss term whose global denominator is zero.
- [U] the user, 2026-11-05: create a new AdamW each cycle, preserve module weights and METRA's dual but not Adam moments, use continuous independent pair RNG `[seed,4]` and checkpoint torch CPU/CUDA RNG. Use immutable per-cycle artifacts under `data/rl_cycle/<cycle:06d>/`; real-world continuation requires the live `Env` or an externally supplied matching world state, not a simulator snapshot implemented by this block.
- [U] the user, 2026-11-05: batch-1 policy calls occur in item order, with one readout shared by logits, shift and detached METRA skill. The differentiable raw encoder uses `Phi.frames_z`; starting training values are `seed=0`, `epochs=1`, `batch=32`, `micro_batch=1`, `items_per_shard=256`, all loss weights 1.0, `betas=[0.9,0.999]`, `adam_eps=1e-8`, `warmup_steps=100`, `lr_floor=0.1`, `grad_clip=1.0`, `checkpoint_steps=100`, `keep_checkpoints=2`, `device=cuda`, with pretrain's parameter-group rates and `metra_dual` replacing `phi_dual`. One CLI call runs one cycle.
- `specs/pretrain.md` (approved): the mixed-item schedule, micro-batch accumulation, global normalisation, parameter groups, clipping and checkpoint protocol to reuse where applicable. `specs/ppo.md` (approved): per-action losses and eligibility, including METRA. `specs/rl-preference-pairs.md` (approved): the fixed preference dataset and deferred persistent-memory publication. `specs/perception.md` (approved): stream rebuilding after $\phi$ changes and the shared raw-frame encoder.

## Notation
- $c$ — one RL cycle and its trajectory shard index; $N$ — the number of real actions sampled in R1; $H$ — PPO's horizon; $\tau_c$ — the fresh trajectory, continuous with the previous one; $\mathcal M_{c-1}$ — the incoming persistent History.
- $U^-$ — the number of real actions before R1; $U=U^-+N$ — the number after R1. Fresh actions are $t\in[U^-,U)$ and R3 receives states $t\in[U^-,U]$.
- A fresh-transition item references one transition $t\to t+1$ of $\tau_c$. It always supplies factual-shift and METRA losses and supplies PPO loss iff $t+H\le U$.
- A pending item references an action of an earlier trajectory whose shift/METRA training has already happened and whose $H$ consequences have only now become complete. It supplies PPO loss only.
- A preference item is one fixed pair of $Q_c$ of `specs/rl-preference-pairs.md` and supplies preference loss only.
- $N_\Delta=N$ — the number of fresh-transition items; $N_A$ — the total number of PPO-eligible positions across fresh-transition and pending items; $R_c$ — the number of preference items. An item occurs once per epoch in exactly one of these three roles.
- $I$ — all items of the cycle; $S$ — their index shards; $b$ — `batch` items per mini-batch; $b_\mu$ — `micro_batch`; $E$ — `epochs`; $M$ — the number of mini-batches per epoch, as `n_batches` of `specs/pretrain.md`.
- $w_{ppo},w_\Delta,w_M,w_P$ — positive loss weights; $\ell^{ppo},\ell^\Delta,\ell^M$ from block 12, $\ell^P$ from block 5.
- $\phi,P,\pi$ without a cycle superscript are the current, changing modules inside training. All raw state references are re-encoded by the current $\phi$ on each use.

## Math
(1) Mini-batch composition [U]: fresh/pending items are `{"kind": "fresh"|"pending", "index": t}` and preference items are self-contained `{kind,a,b,labels}` records. Apply one global `PCG64([seed,2])` permutation before cutting index shards. The three item types share one index in their natural proportions, with no per-batch quota. Per epoch, shuffle the shard order and then the items within each shard by `training.schedule` with `PCG64([seed,3])`; cut each permuted shard into consecutive mini-batches of $b$ items. Every item occurs once per epoch; a mini-batch never crosses a shard.

(2) One micro-batch [U]: read only its referenced raw data, encode all required real histories with the current $\phi$, and compute only the losses supplied by each item. PPO advantages are recomputed at this point by `specs/ppo.md` (1), (2), under stop-gradient. The factual target and the METRA skill are stop-gradient. Preference labels remain fixed. Policy calls are batch-1 in item order; a fresh/pending item has one readout shared by logits, predicted shift and the detached METRA skill, followed by one backward for the micro-batch.

(3) Loss of mini-batch $\mathcal B$ [U], generalising `specs/pretrain.md` (1):
$\mathcal L(\mathcal B)=M\left[\frac{w_{ppo}}{N_A}\sum_{t\in A(\mathcal B)}\ell_t^{ppo}+\frac1{N_\Delta}\sum_{t\in D(\mathcal B)}(w_\Delta\ell_t^\Delta+w_M\ell_t^M)+\frac{w_P}{R_c}\sum_{r\in R(\mathcal B)}\ell_r^P\right]$,
where $A,D,R$ are the supplied PPO, fresh-transition and preference entries of $\mathcal B$; an empty sum is 0. The PPO term is absent when $N_A=0$, and the preference term is absent when $R_c=0$; neither is evaluated as $0/0$. The mean of $\mathcal L$ over the epoch's $M$ mini-batches equals the corresponding global mean over every supplied entry. Each loss's gradient reaches exactly the modules fixed by its own spec: PPO/shift → $\pi,\phi$; preference → $P,\phi$; METRA → $\phi$ and its dual variable, with the predicted shift detached.

(4) Update [U]: cut $\mathcal B$ into consecutive micro-batches of $b_\mu$ items, call (2) and backward in order, accumulating gradients with the factors of (3). After the last micro-batch, apply one global gradient-norm clip and one AdamW step over the declared groups of all modules. There are no separate optimizer steps per network or per loss. A parameter never reached by any backward of the step keeps gradient `None` and is not updated. AdamW and its moments are local to one cycle; module weights and METRA's dual persist.

(5) Repeated epochs [U]: the same items occur once in each of $E$ epochs under a fresh schedule. Behaviour probabilities $\mu$, pair references and preference labels stay fixed; latents, PPO advantages and current action probabilities are recomputed on every occurrence. Pending items do not receive shift/METRA again.

## Behaviour
(C1) R1: continue the endless environment for $N$ actions using the networks left by the previous cycle. Store raw histories, actions, MCTS behaviour distributions, trees and trajectory/RNG state under block 9. No network training occurs while the environment advances. Exact real-world continuation requires the live `Env` or an externally supplied matching world state.

(C2) R3 data: run block 32 on $(\tau_c,\mathcal M_{c-1},\phi_c,P_c)$ before any update of cycle $c$. Its output $Q_c$ is immutable for the rest of the cycle. Seed insertion, comparison updates, aging and cleanup remain staged; no History generation is published before training.

(C3) Index: write shards containing fresh-transition items, newly eligible pending items and preference items. A fresh action without $H$ consequences stays pending; absence of a PPO reward does not discard its shift/METRA transition. An action enters a PPO dataset only when its horizon is complete and is not reintroduced as pending afterwards. Write the remaining tail $[\max(0,U-H+1),U)$ as an explicit atomic shard with continuation state.

(C4) Joint training: perform (1)–(5) with checkpoints every configured number of optimizer steps. A rerun restores the modules, optimizer, torch CPU/CUDA RNG and exact position in the schedule and continues after the last complete checkpoint shard. Pair RNG `[seed,4]` is independent and continuous across cycles.

(C5) R5: after training, call block 32 to re-encode only selected seeds with final $\phi$, apply aging and cleanup, and publish the new persistent History. Save the remaining PPO tail for the next cycle. Begin the next R1 with the final $\phi,P,\pi$ and rebuild perception streams because $\phi$ changed. Stored MCTS trees retain their old coordinates under `specs/mcts.md`.

## Interface
- Planned subpackage `src/mcpref/rl/`: `cycle.py` — (C1)–(C5); `index.py` — (C3) and the mixed item shards; `loss.py` — composition of block 12 and preference losses in (2), (3); `loop.py` — (1), (4), (5) and checkpoints. Exact function signatures remain to be fixed when the loss block and `Phi.frames_z` are implemented.
- `configs/rl-cycle.yaml`: paths and the training fields corresponding to Notation and Sources, including `batch`, `micro_batch`, `epochs`, the four loss weights, optimizer/schedule, checkpoint cadence and retention. One CLI call runs one cycle.
- Consumes: block 9 trajectories and pending tails; block 32's $Q_c$ and deferred History publication; block 12's PPO, factual-shift and METRA losses; blocks 2, 5, 6 modules and parameter groups; `training.schedule` and the reusable update/checkpoint mechanisms of `specs/pretrain.md`.
- Produces under `data/rl_cycle/<cycle:06d>/`: `run.json`, `index/`, `checkpoints/`, `model.pt`, the next pending PPO tail and the state required by the next R1. Phases are `sampled → pairs → indexed → trained → finalized`. Fingerprints use immutable producer identities, config hashes and $Q_c$ identity. All `losses.json` are retained; only old `checkpoint.pt` files are pruned by `keep_checkpoints`.
- Datasets: the mixed index and tail are written shard by shard before training; raw trajectories and $Q_c$ are read one shard at a time, and only micro-batch raw data is resident besides continuation state. Exact shard layouts remain to be fixed during block clarification.
- Progress bars: `rl cycle` (unit `step`) over optimizer steps, with restored initial position and postfix `{ppo, shift, metra, P, loss}`; `rl index` (unit `item`) over mixed items; block 32 owns `rl memory` (unit `seed`) and its other bars. Separate bars of block 9 retain their owner.

## Tests
- Schedule and natural mixing against the pretrain schedule oracle; each item once per epoch and no cross-shard batch.
- (3) against a float64 oracle with every item role, empty role sums, zero global PPO/preference denominators, unequal role counts and micro-batches; global epoch normalisation.
- Gradient-flow tests for all four losses and every stop-gradient; pending items not retraining shift/METRA; fixed labels and behaviour probabilities across epochs with changing current latents/advantages.
- Full-`read_history` rebuilding across trajectory and perception-stream boundaries; sampling-time latent, predicted-shift and preference fields excluded from PPO/factual-shift training (`specs/ppo.md`).
- A tiny two-cycle end-to-end run with $H=N=2$: tail deferred then completed, pair production before updates, one joint update per mini-batch, deferred final-$\phi$ selected-seed publication, previous-cycle nearest/beam and stream rebuilding. Cover empty $Q_c$ and $N_A=0$ separately.
- Checkpoint interruption inside an epoch, at an epoch boundary and after every cycle phase; exact continuation, completion and the config loader.

## Library mapping
Reuse the vetted mappings of `specs/pretrain.md` for schedule, gradient accumulation, clipping, AdamW and checkpoint calls, and of `specs/ppo.md`, `specs/env-sampling.md` and `specs/rl-preference-pairs.md` for their owned computations.

## Out of scope
- The environment wrapper, PPO/factual-shift/METRA math, pair-selection and memory algorithms themselves; simulator world snapshots; EBU and retrospective tree training; validation; distributed training; automatic PPO/shift/METRA losses on preference items.
