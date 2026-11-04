# RL cycle
Status: draft
Stage: block 13 of `specs/00-overview.md` — R1–R5 orchestration and the one joint training loop of $\phi,P,\pi$

## Sources
- [U] the user, 2026-11-03: R2's METRA, R3's fixed-label preference fine-tuning, R4's PPO and factual-shift regression form one joint training loop, analogous to pretraining. All three networks train together; current $\phi,P$ recompute PPO advantages on every optimizer step, and gradients of the policy and preference losses flow into $\phi$.
- [U] the user, 2026-11-03: batch formation follows pretraining's principle: trajectory items and fixed preference items mixed in their natural proportion, each item once per epoch, shard order and items within a shard reshuffled per epoch, a mini-batch within one shard, gradient accumulation over micro-batches and one shared update. Unlike pretraining pair clips, an RL preference item contributes only preference loss. A trajectory item contributes the losses for which it has the required data.
- [U] the user, 2026-11-03: cycle order is R1 → fixed preference-pair production → joint training → R5. Fresh trajectory states are available to pair production before permanent memory insertion. Pending PPO tails continue into the next cycle; shift and METRA need only one factual successor. Memory is updated after training, with fresh states encoded by the final $\phi$; the next R1 uses the final networks and rebuilds perception streams.
- `specs/pretrain.md` (approved): the mixed-item schedule, micro-batch accumulation, global normalisation, parameter groups, clipping and checkpoint protocol to reuse where applicable. `specs/ppo.md` (draft): per-action losses and eligibility. `specs/rl-preference-pairs.md` (draft): the fixed preference dataset. `specs/perception.md` (approved): stream rebuilding after $\phi$ changes.

## Notation
- $c$ — one RL cycle; $N$ — the number of real actions sampled in R1; $H$ — PPO's horizon; $\tau_c$ — the fresh trajectory, continuous with the previous one; $\mathcal M_{c-1}$ — the incoming memory.
- A fresh-transition item references one transition $t\to t+1$ of $\tau_c$. It always supplies factual-shift and METRA losses and supplies PPO loss iff $t$ is eligible by `specs/ppo.md`.
- A pending item references an action of an earlier trajectory whose shift/METRA training has already happened and whose $H$ consequences have only now become complete. It supplies PPO loss only.
- A preference item is one fixed pair of $Q_c$ of `specs/rl-preference-pairs.md` and supplies preference loss only.
- $N_\Delta$ — the number of fresh-transition items; $N_A$ — the total number of PPO-eligible positions across fresh-transition and pending items; $R_c$ — the number of preference items. An item occurs once per epoch in exactly one of these three roles.
- $I$ — all items of the cycle; $S$ — their index shards; $b$ — `batch` items per mini-batch; $b_\mu$ — `micro_batch`; $E$ — `epochs`; $M$ — the number of mini-batches per epoch, as `n_batches` of `specs/pretrain.md`.
- $w_{ppo},w_\Delta,w_M,w_P$ — positive loss weights; $\ell^{ppo},\ell^\Delta$ from block 12, $\ell^M$ from block 10, $\ell^P$ from block 5.
- $\phi,P,\pi$ without a cycle superscript are the current, changing modules inside training. All raw state references are re-encoded by the current $\phi$ on each use.

## Math
(1) Mini-batch composition [U]: the three item types share one index. Their global proportions are their natural counts, with no per-batch quota. Per epoch, shuffle the shard order and then the items within each shard by `training.schedule`; cut each permuted shard into consecutive mini-batches of $b$ items. Every item occurs once per epoch; a mini-batch never crosses a shard.

(2) One micro-batch [U]: read only its referenced raw data, encode all required real histories with the current $\phi$, and compute only the losses supplied by each item. PPO advantages are recomputed at this point by `specs/ppo.md` (1), (2), under stop-gradient. The factual target and the METRA skill are stop-gradient. Preference labels remain fixed.

(3) Loss of mini-batch $\mathcal B$ [U], generalising `specs/pretrain.md` (1):
$\mathcal L(\mathcal B)=M\left[\frac{w_{ppo}}{N_A}\sum_{t\in A(\mathcal B)}\ell_t^{ppo}+\frac1{N_\Delta}\sum_{t\in D(\mathcal B)}(w_\Delta\ell_t^\Delta+w_M\ell_t^M)+\frac{w_P}{R_c}\sum_{r\in R(\mathcal B)}\ell_r^P\right]$,
where $A,D,R$ are the supplied PPO, fresh-transition and preference entries of $\mathcal B$; an empty sum is 0. The mean of $\mathcal L$ over the epoch's $M$ mini-batches equals the corresponding global mean over every supplied entry. Each loss's gradient reaches exactly the modules fixed by its own spec: PPO/shift → $\pi,\phi$; preference → $P,\phi$; METRA → $\phi$ and its dual variable, with the predicted shift detached.

(4) Update [U]: cut $\mathcal B$ into consecutive micro-batches of $b_\mu$ items, call (2) and backward in order, accumulating gradients with the factors of (3). After the last micro-batch, apply one global gradient-norm clip and one AdamW step over the declared groups of all modules. There are no separate optimizer steps per network or per loss. A parameter never reached by any backward of the step keeps gradient `None` and is not updated.

(5) Repeated epochs [U]: the same items occur once in each of $E$ epochs under a fresh schedule. Behaviour probabilities $\mu$, pair references and preference labels stay fixed; latents, PPO advantages and current action probabilities are recomputed on every occurrence. Pending items do not receive shift/METRA again.

## Behaviour
(C1) R1: continue the endless environment for $N$ actions using the networks left by the previous cycle. Store raw histories, actions, MCTS behaviour distributions, trees and trajectory/RNG state under block 9. No network training occurs while the environment advances.

(C2) R3 data: run block 32 on $(\tau_c,\mathcal M_{c-1},\phi_c,P_c)$ before any update of cycle $c$. Its output $Q_c$ is immutable for the rest of the cycle.

(C3) Index: write shards containing fresh-transition items, newly eligible pending items and preference items. A fresh action without $H$ consequences stays pending; absence of a PPO reward does not discard its shift/METRA transition. An action enters a PPO dataset only when its horizon is complete and is not reintroduced as pending afterwards.

(C4) Joint training: perform (1)–(5) with checkpoints every configured number of optimizer steps. A rerun restores the modules, optimizer, RNG states and exact position in the schedule and continues after the last complete checkpoint shard.

(C5) R5: after training, insert $\tau_c$ into memory; fresh memory latents are encoded by the final $\phi$. Save the remaining PPO tail for the next cycle. Begin the next R1 with the final $\phi,P,\pi$ and rebuild perception streams because $\phi$ changed. Stored MCTS trees retain their old coordinates under `specs/mcts.md`.

## Interface
- Planned subpackage `src/mcpref/rl/`: `cycle.py` — (C1)–(C5); `index.py` — (C3) and the mixed item shards; `loss.py` — (2), (3); `loop.py` — (1), (4), (5) and checkpoints. Exact function signatures remain to be fixed after block 9's trajectory format and block 10's METRA interface.
- `configs/rl-cycle.yaml`: paths and the training fields corresponding to Notation, including `batch`, `micro_batch`, `epochs`, the four loss weights, optimizer/schedule, checkpoint cadence and retention. Types and starting values are not yet fixed.
- Consumes: block 9 trajectories and pending tails; block 32's $Q_c$; block 12's losses; block 10's METRA; blocks 2, 5, 6 modules and parameter groups; block 11 memory; `training.schedule` and the reusable update/checkpoint mechanisms of `specs/pretrain.md`.
- Produces: cycle checkpoints, updated module states, the next pending PPO tail, updated memory and the state required by the next R1. Concrete artifact paths and formats remain open.
- Datasets: the mixed index is written shard by shard before training; raw trajectories and $Q_c$ are read one shard at a time, and only micro-batch raw data is resident besides continuation state. Exact shard layouts and rerun fingerprints are open.
- Progress bars: `rl cycle` (unit `step`) over optimizer steps, with restored initial position; separate bars of blocks 9 and 32 retain their owners. Index-building and memory-update bars are to be named with their formats.

## Tests
- Schedule and natural mixing against the pretrain schedule oracle; each item once per epoch and no cross-shard batch.
- (3) against a float64 oracle with every item role, empty role sums, unequal role counts and micro-batches; global epoch normalisation.
- Gradient-flow tests for all four losses and every stop-gradient; pending items not retraining shift/METRA; fixed labels and behaviour probabilities across epochs with changing current latents/advantages.
- A tiny end-to-end cycle: tail deferred then completed, pair production before updates, one joint update per mini-batch, R5 after training, final-$\phi$ memory encoding and stream rebuilding.
- Checkpoint interruption inside an epoch and at an epoch boundary, exact continuation, completion and the config loader.

## Library mapping
Reuse the vetted mappings of `specs/pretrain.md` for schedule, gradient accumulation, clipping, AdamW and checkpoint calls. Additional mappings for block 9/10/11/32 calls remain to be vetted after their interfaces are fixed.

## Out of scope
- The environment wrapper, PPO math, METRA math, pair-selection algorithm and memory search themselves; EBU and retrospective tree training; validation; distributed training; automatic PPO/shift/METRA losses on preference items.
