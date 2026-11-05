# PPO
Status: approved (2026-11-05)
Stage: block 12 of `specs/00-overview.md` — the $H$-step preference advantage, the clipped policy loss and the factual-shift loss for R4

## Sources
- [U] the user, 2026-11-03: for each real state independently, compare it with each of the next $H$ real states by the MCTS preference value, discount the comparisons with the MCTS $\gamma$, and use their sum as the reward of the action chosen at that state. A state without all $H$ consequences has no PPO reward until a later sampling cycle supplies them. The reward itself is the advantage: no critic, returns, GAE or second discounting. PPO increases the probability of an action with positive advantage and decreases it for a negative advantage.
- [U] the user, 2026-11-03: use the stored MCTS visit distribution as the behaviour probability; recompute real latents and advantages with the current $\phi,P$ on every optimizer step; the PPO advantage and the factual shift target are stop-gradient. Gradients of the policy losses flow into $\phi$. The shift head is trained outside PPO proper, against factual consequences in the coordinates of the current $\phi$.
- [U] the user, 2026-11-05: starting `horizon` = 64 and `clip` = 0.2; no entropy bonus or KL penalty. `reward` receives `horizon` explicitly and accepts only complete eligible rows. Block 13 rebuilds each real history through the full `read_history` context. Inputs of each loss share one floating dtype and device. R4 keeps the modules in train mode, including the policy's approved action dropout.
- `specs/mcts.md` (approved): $V=(p_{inc}+m_{min})(2p_\succ-1)$, the depth weight $\gamma^d$, and the stored decision distribution. `specs/preference.md` (approved): $P$ returns the logits $(s,u)$ of $(p_\succ,p_{inc})$. `specs/policy.md` (approved): full unmasked logits, histories $(Y,C)$, the action latency convention, action dropout and the squared $d_\alpha$ shift loss. `specs/env-sampling.md` (approved): `StepRecord`, `load_step` and the full raw-history reader. `specs/metra.md` (future): R2's separate METRA loss.
- [T] torch 2.14.1, installed package and source at tag `v2.14.1`, vetted 2026-11-05: `nn/functional.py:2293-2324` (`log_softmax` dispatch); `aten/src/ATen/native/SoftMax.cpp`, `cpu/SoftMaxKernel.cpp`, `cuda/SoftMax.cu` (stable $x-\max x-\log\sum\exp(x-\max x)$ and its backward); `TensorCompare.cpp`, `cpu/TensorCompareKernel.cpp`, `cuda/UnaryOpsKernel.cu` (scalar clamp); `BinaryOps.cpp`, `cpu/BinaryOpsKernel.cpp`, `cuda/MaxMinElementwiseKernel.cu` (`minimum`); `UnaryOps.cpp`, `cpu/UnaryOpsKernel.cpp`, `cuda/UnaryOpsKernel.cu` (`exp`, `square`); `TensorAdvancedIndexing.cpp`, `cpu/IndexKernel.cpp`, `cuda/IndexKernel.cu` (advanced indexing); `RangeFactories.cpp`, `cuda/RangeFactories.cu` (`arange`); `TensorShape.cpp:3418-3431,4418-4426` (`stack`, `detach`); installed `torch/autograd/grad_mode.py:22-86` (`no_grad`); `tools/autograd/derivatives.yaml` (`index.Tensor` backward, `minimum` splitting an exact tie equally between its two inputs). Sigmoid is vetted in `specs/mcts.md` [T].
- [P] CPython 3.10.20, source at tag `v3.10.20`: `Python/bltinmodule.c`, `builtin_sum_impl` (iteration in input order, default start 0, then repeated `PyNumber_Add` for tensor items).

## Notation
- $\phi$ — current perception; $P$ — current preference network; $\pi_\theta$ — current policy with parameters $\theta$, including its input encoding by $\phi$. $D$ — latent size; $A$ — the action set of `specs/policy.md`; $m\ge1$ — number of per-action rows in a call; $i=0,\dots,m-1$ — row index.
- $t$ — a real action position of the endless timeline; $s_t$ — the real `StepRecord.before` state before the action; $a_t$ — the action chosen by MCTS at that position; $\mu_t(a_t)>0$ — its stored, fixed visit probability. The action and its input history follow the one-interval convention of `specs/policy.md`.
- $\lambda_t\in\mathbb R^{|A|}$ — the policy logits used for $\rho_t$; $\ell_t,\ell_{t,k}\ge1$ — history lengths; $L$ — `states` of `configs/perception-data.yaml`, the real-history cap; $R$ — the last policy readout used for its logits and shift.
- $z_t$ — the real latent recomputed by the current $\phi$ from the stored raw history; $Y_t$ — the latent history ending at $z_t$, truncated by $P$ and $\pi$ internally to their own configured history lengths; $C_t$ — the corresponding executed-action weights.
- $H$ — `horizon`, a positive integer; $\gamma$ — `discount` and $m_{min}$ — `min_p` of `configs/mcts.yaml`, not duplicate PPO hyperparameters. A position $t$ is PPO-eligible iff the continuous stored timeline contains all real states $t+1,\dots,t+H$. Eligibility is independent for every $t$; there is no bootstrap or partial-horizon reward.
- $k$ — consequence index, $1\le k\le H$; $(s_{t,k},u_{t,k})$ — the two logits returned by $P$; $p_{\succ,t,k},p_{inc,t,k}$ — their sigmoids; $V_{t,k}$ — the preference value; $A_t$ — the advantage.
- $\sigma$ — the logistic sigmoid; $\epsilon$ — `clip`, in $(0,1)$; $\mathrm{sg}$ — stop-gradient (`detach`); $\rho_t$ — the PPO ratio; $\ell^{ppo}_t$ — the per-action clipped policy loss.
- $\pi_\theta$'s action probabilities are the full softmax of `specs/policy.md` (5), with no action mask in the training forward.
- $d_\alpha$ — `latent_metric` of `specs/perception.md`; $\widehat{\Delta z}_t(a_t)$ — the current policy's predicted shift for the executed action; $\ell^\Delta_t$ — the per-transition factual-shift loss.

## Math
(1) For an eligible $t$ and each $k=1,\dots,H$ [U], with the current $\phi,P$:
$(s_{t,k},u_{t,k})=P(\mathrm{sg}(Y_{t+k}),\mathrm{sg}(Y_t))$,
$p_{\succ,t,k}=\sigma(s_{t,k})$, $p_{inc,t,k}=\sigma(u_{t,k})$,
$V_{t,k}=(p_{inc,t,k}+m_{min})(2p_{\succ,t,k}-1)$.
The first argument is the consequence, the second its origin. Every pair is computed independently: exactly $H$ calls of $P$ per eligible action, not a reused cumulative return. Each batch-1 call's two outputs are scalarized immediately by `s[0]`, `u[0]`, so $V_{t,k}$ and $A_t$ are 0-d tensors and `torch.stack(..., dim=0)` over $m$ origins returns shape $(m,)$.

(2) Advantage [U]:
$A_t=\mathrm{sg}\left(\sum_{k=1}^{H}\gamma^k V_{t,k}\right)$, summed in ascending $k$.
The immediate consequence has weight $\gamma$, as a depth-1 leaf in MCTS backup. There is no discounted sum over $A_{t+j}$, no GAE, no value baseline, no reward or advantage normalisation and no clipping of $V$ or $A$.

(3) PPO ratio and per-action clipped loss [U]:
$\rho_t(\theta)=\pi_\theta(a_t\mid Y_t,C_t)/\mathrm{sg}(\mu_t(a_t))$,
$\ell^{ppo}_t=-\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)$.
$\mu_t$ is fixed at sampling and detached in the loss. The current $Y_t,C_t$ are recomputed on every optimizer step; gradients reach the parameters of $\pi$ and, through $Y_t$, $\phi$. The computation of $A_t$ is under `torch.no_grad`, and `ppo_loss` also detaches its `advantages` input, so neither $P$ nor $\phi$ receives a PPO gradient through the advantage. The ratio is computed as $\exp(\log\mathrm{softmax}(\lambda_t)_{a_t})/\mu_t(a_t)$, a mathematically identical stable rewrite. The action mask is absent.

(4) Factual-shift loss for every real transition with $z_{t+1}$, whether or not $t$ is PPO-eligible [U]:
$\ell^\Delta_t=d_\alpha^2\left(\widehat{\Delta z}_t(a_t)-\mathrm{sg}(z_{t+1}-z_t)\right)$.
Both latents come from the current $\phi$. The gradient reaches $\pi$ through its shift output and $\phi$ through the prediction's input history, but not through the factual target. This term is the RL counterpart of the shift term of `specs/policy.md` (9), not a PPO term.

## Domain
- Inputs are real, finite observations and histories of one continuous timeline; all networks and latents share the caller's device and use the dtypes of their approved blocks. Each tensor function has $m\ge1$ rows. Exact tensor shapes, dtypes and action-index bounds are fixed in Interface.
- During R4 the caller keeps $\phi,\pi,P$ in train mode, so the approved action dropout of `specs/policy.md` (2) is active. `reward` changes neither module modes nor `requires_grad`; only its local preference/advantage computation is under `torch.no_grad`. The policy and $\phi$ remain differentiable for (3), (4).
- The pending tail is not terminal and is never padded with zero advantages. Death/respawn handling is the environment's; no episode-boundary rule is introduced here. Trajectory/shard boundaries and perception-stream restarts do not interrupt PPO-eligibility.
- An action used for PPO has its originally stored $\mu_t(a_t)$, however many cycles later its horizon becomes complete. Sampling-time `StepRecord.delta`, `predicted`, `preference_close` and `preference_start` are not PPO or factual-shift training inputs.

## Interface
- `src/mcpref/ppo/reward.py`: `def reward(P: Preference, origins: Sequence[Tensor], futures: Sequence[Sequence[Tensor]], horizon: int, discount: float, min_p: float) -> Tensor` — (1), (2). `len(origins) == len(futures) == m`; `horizon` = $H$ and `len(futures[t]) == horizon` for every row, so only eligible positions are accepted. `origins[t]` is $Y_t$ of shape $(1,\ell_t,D)$, `futures[t][k-1]` is $Y_{t+k}$ of shape $(1,\ell_{t,k},D)$, all $\ell\ge1$; lengths may differ. Every history is finite and in $P$'s floating dtype/device; `discount` is in $(0,1]$ and `min_p` > 0. Row $t$ of the result is $A_t$: shape $(m,)$, dtype/device of $P$'s logits, `requires_grad=False`. The whole computation is under `torch.no_grad`, without padding or mode changes.
- `src/mcpref/ppo/loss.py`: `def ppo_loss(logits: Tensor, actions: Tensor, behavior: Tensor, advantages: Tensor, clip: float) -> Tensor` — (3) per row. `logits` has shape $(m,|A|)$; `behavior` and `advantages` have shape $(m,)$ and the same floating dtype/device as `logits`; `actions` is int64 on that device, shape $(m,)$, with $0\le a_i<|A|$; `behavior[i]` = $\mu_t(a_t)>0$; `clip` = $\epsilon$. The function detaches `behavior` and `advantages` internally and returns shape $(m,)$ in `logits`' dtype.
- `src/mcpref/ppo/loss.py`: `def shift_loss(predicted: Tensor, actual: Tensor, metric: Callable[[Tensor], Tensor]) -> Tensor` — (4) per row. `predicted` = $\widehat{\Delta z}_t(a_t)$ and `actual` = the current $z_{t+1}-z_t$ (not stored `StepRecord.delta`), both $(m,D)$ in one floating dtype/device; `metric` = $d_\alpha$, returning $(m,)$. The function detaches `actual` internally and returns `metric(predicted - actual.detach()).square()`, shape $(m,)$, in `predicted`'s dtype.
- `src/mcpref/ppo/config.py`: `@dataclass(frozen=True) class PpoConfig` with fields in order `horizon: int`, `clip: float`; `def load_config(path: Path) -> PpoConfig`. Read `configs/ppo.yaml` with `config_mapping`: missing/unknown keys and violated constraints raise `ValueError`; `horizon` is `positive_int`, `clip` is `number` strictly in $(0,1)$. Starting values: `horizon` = 64, `clip` = 0.2 [U].
- Block 13's policy wiring, with no additional PPO truncation or action mask: `R = pi.readout(Y, C)[:, -1]`, then `logits = pi.logits(R)` and `predicted = pi.shift(R, actions)`; `Preference` and `Policy` apply their own history limits internally.
- Block 13's history wiring: for every required real state $s_t$, obtain `RawHistory` by `read_history(out_dir, t, states)` from block 9, encode `RawHistory.states` in order with the current $\phi$ from `RawHistory.restart`, construct executed-action rows from `RawHistory.actions`, and take the last $L$ latent rows. It obtains $a_t$ and the full stored probability vector through `load_step`/`StepRecord.action` and `StepRecord.probabilities`, and pending item references and eligibility from its own index/continuation state. Block 12 itself consumes only the tensors passed by block 13 and owns no pending dataset.
- Consumes: raw histories, actions and behaviour probabilities from block 9; pending positions and current $\phi,P,\pi$ from block 13; $\gamma,m_{min}$ from block 8; $d_\alpha$ from block 2. Produces per-action losses for block 13. Datasets, checkpoints, files and progress bars of its own: none.

## Tests
- `tests/ppo/test_reward.py`: CPU float64 oracles for (1) direction, sigmoids, all $H$ independent consequence-first pairs and current histories of different lengths; (2) hand-computed ascending $\gamma^k$ sums, no normalisation, exact complete `horizon`, output shape/dtype and `requires_grad=False`; no gradient reaches histories or $P$, module modes and the outer grad mode are unchanged.
- `tests/ppo/test_loss.py`: CPU float64 oracles for (3), positive, negative and zero advantages below, inside, at and above both clipping bounds; fixed detached behaviour probabilities, full unmasked softmax, action indexing, ratio and loss dtype, gradients through logits but not behavior/advantages; exact clipping-tie values and the mapped autograd tie rules. (4) a hand-computed squared $d_\alpha$, output shape/dtype, gradients through prediction but not its target, and factual transitions without PPO eligibility.
- `tests/ppo/test_config.py`: field order, frozen dataclass, valid boundaries, missing/unknown keys and invalid values; the starting values 64 and 0.2 in `configs/ppo.yaml`.
- A tiny `Policy` wiring check uses its last readout for both losses in train mode, with action dropout active. Block 13's integration tests cover full-`read_history` rebuilding across trajectory and perception-stream boundaries and exclusion of sampling-time latent/preference fields.

## Revisions on approval
- `specs/00-overview.md`: block 12's Questions becomes `` `specs/ppo.md`, EBU ``; the package map gains `ppo` → R4 → `specs/ppo.md`.
- `specs/rl-cycle.md`: its source reference to `specs/ppo.md` loses `(draft)`; Tests gain full-`read_history` rebuilding and exclusion of sampling-time latent/preference fields as specified here. The RL-cycle spec itself remains draft.
- `specs/00-questions.md`: POL-5, POL-6, RLS-3 and RLS-4 lose `(draft)` only from their PPO links; PPO-1/2 link this approved spec instead of its draft, and PPO-4 gains a new link to it. PPO-3 records `clip` = 0.2, the stored MCTS denominator, no entropy bonus or KL penalty, and leaves epochs/minibatches to block 13; PPO-1 also records starting `horizon` = 64.

## Library mapping
| Eq. | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1) | `Preference.forward`, `torch.sigmoid`, `torch.no_grad` | this repo; torch 2.14.1 | consequence first, origin second; scalarize each logit by `[0]` | `specs/preference.md`; [T] |
| (2) | Python `sum`, `torch.stack` | CPython 3.10.20; torch 2.14.1 | ascending $k=1..H$; `dim=0` | [P]; [T] `TensorShape.cpp:3418-3431` |
| (3) | `torch.log_softmax`, `torch.arange`, advanced indexing, `torch.exp` | 2.14.1 | `dim=-1`, `dtype=None`; row indices `torch.arange(0, m, 1, dtype=torch.int64, device=logits.device)` and int64 `actions` | [T], including `RangeFactories.cpp`, `cuda/RangeFactories.cu` |
| (3) | `torch.clamp`, `torch.minimum`, `Tensor.detach` | 2.14.1 | `min=1-clip`, `max=1+clip`; detach `behavior` and `advantages` | [T], including `derivatives.yaml` |
| (4) | `latent_metric`, `Tensor.square`, `Tensor.detach` | this repo; torch 2.14.1 | metric on `predicted - actual.detach()`, then square | `specs/perception.md`; [T] |
| wiring | `Policy.readout`, `Policy.logits`, `Policy.shift`; `read_history`, `load_step` | this repo | last readout row; full raw context from `RawHistory.restart` | `specs/policy.md`, `specs/env-sampling.md`; installed implementations |

## Out of scope
- The optimizer, batch construction, loss weights, epochs and checkpoints (block 13); preference-pair production (block 32); METRA (block 10); memory and environment sampling (blocks 9, 11); EBU and retrospective tree updates; entropy bonuses, KL penalties, critics, GAE and advantage normalisation unless separately approved.
