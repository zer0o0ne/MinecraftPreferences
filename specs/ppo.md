# PPO
Status: draft
Stage: block 12 of `specs/00-overview.md` — the $H$-step preference advantage, the clipped policy loss and the factual-shift loss for R4

## Sources
- [U] the user, 2026-11-03: for each real state independently, compare it with each of the next $H$ real states by the MCTS preference value, discount the comparisons with the MCTS $\gamma$, and use their sum as the reward of the action chosen at that state. A state without all $H$ consequences has no PPO reward until a later sampling cycle supplies them. The reward itself is the advantage: no critic, returns, GAE or second discounting. PPO increases the probability of an action with positive advantage and decreases it for a negative advantage.
- [U] the user, 2026-11-03: use the stored MCTS visit distribution as the behaviour probability; recompute real latents and advantages with the current $\phi,P$ on every optimizer step; the PPO advantage and the factual shift target are stop-gradient. Gradients of the policy losses flow into $\phi$. The shift head is trained outside PPO proper, against factual consequences in the coordinates of the current $\phi$.
- `specs/mcts.md` (approved): $V=(p_{inc}+m_{min})(2p_\succ-1)$, the depth weight $\gamma^d$, and the stored decision distribution. `specs/preference.md` (approved): $P$ returns the logits $(s,t)$ of $(p_\succ,p_{inc})$. `specs/policy.md` (approved): full unmasked logits, histories $(Y,C)$, the action latency convention and the squared $d_\alpha$ shift loss. `specs/metra.md` (future): R2's separate METRA loss.

## Notation
- $t$ — a real action position of the endless timeline; $a_t$ — the action chosen by MCTS at that position; $\mu_t(a_t)>0$ — its stored visit probability. The action and its input history follow the one-interval convention of `specs/policy.md`.
- $z_t$ — the real latent recomputed by the current $\phi$ from the stored raw history; $Y_t$ — the latent history ending at $z_t$, truncated by $P$ and $\pi$ to their own configured history lengths; $C_t$ — the corresponding executed-action weights.
- $H$ — `horizon`, a positive integer; $\gamma$ — `discount` and $m_{min}$ — `min_p` of `configs/mcts.yaml`, not duplicate PPO hyperparameters.
- A position $t$ is PPO-eligible iff the continuous stored timeline contains all real states $t+1,\dots,t+H$. Eligibility is independent for every $t$; there is no bootstrap or partial-horizon reward.
- $\sigma$ — the logistic sigmoid; $\epsilon$ — `clip`, in $(0,1)$; $\mathrm{sg}$ — stop-gradient (`detach`).
- $\pi_\theta$ — the current policy, including its input encoding by the current $\phi$; its action probabilities are the full softmax of `specs/policy.md` (5), with no action mask in the training forward.
- $d_\alpha$ — `latent_metric` of `specs/perception.md`; $\widehat{\Delta z}_t(a_t)$ — the current policy's predicted shift for the executed action.

## Math
(1) For an eligible $t$ and each $k=1,\dots,H$ [U], with the current $\phi,P$:
$(s_{t,k},u_{t,k})=P(\mathrm{sg}(Y_{t+k}),\mathrm{sg}(Y_t))$,
$p_{\succ,t,k}=\sigma(s_{t,k})$, $p_{inc,t,k}=\sigma(u_{t,k})$,
$V_{t,k}=(p_{inc,t,k}+m_{min})(2p_{\succ,t,k}-1)$.
The first argument is the consequence, the second its origin. Every pair is computed independently; there are approximately $H$ pairs per eligible action, not a reused cumulative return.

(2) Advantage [U]:
$A_t=\mathrm{sg}\left(\sum_{k=1}^{H}\gamma^k V_{t,k}\right)$.
The immediate consequence has weight $\gamma$, as a depth-1 leaf in MCTS backup. There is no discounted sum over $A_{t+j}$, no GAE, no value baseline, no reward or advantage normalisation and no clipping of $V$ or $A$.

(3) PPO ratio and per-action clipped loss [U]:
$\rho_t(\theta)=\pi_\theta(a_t\mid Y_t,C_t)/\mu_t(a_t)$,
$\ell^{ppo}_t=-\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)$.
$\mu_t$ is fixed at sampling. The current $Y_t,C_t$ are recomputed on every optimizer step; gradients reach the parameters of $\pi$ and, through $Y_t$, $\phi$. The computation of $A_t$ is under `torch.no_grad`, so neither $P$ nor $\phi$ receives a PPO gradient through the advantage.

(4) Factual-shift loss for every real transition with $z_{t+1}$, whether or not $t$ is PPO-eligible [U]:
$\ell^\Delta_t=d_\alpha^2\left(\widehat{\Delta z}_t(a_t)-\mathrm{sg}(z_{t+1}-z_t)\right)$.
Both latents come from the current $\phi$. The gradient reaches $\pi$ through its shift output and $\phi$ through the prediction's input history, but not through the factual target. This term is the RL counterpart of the shift term of `specs/policy.md` (9), not a PPO term.

## Domain
- Inputs are real, finite observations and histories of one continuous timeline; all networks and latents share the caller's device and use the dtypes of their approved blocks. $P$ is called under `torch.no_grad` for (1); the policy and $\phi$ remain differentiable for (3), (4).
- The pending tail is not terminal and is never padded with zero advantages. Death/respawn handling is the environment's; no episode-boundary rule is introduced here.
- An action used for PPO has its originally stored $\mu_t(a_t)$, however many cycles later its horizon becomes complete.

## Interface
- `src/mcpref/ppo/reward.py`: `def reward(P: Preference, origins: Sequence[Tensor], futures: Sequence[Sequence[Tensor]], discount: float, min_p: float) -> Tensor` — (1), (2); `origins[t]` is $Y_t$ with batch dimension 1, `futures[t][k-1]` is $Y_{t+k}$, and row $t$ of the result is $A_t$. Histories may have different lengths. The function has no gradient side effect.
- `src/mcpref/ppo/loss.py`: `def ppo_loss(logits: Tensor, actions: Tensor, behavior: Tensor, advantages: Tensor, clip: float) -> Tensor` — (3) per row, with `logits` $(m,|A|)$, int64 `actions` $(m)$, positive `behavior` $(m)$ and `advantages` $(m)$; `def shift_loss(predicted: Tensor, actual: Tensor, metric: Callable[[Tensor], Tensor]) -> Tensor` — (4) per row, with both tensors $(m,D)$ and `metric` = $d_\alpha$. Each returns shape $(m,)$ and performs its specified detach internally.
- `configs/ppo.yaml` → frozen `PpoConfig`: `horizon` (`positive_int`), `clip` (`number`, in $(0,1)$). Starting numeric values are not yet chosen.
- Consumes: raw real histories, $a_t$, $\mu_t$ and pending positions from blocks 9/13; current $\phi,P,\pi$ from the RL loop; $\gamma,m_{min}$ from block 8; $d_\alpha$ from block 2. Produces per-action losses for block 13. Datasets, checkpoints, files and progress bars of its own: none.

## Tests
- CPU, float64 oracles: (1) direction, sigmoids, all $H$ independent pairs and current-history inputs; (2) hand-computed $\gamma^k$ sums, horizon boundaries and no normalisation; (3) positive, negative and zero advantages on both sides of the clipping interval, fixed behaviour probabilities, full unmasked softmax and detach; (4) a hand-computed squared $d_\alpha$, gradients through the prediction but not its target, and transitions without PPO eligibility; the config loader.

## Library mapping
To be vetted before approval: the explicit `torch.sigmoid`, `torch.log_softmax`, `torch.clamp`, `torch.minimum`, indexing and `detach`/`no_grad` calls implementing (1)–(4), with their torch 2.14.1 source references.

## Out of scope
- The optimizer, batch construction, loss weights, epochs and checkpoints (block 13); preference-pair production (block 32); METRA (block 10); memory and environment sampling (blocks 9, 11); EBU and retrospective tree updates; entropy bonuses, KL penalties, critics, GAE and advantage normalisation unless separately approved.
