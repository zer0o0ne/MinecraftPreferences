# Pipeline overview
Status: draft
Sources: [U] the user, 2026-10-09. Open questions: `specs/00-questions.md`, cited as (Q <id>).

## Principles
- One linear timeline: a single Minecraft environment, never forked, rewound or reset to try alternatives. Alternatives exist only as imagined MCTS branches in latent space.
- The game is paused while networks train; every sampling phase resumes from the state where the previous one stopped.
- Every input of the preference network is encoded by the current perception.
- Pretraining uses the VPT dataset and VLM labels; RL uses the environment, the networks and the memory (Q G-4).
- Data and environment: the original VPT recordings (640×360, every tick) plus BASALT 2022; RL and new human recordings in the MineStudio simulator (MC 1.16.5). Python 3.10.
- Elementary states are a fixed $\Delta$ ticks apart; the agent acts once per $\Delta$; human data carries, per interval, the number $n$ of legal actions of $A$ made in it, and only sequences with $n \ge 1$ in every interval are used (Q G-12, G-13, G-14).
- Every exact number (hyperparameters, thresholds, API constants) is a config field (Q G-11).
- Inference speed ([U] 2026-10-20): the agent's inference as fast as possible; all of perception ideally within 150 ms per step with a ~2B transformer inside; every component on the per-step path states a latency budget in its spec.

## Shared notation
- $o_t$ — state at step $t$: frame + meta (inventory, position, …) (Q D-7, D-8).
- $h_t$ — window of consecutive elementary states, $\Delta$ ticks apart, ending at $t$ (Q G-12).
- $\phi$ — perception; $z_t = \phi(h_t) \in \mathbb{R}^D$ — internal (latent) state.
- $\alpha \in \mathbb{R}^D_{>0}$ — per-coordinate metric weights (the user's $a_i$) (Q PER-2); $d_\alpha^2(z, z') = \sum_i \alpha_i^2 (z_i - z'_i)^2$. METRA geometry holds in the coordinates $\alpha \odot z$, the norm bound for $z$.
- $c$ — norm bound of $z$ (example: 1).
- $P$ — preference network; $P(x, y) = (p_\succ, p_{cmp})$: probability that $x$ is preferred to $y$, probability that $x, y$ are comparable; $x, y$ are latent histories.
- $A$ — discrete action set (Q ACT, versions 1 and 2): 34 world actions (`WORLD_ACTIONS` of `specs/vpt-actions.md`) and 361 GUI actions (`gui_action_names` of `specs/vpt-gui-actions.md`: close, left, right, shift and double click on element $i = 1..n_{max}$ of the recognised screen), 395 in all; masked by GUI state, screen type and pitch; a screen that is not recognised is closed at once (Q ACT-19). An action holds its input until its effect completes, at most $\Delta$, and continues without release while chosen again (Q G-13); $\pi$ learns to lower the logits of masked actions (Q POL-7).
- $\pi$ — policy; $\pi(\cdot \mid z\text{-history})$ — probabilities over $A$; $\widehat{\Delta z}(a)$, $a \in A$ — predicted latent shift of each action (output $|A| \times D$).
- $\Delta z_t = z_{t+1} - z_t$ — actual latent shift.
- $\mathcal{M}$ — memory of past states (Q MEM-1, MEM-2); $\mathcal{T}$ — stored MCTS trees.
- $C(\cdot)$ — preference correction: maximum-weight acyclic subgraph plus modifications (Q ACY-1).

## Networks
| Net | Input | Output | Trained in |
|---|---|---|---|
| $\phi$ perception | $h_t$ (frames + meta) | $z_t$ | P1 (truncated METRA loss), R2 (METRA loss) |
| $P$ preference | two latent histories | $p_\succ$, $p_{cmp}$ | P4 (corrected VLM labels), R3 (corrected self-labels) |
| $\pi$ policy | last $k_\pi$ latent states | probabilities over $A$, $\widehat{\Delta z}(a)$ for every $a \in A$ | P5 (action + shift regression), R4 (PPO) |

## Pretraining (VPT dataset)
- **P1 Perception.** Random dataset pairs $(h_t, h_{t+\Delta})$, with the number $n$ of legal actions of every interval (Q G-12, PER-9). Loss: maximise $d_\alpha(z_t, z_{t+1})$ subject to $d_\alpha^2 \le 1$ via METRA's dual variable, only on transitions that changed something (meta differs or the pixel difference exceeds a threshold); unchanged transitions are dropped (Q PER-8), plus $\beta_c \max(0, \lVert z \rVert_2 - c)^2$ for both $z_t, z_{t+1}$. Large $\alpha_i$ allow long chains along coordinate $i$ inside the ball.
- **P2 VLM labels.** A random pool of dataset states; each is linked at random to several others of the pool; while there are many components and few cycles, random edges between components are added (Q VLM-2). `Qwen3-VL-30B-A3B-Instruct-FP8`, served on an NVIDIA DGX Spark (`specs/vlm-client.md`), gets each state as several screenshots plus the dataset info (inventory, …) and answers: first preferred, second preferred, or refuses to compare (Q VLM-*).
- **P3 Correction.** VLM labels → preference graph → $C$ → training labels (Q ACY-*).
- **P4 Preference.** Inputs: $\phi$ (from P1) applied to each state with its history. Targets: the labels from P3 for both outputs.
- **P5 Policy.** Random consecutive pairs $(t, t+1)$ with history. Input: latent history up to $t$. Targets: $\Delta z_t$ for $\widehat{\Delta z}$, the dataset action $a_t$ for the action head (Q POL-4).

## RL cycle (repeats; one cycle = one trajectory)
- **R1 Sampling**, $n$ steps from where the previous cycle stopped. Each step:
  1. Encode the current history; run MCTS with $\pi$ and $P$: $\pi$ gives action probabilities and $\widehat{\Delta z}$ → move to the imagined state $z + \widehat{\Delta z}$ → evaluate $P$ there → $\pi$ again; repeated as in classic MCTS (Q MCTS-*).
  2. Choose the action from the tree; act; observe $o_{t+1}$.
  3. Store: action probabilities, the action, $\Delta z_t$ and $\widehat{\Delta z}_t$, and $P$ between the start point and the result (both outputs) (Q RLS-3).
  4. Keep the tree: later decisions extend it; it is used for retrospective training (Q MCTS-7). Imagined nodes keep their latent coordinates after $\phi$ changes; the networks must cope with stale coordinates.
- **R2 Perception.** METRA perception loss on the trajectory with $\widehat{\Delta z}_t$ as METRA's skill $z$, in the coordinates $\alpha \odot z$: $(\alpha \odot \Delta z_t)^\top (\alpha \odot \widehat{\Delta z}_t)$ s.t. $d_\alpha^2 \le 1$ (Q MET-*).
- **R3 Preference.** Pick random trajectory states. For each: nearest past states in $\mathcal{M}$, then a beam search in $\mathcal{M}$ along most / least preferred states (Q MEM-5) → several memory states (the trajectory itself is in $\mathcal{M}$). Label the pairs with the current $P$ on inputs encoded by the $\phi$ from R2 → $C$ → labels → fine-tune $P$.
- **R4 Policy.** PPO maximizing the preferences observed in the trajectory, both between close points and over the episode or its large parts; no networks beyond $\phi$, $P$, $\pi$ (Q PPO-*). Past trees from $\mathcal{T}$ are trained on as in Episodic Backward Update (Lee, Sung, Choi, NeurIPS 2019, arXiv:1805.12375); this retrospective training is mixed into the fresh trajectory, to learn from errors and successes of the distant past (Q EBU-*).
- **R5 Memory update** (Q MEM-1).

## Data flow
```
VPT dataset ─P1→ φ0
VPT dataset ─P2 (VLM)→ labels ─P3 (C)→ corrected labels ─P4 (φ0)→ P0
VPT dataset ─P5 (φ0)→ π0
(φ, P, π, M, T) + env ─R1→ trajectory τ, trees ─R2→ φ' ─R3 (M, P, φ')→ P' ─R4 (τ, T, π)→ π' ─R5→ M' ─→ R1 …
```

## Candidate blocks
Each goes through the block cycle; split or merge when its spec is clarified.
| # | Block | Stages | Depends on | Questions |
|---|---|---|---|---|
| S | spark-env — `setup_env.sh`, `pyproject.toml`, `uv.lock`; script conventions | all | — | `specs/spark-env.md` |
| 0 | vlm-client — Python API to the served VLM; `serve_vlm.sh` | P2 | S | `specs/vlm-client.md` |
| 1 | data — dataset access: windows, consecutive and random pairs | P1, P2, P5 | — | G, D |
| 2 | perception — $\phi$ and the P1 loss | P1 | 1 | PER |
| 3 | acyclic — correction $C$ | P3, R3 | — | ACY |
| 4 | vlm-labels — prompt, query, parsing, label artifact | P2 | 1 | VLM |
| 5 | preference — $P$ and its loss | P4, R3 | 2 | PRF |
| 6 | policy — $\pi$ and the P5 loss | P5 | 2 | POL |
| 7 | pretrain — orchestration of P1–P5 | P1–P5 | 1–6 | G |
| 8 | mcts — search, decision, tree storage | R1 | 5, 6 | MCTS |
| 9 | env-sampling — environment loop and trajectory record | R1 | 8 | RLS |
| 10 | metra — METRA perception loss | R2 | 2 | MET |
| 11 | memory — storage, nearest states, beam search | R3, R5 | 5 | MEM |
| 12 | ppo — policy update, with backward updates on past trees mixed in | R4 | 6, 8 | PPO, EBU |
| 13 | rl-cycle — orchestration of R1–R5 | R1–R5 | 8–12 | G |
| 14 | env — MineStudio wrapper: macro actions, masks, endless timeline | R1, recordings | 15 | ENV, ACT |
| 15 | gui-screens — catalogue of survival screens: window, textures, dynamic regions, elements with canonical numbering | P1, P5, R1 | S | ACT, GX |
| 16 | vpt-gui-frames — random VPT frames with an open GUI, fetched without whole videos; `download_data.sh` | GX experiment | S | GX |
| 17 | gui-labels — screen type of each sampled frame by the VLM | GX experiment | 0, 15, 16 | GX |
| 18 | gui-masks — six mask variants, template classifier, agreement with VLM labels, cross-fitted features; later rebuilt on engine frames; its variant (d) is the pipeline's classifier (block 28) | GX experiment, then 14 | 15, 16, 17 | GX |
| 19 | gui-stack — MLP over the mask features of block 18 | GX experiment | 18 | GX |
| 20 | vpt-actions — legal world actions of $A$ per interval of a VPT episode, modes, exclusions | P1, P2, P5 data | — | `specs/vpt-actions.md` |
| 21 | vpt-gui-actions — GUI lines of a VPT episode: clicks → elements (left, right, shift, double), close, digit chains, cut spans; classes from a pluggable classifier | P1, P2, P5 data | 15, 20 | `specs/vpt-gui-actions.md` |
| 22 | clips — grid of elementary states, interval validity, sampling of non-overlapping clips | P1, P2 data | 20, 21 | `specs/clips.md` |
| 23 | preference-pool — `collect_preferences_pretrain_data.sh`: clips of $k_P + k_\phi - 1$ states, pairs within one game world (not necessarily one episode), then block 4 adds the VLM labels | P2 | 22, 24, 25, 0 | VLM |
| 24 | f3-overlay — debug-overlay detector on frame 0 of an episode | P1, P2 data | 20 | `specs/f3-overlay.md` |
| 25 | perception-data — pipeline over whole VPT episodes (download, two decodes, classifier calls, sampling, storage, rerun); `collect_perception_pretrain_data.sh` with the classifier | P1 | 21, 22, 24, 28 | `specs/perception-data.md` |
| 26 | action-mask — the mask of $A$ per state, masking as much as possible (unsupported shift / double clicks, …) | P5, R1 | 21 | [U] 2026-10-15 |
| 27 | shards — on-disk shards of every dataset: atomic, durable, in order (AGENTS.md Code, Data) | all data stages | — | `specs/shards.md` |
| 28 | gui-classifier — the pipeline's screen classifier: variant (d) of block 18 from its templates and thresholds; latency budget | P1, P2 data, R1 (env) | 18 | GX |

## Package map
Filled as block specs are approved: subpackage → stage(s) → spec.
- `vlm` → P2 (later R-stage labelling) → `specs/vlm-client.md`
- `scripts/`, `pyproject.toml`, `uv.lock`, `configs/spark-env.yaml` → all → `specs/spark-env.md`
- `vpt` → GUI-recognition experiment, P1, P2, P5 data → `specs/vpt-gui-frames.md`, `specs/vpt-actions.md`
- `gui` → GUI-recognition experiment, env, human-click mapping, P1, P2, P5 data → `specs/gui-screens.md` (data: `configs/gui-screens.yaml`), `specs/gui-labels.md`, `specs/vpt-gui-actions.md`
- `recognition` → GUI-recognition experiment, P1, P2 data, later env → `specs/gui-masks.md`, `specs/f3-overlay.md`, `specs/gui-classifier.md`
- `clips` → P1, P2 data → `specs/clips.md`, `specs/perception-data.md`
- `stacking` → GUI-recognition experiment → `specs/gui-stack.md`
- `http` → shared HTTP retries (textures, VPT downloads, VLM client) → the retry rules of `specs/vlm-client.md` (6), `specs/gui-screens.md`, `specs/vpt-gui-frames.md`
- `cli` → shared one-line failure messages of the command-line entry points
- `shards` → every block that builds or reads a dataset → `specs/shards.md`
