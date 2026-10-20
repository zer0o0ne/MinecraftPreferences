# Open questions
Status: draft
Questions on `specs/00-overview.md`, by block. **[later]** — the user announced the answer. **→ [U]** — the user's answer (2026-10-09); it moves into the block spec that needs it, and the question leaves this file when that spec is approved. Every exact number is a config field and is not asked (G-7).
Facts about MineStudio: source at commit `278aa85` (v1.1.5), checked 2026-10-09.

## G — Global
- **G-1** Step granularity: one dataset / env step = one 20 Hz frame, or every $s$-th frame (frame skip in the dataset, action repeat in the env)?
  **→ [U]** one agent step = one action of $A$; its duration in ticks follows from the action's definition (ACT-2).
- **G-2** History lengths: perception input $k_\phi$, preference $k_P$, policy $k_\pi$, VLM screenshots $k_V$ — equal or separate, values, stride inside a window?
  **→ [U]** $\phi$ sees past frames at a stride over a span (example: every 3rd frame over the last 30) plus all meta of those frames. Open: exact stride and span; does the window end at the current frame; $k_P$, $k_\pi$, $k_V$.
  **→ [U]** (2026-10-20) $\phi$'s input is the sequence of elementary states as interleaved (screenshot$_i$, meta$_i$); in training whole clips go through the model; at sampling (R1) $\phi$ runs streaming (each step adds only the new state), and the stream is restarted every $L$ states, so a state sees 1..$L$ states of context, as inside a training clip. Measured: Qwen3.5-2B takes 222 tokens per 640 × 360 frame and ≈ 65–180 tokens per meta as text; its context is 262,144 tokens.
- **G-3** History at an episode start ($t < k$): pad (repeat first state / zeros) or exclude such $t$?
- **G-4** Is the VLM used only in pretraining, so that in RL all preference labels come from the current $P$?
- **G-5** Per cycle: R2 → R3 → R4 → R5 in this order, once each? Gradient steps / epochs for each.
- **G-6** Architectures: $\phi$ backbone (VPT pretrained / ImpalaCNN from scratch / pretrained vision encoder), $P$ and $\pi$ (MLP / transformer over latent histories), sizes, $D$.
  **→ [U]** (2026-10-20) $\phi$ = a small standard VLM (e.g. Qwen3.5-2B) over the interleaved states of G-2; its last hidden state goes to an MLP head that maps it to $z$. $P$ and $\pi$ are lightweight (MCTS-5). Open: the exact VLM, the head, $D$, $P$ and $\pi$.
- **G-7** Optimisers, learning rates, batch sizes per training stage; fixed in each block spec, or one config file with values from you?
  **→ [U]** every exact number goes into a config, never asked; where and how: G-11.
- **G-8** Checkpointed across cycles for resuming: nets, optimiser states, METRA $\lambda$, $\mathcal{M}$, $\mathcal{T}$, env state?
- **G-10** Python version for the project (no `pyproject.toml` yet; the first block creates it): 3.10 (MineStudio requires ≥ 3.10 and pins `opencv-python==4.8.0.74`, built for numpy 1.x → numpy 1.26.4) or 3.12 (numpy 2.x; only if MineStudio is dropped, D-7)?
  **→ [U]** 3.10.
- **G-9** With a frame stride $s$ in the window, is one agent / dataset step 1 frame or $s$ frames, i.e. is P1's consecutive pair $(t, t+1)$ or $(t, t+s)$? (Ties to G-1.)
  **→ [U]** see G-1.
- **G-11** Config: proposed `configs/<block>.yaml` at the repository root, loaded with PyYAML (already required by MineStudio via hydra-core) into a frozen dataclass per block, defined in the block's module; no defaults in code; secrets (API key) from environment variables. Needs `configs/` in the AGENTS.md Layout.
  **→ [U]** yes; `configs/` added to the AGENTS.md Layout.
- **G-12** Fixed time step. **[U]** (2026-10-15): human play rarely has states that differ by exactly one action of $A$, so states are taken a fixed $\Delta$ ticks apart (config); the inputs of $\phi$ and $P$ are sequences of states $\Delta$ apart; the agent will act once per $\Delta$; for P1 every interval stores $n$, the number of legal actions of $A$ the human made in it (for METRA's geometry); intervals with $n = 0$ are not used. Supersedes G-1, the tick stride of G-2 and "one action between consecutive states". Consequences: a pair of P1 spans $k_\phi + 1$ states; an input history of $P$ ($k_P$ latent states) needs $k_P + k_\phi - 1$ raw states.
- **G-13** What an agent action does during its $\Delta$ ticks (replaces ENV-2 (b) "until the effect completes"; fixes how human actions are counted in $n$): (a) its input is applied for the whole interval (keys / buttons held $\Delta$ ticks, a 10° turn or a click at its start); (b) held until its effect completes, at most $\Delta$ ticks, idle for the rest; long effects (breaking, eating) take several actions in a row in both.
  **→ [U]** (2026-10-15) (b), refined: an action holds its input until its effect completes, at most $\Delta$ ticks; a long effect takes several actions; when the next chosen action is the same, its input continues without release, so the effect is not cut at the boundary; it is released when another action is chosen.
- **G-14** Where $n = 0$ excludes data: (a) only the transition of a P1 pair; (b) every interval of a sequence (the $\phi$ window, the $P$ history), so that human sequences look like agent ones, where every interval holds an action.
  **→ [U]** (2026-10-15) (b): $\Delta$ is the time between elementary states; $\phi$ and $P$ are fed several consecutive elementary states (from $t$ to $t + k\Delta$); only sequences in which every $\Delta$-interval holds a positive number of actions are used.

## D — Data (facts below are for VPT via MineStudio; D-7 may replace it)
- **D-1** Which subsets: 6xx free play, 7xx early game, 8xx house from scratch, 9xx house from given materials, 10xx diamond pickaxe (together 15,655 episodes, 160M frames ≈ 2,229 h at 20 Hz)? Train / validation split (MineStudio splits by episode)?
- **D-2** Dataset meta per frame: yaw, pitch, position, selected hotbar slot, inventory (items and counts, no slot positions), GUI flags, per-frame stat increments (`mine_block`, `pickup`, `damage_taken`, …), cursor. Health, food and armor are absent (the simulator has them). (a) Which fields go to the VLM instead of health? (b) $o_t$ for $\phi$: frame only, or frame + meta (which fields, encoding)? In RL only fields present in both dataset and simulator?
  **→ [U]** (b): frames + all meta of the window's frames. Open: (a).
- **D-3** Dataset frames are stored as 224×224, squashed from 640×360. Use them as is for $\phi$ and the VLM, or decode the original VPT videos (at least for the VLM)? The simulator renders 640×360 and resizes to a chosen size, which must match.
  **→ [U]** original VPT videos at 640×360 (D-7 (a)); input size of $\phi$ is a config value.
- **D-4** MineStudio removed no-op frames from the dataset (no all-zero actions) and merges recorded segments across small gaps, so dataset neighbours are not always neighbouring game ticks; the simulator advances one tick per step, no-ops included. Accept dataset neighbours as consecutive (as said in P1), or exclude pairs across gaps? Does the policy see no-op as an action in RL? (Alignment fact: $a_t$ follows $o_t$, as in MineStudio's behaviour cloning.)
  **→ [U]** superseded: the original VPT files keep every tick; their segmentation into actions is ACT-3.
- **D-5** "Random consecutive pairs": uniform over all frames of all episodes, with or without replacement, how many?
- **D-6** Identical consecutive frames (standing still, open GUI): if $z_t = \phi(o_t)$, then $d_\alpha(z_t, z_{t+1}) = 0$ and "exactly 1" is unsatisfiable. Drop such pairs, use $\le 1$ (METRA), or let $\phi$ see a history window?
  **→ [U]** $\phi$ sees a window with meta (G-2). Windows identical over the whole span remain possible (standing still); handled by PER-3.
- **D-7** Dataset and recorder. **[U]**: VPT's resolution and action count are unsatisfactory; find alternatives. Research of 2026-10-09: no public dataset has human play at ≥ 360p with a small action set, health / food and a matching RL simulator. Options:
  (a) original VPT mp4 + jsonl at 640×360 (not MineStudio's 224×224 copy), plus BASALT 2022 (361 h, 640×360), mapped to $A$ procedurally (POL-1); RL and new human recordings in the MineStudio simulator (`PlayCallback` lets a human play the same env at 20 Hz, `RecordCallback` writes mp4 + actions + info incl. health / food). One engine (MC 1.16.5) for data, recording and RL.
  (b) own stack: Mineflayer bot + headless official client as camera (SolarisEngine pattern, MC 1.21, 1280×720), full meta, discrete control; the lockstep RL wrapper and the human recorder must be built; existing data: Solaris 12.6M bot frames only.
  (c) Craftium (Luanti, not Minecraft): synchronous, any resolution, discrete wrapper; no human data, no recorder.
  Rejected: MineRL v0 (64×64, MC 1.11), Matrix-Game-MC (not released), PLAICraft (no meta, no env, 200 h), GF-Minecraft (scripted random actions), MineDojo (no action labels).
  **→ [U]** (a); Python 3.10 (G-10).
- **D-8** Meta encoding for $\phi$: inventory (counts over a fixed item vocabulary?), position (absolute world coordinates differ between worlds: raw, relative to the window's last frame, or relative to episode start?), yaw / pitch (raw or sin / cos), GUI flags, which stat increments. Fixed after D-7.
- **D-9** Cursor: the raw VPT videos have no cursor (VPT's loader draws a 16×16 cursor sprite on GUI frames from the recorded position), while the MineStudio simulator renders one. Draw it onto VPT GUI frames for $\phi$, as VPT and MineStudio loaders do?
  **→ [U]** yes.

## PER — Perception pretraining (P1)
- **PER-1** Weighted distance; with weight $w_i$ per squared difference, one unit step along $i$ moves $1/\sqrt{w_i}$, so the norm bound 1 is reached after $\sqrt{w_i}$ straight steps or $w_i$ random-walk steps:
  (a) $d^2 = \sum_i \alpha_i (z_i - z'_i)^2$ — 512 → 23 straight / 512 random-walk steps;
  (b) $d^2 = \sum_i \alpha_i^2 (z_i - z'_i)^2$ — 512 → 512 straight / 512² random-walk steps;
  (c) $d = \sum_i \alpha_i |z_i - z'_i|$ — 512 → 512 straight steps.
  Which formula, and does "512·512 steps" count a random walk?
  **→ [U]** (b): $d_\alpha^2(z, z') = \sum_i \alpha_i^2 (z_i - z'_i)^2$. METRA geometry holds in the coordinates $\alpha \odot z$; the norm bound holds for $z$, the input of the other networks.
- **PER-2** $\alpha$: fixed hyperparameters (which values, e.g. a geometric ladder $1 \dots 512$) or learned (then how kept from collapsing)?
  **→ [U]** fixed; values in the config, chosen with the architecture.
- **PER-3** "Exactly 1". **[U]**: driven to 1 "as in METRA". Without the skill term METRA's loss only bounds $d_\alpha \le 1$ and nothing pulls $d_\alpha$ up. Which loss:
  (a) $\mathbb{E}[(d_\alpha^2 - 1)^2]$ (or $(d_\alpha - 1)^2$) with weight $w$; for windows that do not change, it pushes the impossible $0 \to 1$;
  (b) METRA dual with a pulling term: $\max \mathbb{E}[d_\alpha]$ s.t. $d_\alpha^2 \le 1$, i.e. $L = -\mathbb{E}[d_\alpha + \lambda \min(\varepsilon, 1 - d_\alpha^2)]$, $\lambda$ updated as in METRA;
  (c) other.
  **→ [U]** (b), maximising $d_\alpha$ as in METRA, only on transitions that changed something (frames or meta); actions are chosen so that they likely change something (ACT).
- **PER-8** "Changed": decoded video frames of a static scene still differ by compression noise. Proposed: changed ⇔ meta differs, or mean $|\Delta \text{pixel}|$ of the last frame > $\tau$ (config). Unchanged transitions: left out of P1 entirely, or kept in the constraint term only?
  **→ [U]** changed ⇔ meta differs or the pixel difference exceeds a config threshold; unchanged transitions are dropped from the pairs, "possibly in the edge-adding loop, so that the gaps are healed by other pairs at once". Open: what heals a gap in P1: (a) replacement pairs are drawn so that the pair count stays the same; (b) a run of unchanged steps is bridged by the pair (last state before the run, first changed state after it)?
- **PER-4** Norm penalty: plain $\lVert z \rVert_2$ or weighted; $\max(0, \lVert z\rVert - c)$ or its square; on both $z_t, z_{t+1}$; coefficient; $c$.
  **→ [U]** plain $\lVert z \rVert_2$, on both $z$ and $z'$, squared hinge $\beta_c \max(0, \lVert z \rVert_2 - c)^2$; $\beta_c$, $c$ in the config.
- **PER-9** With $n$ legal actions in the interval of a pair (G-12): constraint $d_\alpha(z_t, z_{t+\Delta}) \le n$, i.e. $d_\alpha^2 \le n^2$, or another form? Decided with the perception block.
- **PER-5** "Truncated loss" = METRA perception loss without $(\phi(s') - \phi(s))^\top z$, i.e. P1 loss = PER-3 + PER-4 only?
- **PER-6** Output layer: unbounded linear or bounded (tanh)?
- **PER-7** $\phi$ per frame ($z_t = \phi(o_t)$; "history" = sequence of per-frame $z$) or on a window $\phi(o_{t-k_\phi+1..t})$?
  **→ [U]** on a window (G-2).
  **→ [U]** (2026-10-20) the window is the stream since its last restart, at most $L$ states (G-2).

## VLM — Labels (P2)
- **VLM-1** [later] Labelling instruction. **→ [U]** model `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8`, served on a Spark; a Python API sends a set of images and text.
- **VLM-2** Pair sampling. With uniform random pairs over millions of frames each state is in ~1 pair, the graph has no cycles and $C$ changes nothing. Draw a pool of $N$ states and many pairs inside it? $N$, pairs per state; same episode or across episodes; minimum time gap.
  **→ [U]** random pool of states; each is linked at random to several others of the pool, with control of graph connectivity and of the presence of cycles. Open: $N$; degree; the exact procedure (e.g. a random $r$-regular graph, connected and full of cycles with high probability for $r \ge 3$, resampled until `is_connected`), what "presence of cycles" requires; pool across episodes; minimum time gap within an episode.
  **→ [U]** add random edges between components while there are many components and few cycles; thresholds in the config. Proposed measure: number of components and cyclomatic number $|E| - |V| + \#\text{components}$ (independent cycles).
- **VLM-3** Per state: which frames (count, stride, relation to $k_P$), which info fields as text, format.
- **VLM-4** Answer space: {first, second, refuse}, or also "equal"? Handling of unparseable answers.
- **VLM-5** Position bias: also query $(B, A)$? How are the two answers combined?
- **VLM-6** Edge weights for $C$: answer-token probabilities, votes over several samples, or constant 1?
- **VLM-7** Decoding: greedy or sampled (temperature)?

## ACY — Correction $C$ (P3, R3)
- **ACY-1** [later] The modifications.
- **ACY-2** Graph: nodes = states, edge $u \to v$ iff $u \succ v$, weight from VLM-6 (P3) / from $(p_\succ, p_{cmp})$ (R3: which function, threshold?). Incomparable pairs: no edge, and their $p_{cmp} = 0$ label kept as is?
- **ACY-3** Solver: exact (igraph `feedback_arc_set`, `algo="ip"`) or heuristic (Eades); expected graph size.
- **ACY-4** Output: kept edges → $p_\succ$ label 1 for $(u, v)$; removed edges dropped or reversed; transitive-closure pairs added?
- **ACY-5** Opposite edges for one pair ($u \to v$ and $v \to u$): merged before $C$ (net weight) or both given to $C$?

## PRF — Preference network (P4, R3)
- **PRF-1** Semantics: $p_\succ = P(x \succ y \mid \text{comparable})$, with its loss only on comparable pairs; or a joint probability?
- **PRF-2** Ties: label $p_\succ = 0.5$, or "incomparable", or absent (VLM-4)?
- **PRF-3** Symmetry $p_\succ(x, y) = 1 - p_\succ(y, x)$, $p_{cmp}(x, y) = p_{cmp}(y, x)$: by architecture, by training on both orders, or not imposed?
- **PRF-4** Loss: BCE on both heads; head weights; hard 0/1 or soft targets?
- **PRF-5** $\phi$ frozen (stop-grad) while $P$ trains, in P4 and R3?
- **PRF-6** Input history: $k_P$ latent states ending at the state itself, oldest first?

## POL — Policy (P5, R4)
- **POL-1** Action space. MineStudio's VPT agent action: 8641 button combinations × 121 camera bins (11×11 mu-law), ≈ 1.05M joint actions. No availability mask exists, and the simulator ignores `drop`. Full space or a reduced discrete set? What makes an action "available"?
  **→ [U]** tens of discrete actions: reduce procedurally or change dataset and recorder (D-7). Open: the action list; "available" = always all of them?
  **→ [U]** start from DreamerV3's 25 actions, possibly extended with our own; decided in section ACT.
  Procedures from the literature: frequency mapping (MineRL 2019 winner: 10 actions, frameskip 4; each 4-frame chunk of human data → its most frequent non-camera action + summed camera); k-means over human actions (30–150 clusters; rare important actions get lost at low counts, fixed by adding "necessary" actions); hand-designed functional set (DreamerV3: 25 actions incl. craft / place / equip / smelt macros). Which one, and does the stride of G-9 equal the action chunk?
- **POL-2** MCTS children and R2 need the shift of a given action, so $\widehat{\Delta z}$ must depend on it: (a) $\widehat{\Delta z} = g(z\text{-history}, a)$ with $a$ as input; (b) one $\widehat{\Delta z}$ output per action (only for a small action set). Which?
  **→ [U]** (b): output $|A| \times D$, one shift per action.
- **POL-3** $\widehat{\Delta z}$ deterministic (regression) or a distribution to sample from ("sampled … shift")? Loss: MSE (Euclidean or $d_\alpha$) or NLL.
- **POL-4** P5 loss: $\mathrm{CE}(a_t) + \beta L_\Delta$; $\beta$; CE over the factorised VPT action; target $\Delta z_t$ from frozen $\phi$; under (a) the shift head is fed the dataset action $a_t$?
- **POL-5** PPO usually needs a critic; the description has none. Add a value head, or advantages from returns without a learned baseline?
  **→ [U]** no new networks: PPO uses only $\phi$ and $P$. Details with PPO [later].
- **POL-6** Is $\widehat{\Delta z}$ trained in R4 too (regression to the actual $\Delta z_t$ under the new $\phi$)?
- **POL-7** **[U]** (2026-10-15): $\pi$ is trained to lower the logits of masked actions, which helps MCTS expand the most probable actions (imagined nodes have no frame to recognise, hence no mask). So every dataset state keeps what its mask needs (GUI open, recognised class, pitch). Open: the loss term, its weight (config), in P5 and R4.
- **POL-8** With $n \ge 1$ human actions per interval (G-12), the action target of P5: (a) only intervals with $n = 1$, their action; (b) the first action of the interval; (c) all its actions (multi-label). The data blocks keep the actions of every interval, so this can be decided with P5.

## MCTS (R1)
- **MCTS-1** Value of an imagined node: $P(\text{node}, \text{root})$, $P(\text{node}, \text{parent})$ summed along the path, …? Scalar from $(p_\succ, p_{cmp})$: e.g. $p_{cmp}(2p_\succ - 1)$, $p_\succ$, $\log p_\succ$?
- **MCTS-2** Selection rule: PUCT with prior $\pi(a \mid \cdot)$ (AlphaZero), $c_{puct}$?
- **MCTS-3** Expansion: which actions get children — all available, top-$K$ by prior, $K$ sampled, progressive widening?
  **→ [U]** the $l$ most probable unmasked actions ($l$ in the config).
- **MCTS-4** Backup: mean or max; discount $\gamma$.
- **MCTS-5** Budget: simulations per decision, maximum depth.
  **→ [U]** (2026-10-20) the agent acts every ≈ 300 ms (one $\Delta$, ≈ 6 ticks at 20 Hz); within it several dozen MCTS iterations, each one call of $P$ and $\pi$: 100–150 ms for 20–30 calls, the rest of the interval for $\phi$ (streaming, G-2) and the other per-step components; speed matters only at sampling. The simulator runs in parallel with the agent's computation, in real time at 20 Hz (its budget is the whole interval). Open: MCTS-10.
- **MCTS-10** With the simulator running in parallel, the action chosen during interval $t$ (from state $t$) takes effect only from $t + 1$, while action $a_t$ is still being executed. Plan from the root $z_t$ as if $a_t$ had not happened, or from the imagined state after it ($z_t + \widehat{\Delta z}_t(a_t)$, its shift predicted by $\pi$)? The human data of P5 has no such delay.
  **→ [U]** (2026-10-20) the human pretraining gets the same delay, simulating the human reaction time (the P300, from receiving information to the first reaction): in P5 the action target of state $t$ (history up to $t$) is the action of the next interval, from $t + 1$, so the agent and the human data share the one-interval latency. Open (POL, MCTS blocks): (a) the shift target of that action — $z_{t+2} - z_{t+1}$ (its own effect) or $z_{t+2} - z_t$ (including the in-flight $a_t$); (b) how the tree accounts for the in-flight $a_t$ at the root; (c) POL-8 still applies to the delayed interval; (d) the last interval of a clip has no delayed target.
  **→ [U]** (2026-10-20) (b) the in-flight $a_t$ is not accounted for: the tree is built from $z_t$; (c) yes, POL-8 is decided for the delayed interval; (d) yes, the last interval of a clip gives no P5 target. (a) asked again: the user's formula $z_{t+1} - z_t$ and the words "the shift with the delay" admit two readings.
  **→ [U]** (2026-10-20) (a): from the history up to $z_t$, $\pi$ predicts the action chosen at $z_t$ (executed from $t + 1$), and the shift head's row of that action gets the target $z_{t+1} - z_t$, though that change is caused by the action still in flight. Reason: a human deciding the action has only $z_t$ and models its consequences from there; the one-interval offset is absorbed in RL ($\pi$ and $\phi$ learn that they work with a delay), and $\pi$ sees several past latent states, so it can estimate the dynamics.
- **MCTS-6** Decision: most visited child, sample $\propto N^{1/\tau}$, or max value? Which probabilities are stored for PPO — the network's $\pi(\cdot)$ or the visit distribution? PPO's ratio needs the probability with which the executed action was actually chosen.
- **MCTS-7** [later] Tree reuse: how the real $z_{t+1}$ attaches (re-root at the executed child with its $z$ replaced, a new root linked to it, …); what "extended by future decisions" means structurally.
- **MCTS-8** Imagined nodes have no observation, so after R2 their $z$ are coordinates of the old $\phi$. Re-simulate them from re-encoded real nodes with the current $\pi$, keep them as is, or keep only real nodes?
  **→ [U]** keep as is; the networks must work acceptably in stale coordinates.
- **MCTS-9** History of an imagined node for $\pi$ and $P$ = real history + imagined path?

## RLS — Sampling (R1)
- **RLS-1** Environment: `MinecraftSim` is synchronous (the world advances only on `step()`, one tick per step), so the pause is free. Settings: seed, spawn biome, initial inventory, observation size. Death does not end the episode (respawn behaviour unconfirmed); on a time-limit or another `done`, reset (a new world, breaking the single timeline) or never end?
- **RLS-2** $n$ per cycle.
- **RLS-3** "Preference of the start point over the action's result": $P(z_t, z_{t+1})$ (state before the action), or $P(z_0, z_{t+1})$ (trajectory start)?
  **→ [U]** both kinds are useful: between close points and over the whole episode or its large parts. Exact math with PPO [later]; it also fixes what R1 stores.
- **RLS-4** Stored $\Delta z_t$ uses $\phi$ at sampling time; stored $\widehat{\Delta z}_t$ is for the executed action?

## MET — Perception fine-tuning (R2)
- **MET-1** METRA (arXiv:2310.08887) dual form $L_\phi = -\mathbb{E}[(\phi(s') - \phi(s))^\top z + \lambda \min(\varepsilon, 1 - \lVert \phi(s') - \phi(s) \rVert^2)]$ with $z := \widehat{\Delta z}_t$. Constraint with plain Euclidean norm (METRA) or $d_\alpha$ (P1); $\le 1$ (METRA) or $= 1$ (P1)? Given PER-1, in the coordinates $\alpha \odot z$: $(\alpha \odot (\phi(s') - \phi(s)))^\top (\alpha \odot \widehat{\Delta z})$ with $d_\alpha^2 \le 1$, or another combination?
  **→ [U]** yes: $(\alpha \odot (\phi(s') - \phi(s)))^\top (\alpha \odot \widehat{\Delta z})$ with $d_\alpha^2 \le 1$.
- **MET-2** $\widehat{\Delta z}_t$ raw or normalised to unit length (METRA normalises continuous skills — to verify); the one stored at sampling or recomputed with the current $\pi$ and $\phi$; stop-grad.
- **MET-3** P1 norm penalty kept in R2?
- **MET-4** $\lambda$: initial value, slack $\varepsilon$, log-parametrisation, learning rate (METRA code: 30, $10^{-3}$ — to verify); persists across cycles?
- **MET-5** Data: only the current trajectory's $n$ transitions, or also transitions from $\mathcal{M}$? Epochs, minibatches.

## MEM — Memory and pair mining (R3, R5)
- **MEM-1** [later] Memory update rule, capacity, eviction.
- **MEM-2** Stored per state: the observation(s) needed to rebuild its history and re-encode it; trees; past labels? Is the VPT dataset (or the P2 pool) in $\mathcal{M}$ at the start of RL?
- **MEM-3** After R2: re-encode all of $\mathcal{M}$, or only candidates during the search?
- **MEM-4** "Nearest": metric (Euclidean or $d_\alpha$; on $z$ or on histories), count $K$; exclude states of the current trajectory near $t$?
- **MEM-5** [later] Beam search: the relation used for "most / least preferred" (current $P$ on the fly, stored labels, corrected graph), beam width, depth, both directions?
- **MEM-6** Number $m$ of picked trajectory states. Pairs only (picked state, found state), or also among found states (graph density for $C$)?

## PPO (R4)
- **PPO-1** [later] (with RLS-3) Reward from the stored preferences, e.g. $r_t = p_{cmp}(1 - 2p_\succ)$ for $P(z_t, z_{t+1})$ (result preferred to start = positive). Values stored at sampling, or recomputed with the new $\phi$ and $P$?
- **PPO-2** Advantage estimator (GAE with $\gamma, \lambda$ / Monte-Carlo returns), normalisation; critic (POL-5).
- **PPO-3** Clip $\epsilon$, epochs, minibatch size, entropy and KL coefficients; ratio denominator (MCTS-6).
- **PPO-4** Policy inputs encoded with the $\phi$ from R2, i.e. new coordinates for both heads?

## EBU — Retrospective training on past trees (R4)
- **EBU-1** [later] Details of the adaptation of Episodic Backward Update (Lee, Sung, Choi, NeurIPS 2019, arXiv:1805.12375).
- **EBU-2** EBU learns a Q-function: it samples a whole past episode and computes targets backward, $\tilde Q[A_{k+1}, k] \leftarrow \beta y_{k+1} + (1 - \beta)\tilde Q[A_{k+1}, k]$, $y_k = r_k + \gamma \max_a \tilde Q[a, k]$. Here: which network learns from these targets (a Q / value head, the action head via a policy target, $\widehat{\Delta z}$)? Do the values of non-executed actions $\tilde Q[a, k]$ come from the imagined children in the stored tree?
- **EBU-3** Unit of a backward pass: one past cycle's trajectory, a path through several trees, the whole history since a point?
- **EBU-4** Rewards $r_k$ on the old path: stored, or recomputed with the current $\phi$ and $P$? Old imagined values: recomputed (MCTS-8)?
- **EBU-5** Diffusion factor $\beta$, discount $\gamma$; which past trees are picked (uniform, recency, priority) and how many per cycle.
- **EBU-6** Mixing with the fresh trajectory: one combined loss with weight / ratio, or separate gradient steps? PPO is on-policy and EBU off-policy: old samples go into the PPO objective (which importance correction?) or only into the EBU loss?

## VLMC — VLM client (`specs/vlm-client.md`)
- **VLMC-1** Prompt layout: (a) an ordered sequence of text and image parts, e.g. "State A:" img … "State B:" img … instruction (draft); (b) all images first, then one text.
- **VLMC-2** System message: none, the instruction goes into the user message (draft); or a fixed system text.
- **VLMC-3** Sampling: Qwen README values for Thinking models, T 0.6, top_p 0.95, top_k 20 (draft); `generation_config.json`, T 0.8; greedy (Qwen advises against it for thinking models: repetition loops). Ties to VLM-7.
- **VLMC-4** (→ config) Output limit and attempt bound: (a) `max_tokens` 40960 (Qwen), $T_{out}$ = 3600 s; (b) 16384, $T_{out}$ = 1800 s. Estimate (unverified): ~300 tok/s total at 16 streams ≈ 19 tok/s per request, so 40960 tokens ≈ 36 min.
  **→ [U]** answer limited to 300 tokens; with it, Thinking vs Instruct: VLMC-14.
- **VLMC-5** (→ config) Image size: `min_pixels` ≤ frame area ≤ `max_pixels` keeps frames at native size (640×360 → 640×352, 220 tokens). Values follow D-3 / D-7.
- **VLMC-6** Failures: thinking cut by the limit, empty answer, attempt over $T_{out}$ → `None` for that prompt; transport / 5xx after $R$ resends, or any 4xx → the whole call raises (draft). Alternative: every failure → `None`.
- **VLMC-7** Seed per request from the caller (draft), for reproducible sampling up to batch effects.
- **VLMC-8** Answer-token probabilities and constrained choice: later, together with VLM-6 (draft), or now.
- **VLMC-9** (→ config) $K$ = 16, $R$ = 3, $\delta$ = 10 s.
- **VLMC-10** Transport: `httpx` (every field explicit, no hidden retries; draft) or the `openai` SDK (vLLM fields via `extra_body`; its default 10 min timeout and 2 retries overridden).
- **VLMC-11** Connection: `base_url` and `api_key` as arguments; SSH tunnel to the Spark, port bound to localhost there (draft).
- **VLMC-12** (→ config) $N_{img}$ and $P_{max}$: proposed 32 images and 8192 tokens (20 frames × 222 tokens + 2K text fit), so `--max-model-len` = `max_tokens` + 8192. Follows VLM-3.
- **VLMC-13** "Spark" = NVIDIA DGX Spark? vLLM v0.31.0 was released 2026-10-05; the Instruct-FP8 sibling is reported working on Spark, this Thinking-FP8 checkpoint on v0.29–0.31 is not; the smoke test (Server, steps 5–6) is the check.
  **→ [U]** NVIDIA DGX Spark.
- **VLMC-14** With a 300-token limit: (a) Thinking model with a forced thinking budget (vLLM `thinking_token_budget` + `--reasoning-config`): the model is trained to think for thousands of tokens, so a cut at 300 is out of its training distribution; (b) `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` @ `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`, prompted for a short justification and then the answer: in distribution, its answer-token probability is usable as an edge weight (VLM-6), and this checkpoint is reported working on DGX Spark. Recommended: (b).
  **→ [U]** (b) Instruct.

## ACT — Action set $A$
**[U]** (2026-10-09), version 1:
- world (no GUI open), 18 actions: turn up / down / left / right by 10°; forward, back, left, right; jump, jump forward, jump back, jump left, jump right; right click (use); left click (attack); hotbar slot right / left with wrap-around; open inventory. The game's Auto-Jump option is on. No sneak, sprint, drop, offhand swap, chat, F-keys (first-person view only).
- GUI open (inventory or any screen): close the GUI; left click on cell $c$; right click on cell $c$; $c = 1..N_{cell}$. The screen is covered by a grid of equal square cells, numbered row-major; any GUI element (slot or button) is clicked through a cell, so no GUI type is needed. Text input is ignored. No shift-click.
- masking: no GUI open → close and cell clicks masked; GUI open → all world actions masked (movement, jumps, turns, both clicks, hotbar shifts, open inventory); turn up at pitch −90° and turn down at +90° masked.
- no action has a waiting time from the agent's view: blocks break almost at once, food is eaten at once, a bow is drawn at once, …; shields ignored (ENV-2).
- MCTS expands only the $l$ most probable unmasked actions ($l$ in the config).
- **[U]** (2026-10-15), version 2 of the world actions: plus four diagonal moves (forward-left, forward-right, back-left, back-right) and four diagonal jumps, 26 world actions in all; a key combination of the human data maps to one action (W + A = forward-left, W + A + jump = jump forward-left). Pause is never an action; it may later be switched on and off procedurally.
- **[U]** (2026-10-15, later): the agent gains `sneak` (Shift), e.g. to leave a boat or a horse; 27 world actions. Riding intervals of the human data are excluded (`specs/vpt-actions.md`).
- **[U]** (2026-10-15, block vpt-gui-actions): GUI actions gain shift + left click on element $i$ (90 more); the classifier is pluggable (screen classes are an input); a span whose first frame is `other` or `pause` is cut (ACT-20), an `other` frame inside a kept span excludes its state; a click on no element, a drag across elements, and Q, F in a GUI exclude the interval; E and ESC in a GUI are `close`; the middle button is ignored; spans opened without an action (death screen) are kept. Keys 1–9: see below.
- **[U]** (2026-10-15, keys 1–9): in the world, `hotbar_1` … `hotbar_9` select a slot and replace the scroll actions `hotbar_left`, `hotbar_right` (34 world actions); every hotbar change of the human data (wheel, digit, middle button) is one `hotbar_<new slot>`. In a GUI the agent has no digit actions; a digit pressed by the human in a GUI (swap of the hovered slot with a hotbar slot) is replaced by the shortest equivalent chain of clicks, counted as 3 clicks (both slots occupied, the general case; slot contents are not in the recordings).
- **[U]** (2026-10-15, data blocks): the data unit is a clip of $L$ consecutive elementary states with every interval valid; episodes uniformly over the subsets, clip starts uniformly among the valid ones, at most $m$ non-overlapping clips per episode, $N$ in all; the train / validation split belongs to the training blocks; a clip stores its state frames (raw yuv420p), all `.jsonl` lines of its span, the options file, per interval $n$ and the actions, per state the mask inputs (GUI open, class, pitch); each episode's video is downloaded whole and decoded locally; the classifier is pluggable (`main` and the script are wired when it is chosen); block perception-data is split into `clips` (grid, validity, sampling) and `perception-data` (pipeline, storage, script); the F3 detector is its own block.
- **[U]** (2026-10-15): a double click (a second left press on the same element within 250 ms, which collects every matching item) is a new action `double_click_`$i$ (90 more; 361 GUI actions, 395 in all), the pair counted as one action; the F3 detector reads the prefix `Minecraft 1.16.5 (`; the classifier is called on batches of frames.
- **[U]** (2026-10-15, after the audits): actions are masked as much as possible — e.g. `shift_click_`$i$ and `double_click_`$i$ on screens that do not support them (future block action-mask, before P5; its inputs are kept by the data blocks); shift + click maps to `shift_click_`$i$ on every element (the game decides the effect); the first line of a container span is a world line; a key held through a cut span is one segment; the remaining audit points of blocks vpt-gui-actions, clips, perception-data, f3-overlay as recommended. Preference pairs (P2, block preference-pool) are drawn among clips of the same game world, not necessarily of the same episode (how VPT episodes map to worlds: to research).
- **[U]** (2026-10-15): blocks 20–22, 24, 25 are committed as approved; block perception-data's storage (appended `.jsonl` records, the plan replayed instead of a stored RNG state) is revised afterwards under the AGENTS.md Data rule, on a shared shards mechanism (a block of its own, proposed by the session working on vpt-gui-frames). The revision spells out perception-data's own rules where it now says "as in vpt-gui-frames" (run.json, the rerun rule (d), the `indices` bar), since the vpt-gui-frames revision changes them there; the shared code keeps perception-data's current behaviour meanwhile.
Open:
- **ACT-16** Grid. Facts (Minecraft 1.16.5 source via MineRL's MCP-Reborn, VPT recordings checked on samples): VPT played at 1280×720 with GUI scale 2, downsampled ×½; MineStudio renders 640×360 at GUI scale 1; so both use the same frame, 1 GUI px = 1 px of the 640×360 frame, identical window origins (e.g. (232, 97) for 176×166 screens); a human click is at GUI (x/2, y/2) of the recorded mouse position. Slot hit areas are 18×18 and tile with no gap; widgets are half-open rectangles. Smallest click targets: recipe-book page arrows 12×17; smaller ones are drag-only scrollbars (merchant 6 px, stonecutter / loom 12 px wide) or text fields (ignored); clicks cannot drag, so list entries beyond the first page of stonecutter, loom and villager trades stay unreachable. Cells tiling 640×360 have a side dividing 40: side 10 (64×36 = 2304 cells) or 8 (80×45) reach every click target; side 20 misses 261. No grid with side ≥ 2 puts all slot borders on cell borders, even within one screen. Human click → cell: (a) the cell containing the click; near a slot border its centre can lie in the neighbouring slot (upper bound for uniformly spread clicks: ~26 % of clicks at side 10, ~13 % at side 5; real clicks cluster at slot centres); (b) data preprocessing only: recognise the open screen from the frame (fixed origins and textures make template matching exact), find the clicked element, and label the click with the cell whose centre lies in that element and is nearest to the click; the agent is unchanged. Which?
  **→ [U]** recognise known GUIs by template comparison; back from cells to a fixed number of GUI elements (slots and buttons): GUI actions are left / right click on element $i$, $i = 1..n_{max}$, the maximum element count over all screens; clicks in human data are mapped to elements after recognition. The comparison masks are the key part and need experiments (block gui-recognition).
- **ACT-19** Consequences to confirm: (1) the simulator does not report the screen type either, so the same recogniser runs on simulator frames, to mask elements absent from the open screen and to know where to click element $i$; (2) element numbering canonical as in ACT-12 (the player's 36 slots at the same indices on every screen); $n_{max}$ ≈ 90 (large chest), $|A|$ ≈ 19 + 2 × 90; (3) a screen no template matches (advancements, books, signs, recipe pop-ups, …) allows only "close", and human segments on such screens map to no action.
  **→ [U]** all three confirmed.
  **→ [U]** (2026-10-15) (3) changed: a screen the recogniser does not recognise is closed by the environment at once, as part of the action that opened it; the agent is not run to close it (the same rule applies to the VPT data, ACT-20).
- **ACT-20** Unrecognised screens in the VPT data with states $\Delta$ apart (G-12): (a) their spans are cut out of the timeline: the action that opened the screen is followed by the first tick after it closed, and $\Delta$ is counted over the remaining ticks (the world moved meanwhile; inputs inside the span are ignored); (b) every sequence with a state or interval touching such a span is dropped. A known screen misrecognised as "other" is treated the same way.
  **→ [U]** (2026-10-15) (a).
- **ACT-17** Is the plain jump kept next to the four directional jumps? (Read as yes.)
- **ACT-18** Output size: with side 10 over the whole screen, $|A|$ = 19 + 2 × 2304 = 4627; the one-shift-per-action head (POL-2 (b)) is then $4627 \times D$ outputs. Keep it, restrict the grid to the area where GUI elements exist (fewer cells; clicks outside a window, which drop the held stack, become impossible), or give GUI clicks an action-conditioned shift $g(z\text{-history}, a)$ with the cell position and click type as input (POL-2 (a)) while world actions keep (b)?
  **→ [U]** resolved by ACT-16: $n_{max}$ elements instead of cells; POL-2 (b) stays.

## ENV — Environment (MineStudio simulator)
- **ENV-1** **[U]** DreamerV3-like mechanics are wanted; research restarted 2026-10-09 (ENV-2..6).
Facts (MineStudio @ 278aa85 and its engine jar `mcprec-6.13.jar` from `CraftJarvis/SimulatorEngine`, read as bytecode; the jar's source is not published; nothing run, this machine has no Java):
- breaking progress per tick = digSpeed / hardness / (30 if harvestable else 100); a block whose first-tick progress ≥ 1 breaks on the click, otherwise each break is followed by a 5-tick delay. Haste multiplies digSpeed by 1 + 0.2(a+1); the client sees the amplifier as a signed byte, so a ≤ 127 (×26.6), which also removes the attack cooldown. Eating (32 ticks, 16 for fast foods) and bow draw (20 ticks) have no vanilla speed-up; changing them needs a patched engine (rebuild MineRL 1.0's MCP-Reborn and port MineStudio's own changes).
- completion signals in `info`: block broken → `mine_block[<id>]` increments (≥ 1 step lag); item consumed → `use_item[<item>]` increments, inventory count drops, `food_level` rises; bow drawn → no signal (count 20 ticks); entity hit → `custom.damage_dealt` grows; attacking air or bedrock → nothing (needs a timeout).
- Auto-Jump: no API; `options.txt` with `autoJump:true` in `$MINESTUDIO_DIR/runtime/` is copied into every instance (untested).
- hotbar: selected slot not in `info` (track it; `hotbar.k` keys are absolute); the mouse wheel exists in the engine but needs a `MouseWheelAction` handler.
- GUI: camera degrees move the cursor, 1° = 6.67 GUI px at any GUI scale; the cursor resets to the screen centre when a GUI opens or closes and is not in `info`; a click takes 2 steps (press, release); ESC closes any GUI once a key handler is added; the inventory key closes container screens. MineStudio renders 640×360 at GUI scale 1.
- `info` per tick: GUI-open flag (no type), position + pitch + yaw, inventory slots 0–35 with type and count, equipped items, health, food level, stat dictionaries; saturation and air need a custom handler.
- death sets `done` and the Python env refuses steps until reset (clearing `done` flags is an untested workaround); `doImmediateRespawn` skips the death screen; no time limit by default; commands via `execute_cmd('/…')`; `sendCommandFeedback false` hides command output; the world is created in survival, difficulty hard.
- **ENV-2** "No waiting time" (ACT): (a) game rules: Haste ≤ 127 makes most blocks break on the click (×26.6), eating and bow need a patched engine; (b) one action holds its button until the completion signal above or a tick limit; world time passes meanwhile, the agent sees only the result; matches the human recordings; (c) both: Haste for breaking, holding for eating and bow. Which?
  **→ [U]** (b): an action runs until its effect completes (or a tick limit), then the agent picks the next one; it cannot be interrupted. No game-rule changes.
- **ENV-3** Action execution: hotbar shift via `hotbar.k` from a tracked slot, or add the wheel handler; a cell click = cursor move from the tracked cursor position + press + release (2 steps inside one action); close = ESC (handler to add).
- **ENV-4** GUI geometry → $N_{cell}$ and the cell side: researched, see ACT-16.
- **ENV-6** One endless timeline: gamerules `doImmediateRespawn true`, `sendCommandFeedback false`; difficulty; `keepInventory`; `doDaylightCycle`; keep stepping after a death (workaround to verify on the real simulator).
- **ENV-7** No Java on this machine: simulator code can be tested here only against a fake env; real runs happen on your GPU machine.

## GX — GUI-recognition experiment
**[U]** (2026-10-10): several thousand random VPT frames with an open GUI, without downloading whole videos; the screen type of each labelled by the VLM; all three mask variants (texture outline — only borders, so items in slots do not spoil it; low variance over real frames; discriminative patches) built and compared by agreement with the VLM labels; the best construction is later rebuilt on frames of the real engine (OptiFine etc. may render differently), assuming the construction method transfers. Code prepared here, run by the user on the DGX Spark. Proposed blocks: gui-screens (screen catalogue), vpt-gui-frames (frame sampling), gui-labels (VLM labelling), gui-masks (three variants, classifier, evaluation); plus vlm-client.
- **GX-1** Sampling: episodes uniformly over the five VPT subsets, then frames uniformly among an episode's GUI frames, at most $m$ per episode. Exactly uniform over all GUI frames of the dataset would need every action file (~15k × 11 MB).
- **GX-2** Classes = distinct element layouts (screens with identical layouts merged: barrel = 3-row chest, dropper = dispenser, …; recipe book open / closed separate, chest rows 1–6 separate), plus "other"; or one class per game screen?
- **GX-3** VLM input: the frame alone with the class list and a one-line visual description per class; VPT's `isGuiInventory` flag kept only as an independent check.
- **GX-4** Mask variants: (a) template = GUI texture at the known window origin, mask = its opaque pixels minus slot / button interiors, text labels and animated widgets (progress arrows, flames, enchanting text); (b) template = per-pixel median of a class's frames, mask = pixels whose spread over those frames is below a threshold; (c) template as (b), mask = the $P$ patches per class with the largest ratio of difference to other classes over within-class spread. Common classifier: score = share of mask pixels within a tolerance of the template; the best class wins if its score passes a threshold and beats the second by a margin, else "other".
- **GX-5** Evaluation: thresholds chosen on a validation split, agreement with VLM labels on a held-out split: per-class precision / recall, confusion matrix; (b) and (c) built on the training split only.
- **GX-6** Everything runs on the Spark (aarch64, Python 3.10, uv): the project's dependencies stay installable there; MineStudio joins later, with the env block.
- **GX-7** GUI textures from the official Minecraft 1.16.5 client jar (Mojang's download), extracted into `data/` at run time, local use only.
  **→ [U]** GX-1…GX-7 all agreed (2026-10-10); every Spark action is a bash script (`setup_env.sh`, `download_data.sh`, `serve_vlm.sh`). Specs: `spark-env.md`, `vlm-client.md`, `vpt-gui-frames.md` (approved), `gui-screens.md` (draft).
