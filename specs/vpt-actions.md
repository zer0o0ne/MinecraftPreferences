# VPT actions
Status: approved (2026-10-15)
Stage: data of P1, P2 (blocks perception-data, preference-pool), later P5 (`specs/00-overview.md`, block 20)

## Sources
- [U] the user, 2026-10-15 (Q G-12, G-13, G-14, ACT version 2): states a fixed $\Delta$ ticks apart; every interval between two states carries $n$, the number of legal actions of $A$ the human made in it; an action holds its input until its effect completes, at most $\Delta$, and continues without release while it is chosen again; $n$ counts, per channel, the action segments touching the interval — a segment is a continuous hold of a key or button, split after every completed effect and ended by an open GUI; a segment spanning several intervals counts once in each; camera turns are counted from the net yaw and pitch change of the interval, rounded to whole turns; four diagonal moves and four diagonal jumps join $A$, and a key combination of the human data is one action; sprint and sneak are ignored (sneak superseded below); the debug overlay (F3), third person (F5) and hidden HUD (F1) exclude what they touch; dropping (Q) and swapping to the off hand (F) exclude their interval; the middle mouse button counts through the hotbar change it causes; GUI lines belong to block vpt-gui-actions.
- [U] the user, 2026-10-15 (later): yaw changes wrapped per line (7); smoothing of jitter and short pieces (3), (4); modes at an episode start: all off when it starts a session, otherwise the overlay read from frame 0 and the HUD and view off; exclusions drop whole windows and histories (12); the agent gains a sneak action (Shift), and the intervals with a growing riding counter are excluded (11); F3 combos ignored; sneak in the human data is its own channel.
- `specs/vpt-gui-frames.md` (approved): `.jsonl` lines, one per tick; line $i$ ↔ frame $i$; frame count = line count + 1; options files; recorder versions ≥ 6.12.
- [R3] Measurements of 2026-10-15 on 10 random episodes of the five subsets (53,764 lines; scratchpad `vpt_eps/`, `vpt_frames/`, checked again by the spec audit):
  - a line has `mouse` {`x`, `y`, `dx`, `dy`, `scaledX`, `scaledY`, `dwheel`, `buttons`, `newButtons`}, `keyboard` {`keys`, `newKeys`, `chars`}, `isGuiOpen`, `isGuiInventory`, `hotbar` (0–8), `yaw`, `pitch`, `xpos`, `ypos`, `zpos`, `tick`, `milli`, `inventory` (list of {`type`, `quantity`}, no slot positions; empty in 4 % of lines), `serverTick`, `serverTickDurationMs`, `stats` (cumulative counters since the session start, e.g. `minecraft.custom:minecraft.jump`, `minecraft.mine_block:minecraft.stone`, `minecraft.use_item:minecraft.oak_planks`, `minecraft.custom:minecraft.damage_dealt`);
  - `keys` and `buttons` are the keys (buttons) held at the previous record plus every key (button) pressed since; `newKeys`, `newButtons` the presses since the previous record, OS auto-repeat events of a held key included; `newKeys` ⊆ `keys` and `newButtons` ⊆ `buttons` on every line ([G] `KeyboardListener.java:254-262, 475-481`, `MouseHelper.java:56-62, 364-371`);
  - the input of line $k$ moves the state from line $k$ to line $k+1$: $\text{yaw}_{k+1} - \text{yaw}_k$ correlates 1.00 with `dx`$_k$ (0.62–0.84 with `dx`$_{k \pm 1}$), outside a GUI; a mouse move right increases yaw, a move down increases pitch; yaw accumulates without wrapping (−2149 … 3007), except three one-line jumps by whole turns (2962.50 → 82.35 and −1701.52 → −258.67 when sleeping in a bed, −1475.63 → −34.43 with `dx` = −54, unexplained);
  - effects appear in the stats of a later line, by 0–5 ticks: a jump 0 ticks after the last line with space held in 331 of 536 jumps, 1–5 in 198, no space in the previous 50 lines in 4; blocks broken (997) with the left button held in the line before in 968; damage dealt 4–5 ticks after the left-button press (8 hits); `use_item` 0–4 ticks after the right button was last held, and also when a block is mined or a mob hit with a tool; a container opens 4 ticks after the right-button press;
  - the inventory key `e` pressed in a line without a GUI: `isGuiOpen` and `isGuiInventory` true in the next line (35 of 35); 70 GUI openings: 35 by `e`, 31 by the right button, 4 by chat (`t`, `/`);
  - hotbar changes (280): 274 with `dwheel` ≠ 0 or a digit key in `newKeys` of the line before the change; cyclic jumps of 1 (236), 2, 3, 4 slots;
  - all options files have the default key bindings, `autoJump` false, `toggleCrouch` and `toggleSprint` false;
  - episodes are pieces of up to 6000 lines (5 min) of longer sessions; 3 of 10 start a session (`stats` empty in lines 0–4), the others start with keys already held;
  - shares of lines without a GUI with the key in `keys`: `w` 32.7 %, `left.control` (sprint) 14.7 %, `space` 10.3 %, `d` 7.7 %, `a` 6.8 %, `s` 6.0 %, `left.shift` (sneak) 4.6 %, `f3` 0.35 %; presses of `f1`, `f5`, `q` none, `f` once (in a GUI); F3 combos: `a` twice; no riding counter grows;
  - F3 in 4 of 10 episodes, mostly as short pairs (overlay on for 1–3 s); on the frames of `cheeky-cornflower-setter-02e496ce4abb-20220421-092639`: `f3` in `keys` of lines 950–953, the overlay absent in frames 948–954 and shown from frame 955; in lines 966–969 again, shown up to frame 970, absent from frame 971;
  - runs of an unchanged movement-key combination (W/A/S/D/space): 2,920; 12 % last 1 line, 11 % 2 lines; 299 runs of 1–2 lines lie between two other combinations, 56 of the 136 one-line ones holding exactly the union of their neighbours' keys (a key pressed or released a tick apart); 66 of 152 attack runs with breaks end with a 1–3-line piece after the last break; 72 of 291 use splits and 37 of 275 jump splits fall on the first two lines of a run that began at most 6 lines after the previous run of the channel ended.
- [G] MCP-Reborn of `specs/gui-screens.md` [G], `src/main/java/net/minecraft/`: `util/MovementInputFromOptions.java:20-22` (forward = 0 when W and S are both held or both free, likewise strafe; jump = space held); `client/KeyboardListener.java:73-175` (`processKeyF3`: true for A, B, C, D, F, G, H, I, N, P, Q, T, F4 with these options), `:333-345` (F3 release when no screen is open or the screen passes events: clears the combo flag if set, else toggles the overlay), `:350-361` (with no screen open: a key press with F3 held sets the combo flag when `processKeyF3` is true; F1 toggles the hidden HUD), `:365-366` (a combo key is not passed to the key bindings); `client/gui/screen/inventory/InventoryScreen.java:36` (the player inventory passes events; chest and horse screens do not, the others keep the default, no); `client/Minecraft.java:1449-1451, 1538-1545` (key bindings, F5 and `e` included, are processed when no screen is open or it passes events; F5 cycles three points of view); `client/renderer/GameRenderer.java:468-471`, `client/gui/IngameGui.java:228-229` (the overlay is drawn on GUI frames too).

## Notation
- Lines $\ell_0, \dots, \ell_{T-1}$ of one episode (parsed JSON objects); frame $k$ belongs to line $k$ (the state of line $k$); the input of $\ell_k$ acts between frames $k$ and $k+1$ [R3].
- $W_k$ — $\ell_k$ is a world line: `isGuiOpen` false. $G_k$ — $\ell_k$ is in a screen that passes events: $W_k$ or `isGuiInventory` true.
- $K_k$ = `keyboard.keys`, $N_k$ = `keyboard.newKeys`, $B_k$ = `mouse.buttons` of $\ell_k$; key names are written without the prefix `key.keyboard.` (`w` = `key.keyboard.w`).
- $h_k$ = `hotbar`; $\psi_k$ = `yaw`, $\theta_k$ = `pitch` (float64); $\sigma_k[x]$ = `stats[x]`, 0 when absent.
- $f_k, r_k, j_k$ — forward, strafe and jump inputs of line $k$ (1); $v^g_k$ — the value of channel $g \in \{\text{move}, \text{sneak}, \text{attack}, \text{use}\}$ on line $k$, none or an action name (1), after smoothing for move (3); $e^g_k$ — an effect of channel $g$ between $\ell_k$ and $\ell_{k+1}$ (2).
- Run — a maximal set of consecutive lines with one non-none value of a channel; piece — a part of a run between splits (4); segment $s$ = (action, first line, last line), inclusive; $S$ — the segments of an episode.
- $a < b$ — the lines of two consecutive states of the consumer's grid ($b - a$ = $\Delta$ plus the cut lines between them); $I \subseteq \{a, \dots, b-1\}$ — the input lines of their interval, without the lines the consumer cuts (Q ACT-20); $S_I$ — the segments with a line in $I$; $n_\psi$, $n_\theta$ — the yaw and pitch turns of the interval; $n_{world}(I)$ — the count of its world actions.
- $\rho$ — degrees per turn (`turn_deg`); $r_{min}$ — the shortest piece and run kept apart (`r_min`).
- $C_3$ — the F3 combo keys; $\eta$ — the combo flag; $O_t$ debug overlay, $H_t$ hidden HUD (booleans), $V_t \in \{0, 1, 2\}$ point of view (0 = first person), per frame $t$; $\Omega_t$ — frame $t$ shows a mode.
- Config `configs/vpt-actions.yaml` → `@dataclass(frozen=True) class ActionsConfig`: `turn_deg` (a YAML integer or float, read with `float()`, > 0), `r_min` (a YAML integer, not a boolean, ≥ 1); violated constraints → `ValueError` at load.

## Math
(1) Channel values; none on every line without $W_k$:
- move: $f_k = [\texttt{w} \in K_k] - [\texttt{s} \in K_k]$, $r_k = [\texttt{d} \in K_k] - [\texttt{a} \in K_k]$, $j_k = [\texttt{space} \in K_k]$ [G]; none if $(f_k, r_k, j_k) = (0, 0, 0)$, else the name of $(f_k, r_k)$ — $(1,0)$ `forward`, $(-1,0)$ `back`, $(0,-1)$ `left`, $(0,1)$ `right`, $(1,-1)$ `forward_left`, $(1,1)$ `forward_right`, $(-1,-1)$ `back_left`, $(-1,1)$ `back_right` — for $j_k = 0$; for $j_k = 1$ the same names prefixed `jump_`, and `jump` for $(0, 0)$ (17 names);
- sneak: `sneak` if $\texttt{left.shift} \in K_k$, else none ([U]: its own channel, so sneaking while walking counts two actions);
- attack: `attack` if $0 \in B_k$, else none; use: `use` if $1 \in B_k$, else none.

(2) Effects, $k = 0..T-2$, each true iff $\exists x: \sigma_{k+1}[x] > \sigma_k[x]$ over: move — $x$ = `minecraft.custom:minecraft.jump`; attack — the keys starting with `minecraft.mine_block:`, and `minecraft.custom:minecraft.damage_dealt`, `…damage_dealt_absorbed`, `…damage_dealt_resisted`; use — the keys starting with `minecraft.use_item:`; sneak has none (its effect, leaving a mount, lies in the intervals excluded by (11)).

(3) Smoothing of the move values, in this order: (a) every maximal run of world lines with value none, shorter than $r_{min}$, whose neighbouring lines (just before and just after it) are world lines with one and the same non-none value $v$, takes $v$; (b) then, in line order, every run with $j = 0$ shorter than $r_{min}$ whose neighbouring lines are world lines with non-none values takes the value of the line before it (as already smoothed).

(4) Splits, per channel: a run $[p, q]$ is split after the lines $k \in [p, q-1]$ with $e^g_k$ (for move only when $j_k = 1$; sneak has no effect): split lines are dropped from the front while the first piece $[p, k_1]$ is shorter than $r_{min}$, and from the back while the last piece $[k_m + 1, q]$ is shorter than $r_{min}$; the remaining split lines cut $[p, q]$ into pieces; each piece $[p', q']$ is a segment $(v, p', q')$. An effect outside a run splits nothing.

(5) Hotbar: for every $k \le T - 2$ with $W_k$ and $h_{k+1} \ne h_k$, $\delta = (h_{k+1} - h_k) \bmod 9$: $\delta$ segments (`hotbar_right`, $k$, $k$) if $\delta \le 4$, else $9 - \delta$ segments (`hotbar_left`, $k$, $k$).

(6) Inventory: for every $k$ with $W_k$ and $\texttt{e} \in N_k$: the segment (`open_inventory`, $k$, $k$).

(7) Turns between the states $a < b$: $\Delta\psi = \sum_{k=a}^{b-1} \operatorname{wrap}(\psi_{k+1} - \psi_k)$, $\operatorname{wrap}(x) = x - 360 \lceil (x - 180) / 360 \rceil \in (-180, 180]$; $n_\psi = \lfloor |\Delta\psi| / \rho + \tfrac12 \rfloor$ turns `turn_right` if $\Delta\psi > 0$, else `turn_left`; $n_\theta = \lfloor |\theta_b - \theta_a| / \rho + \tfrac12 \rfloor$ turns `turn_down` if $\theta_b > \theta_a$, else `turn_up`; float64.

(8) Actions of the interval: the actions of $S_I$, in the order of $S$, then the $n_\psi$ and the $n_\theta$ turns of (7); $n_{world}(I) = |S_I| + n_\psi + n_\theta$, its length. The order of $S$: by first line, then channel in the order move, sneak, attack, use, hotbar, inventory.

(9) Modes [G]: $C_3$ = {`a`, `b`, `c`, `d`, `f`, `g`, `h`, `i`, `n`, `p`, `q`, `t`, `f4`}; start $O_0$, $H_0$, $V_0$ given by the consumer ($O_0 = H_0$ = false, $V_0 = 0$ when the episode starts a session, (13); otherwise $O_0$ read from frame 0 by block perception-data, $H_0$ = false, $V_0 = 0$), $\eta$ = false; for $k = 0..T-1$ in order:
- if $W_k$, $\texttt{f3} \in K_k$ and $N_k \cap C_3 \ne \emptyset$: $\eta$ = true;
- if $k \ge 1$, $\texttt{f3} \in K_{k-1}$, $\texttt{f3} \notin K_k$ and $G_{k-1}$ (release): if $\eta$, $\eta$ = false; else an overlay toggle at frame $k + 1$ ([R3], measured);
- if $W_k$ and $\texttt{f1} \in N_k$: a HUD toggle at frame $k + 2$ (by analogy with the F3 release, handled by the same key listener; no F1 in [R3]);
- if $G_k$ and $\texttt{f5} \in N_k$: a view step at frame $k + 1$ (by analogy with `e`, processed by the same key-binding loop; no F5 in [R3]).
$O_t = O_0 \oplus (\#\text{overlay toggles at frames} \le t \bmod 2)$, $H_t$ likewise, $V_t = (V_0 + \#\text{view steps at frames} \le t) \bmod 3$; $\Omega_t = O_t \vee H_t \vee V_t \ne 0$, $t = 0..T-1$, on world and GUI frames alike.

(10) Foreign input: a line $k$ with $W_k$, $\texttt{f3} \notin K_k$ and $N_k \cap \{\texttt{q}, \texttt{f}\} \ne \emptyset$ ([G] `:365-366`: with F3 held they are combos, nothing is dropped or swapped).

(11) Riding: a line $k \le T - 2$ with $\sigma_{k+1}[x] > \sigma_k[x]$ for some $x \in$ {`minecraft.custom:minecraft.boat_one_cm`, `…horse_one_cm`, `…pig_one_cm`, `…minecart_one_cm`, `…strider_one_cm`}.

(12) Use by the consumers: a state whose frame has $\Omega$, and an interval whose $I$ contains a line of (10) or (11), are not used, and neither is any $\phi$ window or $P$ history that contains them (as $n = 0$ in Q G-14).

(13) Session start: the episode starts a session iff `stats` of $\ell_0$ is empty ([R3]: empty in lines 0–4 of the 3 such episodes, non-empty in line 0 of the others).

## Interface
- Module `src/mcpref/vpt/actions.py`:
  - `WORLD_ACTIONS: tuple[str, ...]` — the 27 world actions in this order: `forward`, `back`, `left`, `right`, `forward_left`, `forward_right`, `back_left`, `back_right`, `jump`, `jump_forward`, `jump_back`, `jump_left`, `jump_right`, `jump_forward_left`, `jump_forward_right`, `jump_back_left`, `jump_back_right`, `sneak`, `turn_left`, `turn_right`, `turn_up`, `turn_down`, `attack`, `use`, `hotbar_left`, `hotbar_right`, `open_inventory`.
  - `ActionsConfig`; `def load_config(path: Path) -> ActionsConfig`.
  - `@dataclass(frozen=True) class Segment`: `action: str`, `first: int`, `last: int`.
  - `def segments(lines: Sequence[Mapping], r_min: int) -> list[Segment]` — $S$ by (1)–(6), in the order of (8).
  - `def interval_actions(segs: Sequence[Segment], lines: Sequence[Mapping], a: int, b: int, cut: Collection[int], turn_deg: float) -> list[str]` — (7), (8): $I = \{a, \dots, b-1\} \setminus$ `cut`; the actions of `segs` with a line in $I$, in the order of `segs`, then the turns.
  - `def mode_frames(lines: Sequence[Mapping], start: tuple[bool, bool, int]) -> np.ndarray` — (9): $\Omega_0..\Omega_{T-1}$, `bool`; `start` = $(O_0, H_0, V_0)$.
  - `def starts_session(lines: Sequence[Mapping]) -> bool` — (13).
  - `def foreign_lines(lines: Sequence[Mapping]) -> np.ndarray` — (10), `bool` per line; `def riding_lines(lines: Sequence[Mapping]) -> np.ndarray` — (11), `bool` per line.
- Domain: recorder versions ≥ 6.12 and options with `key_key.forward` = `key.keyboard.w`, `key_key.back` = `key.keyboard.s`, `key_key.left` = `key.keyboard.a`, `key_key.right` = `key.keyboard.d`, `key_key.jump` = `key.keyboard.space`, `key_key.inventory` = `key.keyboard.e`, `key_key.drop` = `key.keyboard.q`, `key_key.swapOffhand` = `key.keyboard.f`, `key_key.togglePerspective` = `key.keyboard.f5`, `key_key.sneak` = `key.keyboard.left.shift`, `key_key.attack` = `key.mouse.left`, `key_key.use` = `key.mouse.right`; the consumers check them in their `required_options` (an episode that differs gets the reason `options`, as in `specs/vpt-gui-frames.md` (2)).
- Callers: blocks perception-data and preference-pool call `segments`, `interval_actions`, `starts_session`, `mode_frames` (with the start of (9)), `foreign_lines`, `riding_lines` and apply (12); block vpt-gui-actions owns the GUI lines (`isGuiOpen` true), including `q`, `f`, `e` and clicks there, and adds its actions to $n$.
- Consumes: parsed `.jsonl` lines (`specs/vpt-gui-frames.md`).
- Produces: nothing on disk.
- No progress bar: no loop over work items (one episode's lines in memory).
- `specs/00-overview.md`: block 20 vpt-actions; $A$ gains the eight diagonal actions and `sneak` (27 world actions); package map `vpt` → also P1, P2, P5 data → this spec.
- Starting config values: `turn_deg` = 10, `r_min` = 3.
- Tests: hand-made line sequences for each rule: labels incl. W+S and A+D cancelling and the 17 names; sneak runs, also inside a move run; (3a) a 1-line release inside a hold, (3b) a 1-line combination between two others, an isolated short tap kept, a short jump run kept; runs ended by a GUI line and by a value change; splits by an effect inside a run, none on the last line or outside a run, jump splits only with $j = 1$; damage and mining both split attack; (4) short first and last pieces merged, a short middle piece kept; hotbar 0 → 8 (`hotbar_left` × 1), 0 → 4 (right × 4), 0 → 5 (left × 4), no segment on a GUI line; `open_inventory`; (7) exactly half a turn, a 2880° jump between two lines, a real 170° flick, pitch sign; (8) a segment spanning three intervals counted in each, `cut` lines, order; (9) toggles one frame after the release, a release inside the player inventory toggles, inside a chest does not, the combo flag (F3+A: no toggle; a flag set in the world survives a release inside a chest), F1 at $k + 2$, F5 cycling, start values; (10) with and without F3 held; (11); (13); the config loader (types, a boolean `r_min`, `turn_deg` = 0).

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (7) | `math.floor`, `math.ceil` | Python 3.10 stdlib | — | — |
| config | `yaml.safe_load` | PyYAML 6.0.3 | — | as in `specs/gui-screens.md` |

## Out of scope
- Lines with a GUI open (block vpt-gui-actions); the grid of states, cut spans and the composition of $n$ (block perception-data).
- Sprint, chat, pause and every other key not named above ([U]: ignored, or a screen of block vpt-gui-actions); keys that change state within one record period (F3 released and pressed again between two records: no case in [R3]).
- F3 combos and their effects (chunk reload, hitboxes, chunk borders, render distance, resource reload): ignored ([U]).
- Auto-jump: the human data has none, the agent has it; human jumps count as jumps.
- The agent's own turn size: block env decides whether it reads `turn_deg`.
- Frames: nothing here reads a frame; the overlay detector of (9) belongs to block perception-data.
- Sneak's effect on movement (slower, no falling off edges): sneak counts as its own action; the agent sneaks or moves within one interval, not both.
